---
name: grab-voice
description: Bank a TTS voice reference (plus room tone) for a speaker. Given a person's name, web-search for a good long-form interview or talk on YouTube, pick a clean single-speaker section, and grab it with `python3 main.py voice-grab`. Given a YouTube URL, grab from it directly. Always QC the banked reference by transcribing it.
---

# grab-voice

Bank a ~25s voice reference and same-session room tone into `voices/<name>.wav`
(+ `voices/<name>.roomtone.wav`) using this repo's `voice-grab` CLI.

## Input forms

- `/grab-voice theodore_roosevelt` — a speaker name: find a source video yourself (Step 1).
- `/grab-voice https://youtube.com/watch?v=...` — a URL: skip to Step 2. If a name
  is also given, use it as the bank name; otherwise derive a `snake_case` name from
  the video's subject (ask only if genuinely ambiguous).
- Optional user hints like a timestamp ("around 8:30") mean the user has already
  scouted that spot — trust it as the section to target.

## Step 1 — Find a source video (name given)

Use WebSearch/yt-dlp to find candidate YouTube videos. Prefer, in order:

1. **Unedited long-form interviews or lectures** (30+ min): natural pauses give
   usable room tone, and long guest monologues are easy to isolate.
2. Produced interviews (talk shows, podcasts with music beds): fine for the voice
   reference but room tone is usually unusable (music under everything).
3. Avoid: compilations, clips with background scores throughout, audiobook/
   narration samples of OTHER people's writing (rights-muddy and often processed),
   anything under ~3 min.

Check duration and channel with:
`yt-dlp --print "%(id)s %(duration)s %(channel)s %(title)s" --skip-download <url>`
The channel tells you who the HOST is — you need that to tell host and guest
apart later.

## Step 2 — Scout a single-speaker window (critical)

Interviews contain two voices, and Whisper does not diarize. A reference polluted
with even short host backchannels ("Right", "Really?", "Yeah") degrades the
voiceprint. The auto-picker in `voice_intake.py` rejects >0.8s gaps but cannot
detect speaker changes, so never trust a blind grab of interview audio.

Scout first: download and transcribe a few minutes with segment timestamps, e.g.

```python
from pipeline.voice_intake import fetch_audio
import mlx_whisper
from evals.audioutil import WHISPER_MODEL
wav = fetch_audio(url, "2:00", "10:00")   # mid-video; skip the intro
r = mlx_whisper.transcribe(str(wav), path_or_hf_repo=WHISPER_MODEL)
for s in r["segments"]:
    print(f"{s['start']:7.1f}-{s['end']:7.1f}  {s['text'].strip()}")
```

Then identify who is who from CONTENT, not position:

- First-person biography that matches the target ("when I wrote...", "my father...")
  = the guest. Questions and praise directed at "you" = the host.
- Watch for traps: a host can monologue at length too (self-deprecation, framing).
  Cross-check against known facts about each person (accent jokes, ethnicity,
  nationality, life events mentioned).
- Short interjection segments (<1s: "Right.", "Really?", "Okay.") mark the OTHER
  speaker backchanneling — treat them as window boundaries.
- Repeated identical segments over and over = Whisper hallucinating on music or
  silence, not speech. Exclude that region entirely.

Pick a 25-40s stretch that is purely the target speaker, bounded by their sentence
starts/ends.

## Step 3 — Grab

```bash
python3 main.py voice-grab <snake_case_name> "<url>" --start <sec|MM:SS> --end <sec|MM:SS>
```

Note: yt-dlp section downloads snap to keyframes, so the actual clip may start a
few seconds before `--start`. Keep your window's margins inside pure-target
speech when possible.

## Step 2b — Prefer plain speech over performance

For narration cloning, a window of NEUTRAL connected speech beats an expressive
one. Avoid: in-character dialogue (an audiobook narrator doing voices), shouting,
recitation with dramatic swings. These are single-speaker but teach the clone an
unstable register. The pitch audit below flags them the same way it flags a
second speaker — both mean "re-window".

## Step 4 — QC (never skip)

`voice-grab` now runs `pipeline.voice_intake.audit_reference` automatically after
banking and prints `REF QC:` warnings for (a) pitch spread > 0.35 (second speaker
or in-character dialogue — an interviewer bleed polluted a banked reference this
way once and an entire book render had to be discarded) and (b) an onset
blob-then-gap at the head (the clone reproduces it at take starts). NEVER bank a
voice with unresolved REF QC warnings unless the source is antique shellac (the
pitch tracker is unreliable there). Then ALSO do the transcript QC below:

Transcribe the banked file and check:

```python
import mlx_whisper, soundfile as sf, numpy as np
from evals.audioutil import WHISPER_MODEL
a, sr = sf.read(f"voices/{name}.wav")
print(len(a)/sr, "s, rms", 20*np.log10(np.sqrt(np.mean(a**2))+1e-12), "dB")
print(mlx_whisper.transcribe(f"voices/{name}.wav", path_or_hf_repo=WHISPER_MODEL)["text"])
```

- **Single speaker**: no questions/answers alternating, no backchannels from a
  second voice. If polluted, go back to Step 2 with a different window.
- **Right speaker**: content reads as the target (first-person, consistent with
  their biography). If it reads like the host, re-window.
- **Duration** 18-30s (longer beats shorter: a 30s ref has consistently outperformed sub-10s ones).
- **Level**: RMS anywhere in roughly -18 to -37 dB is fine — mastering normalizes
  takes to -20 dBFS downstream, so do NOT gain-adjust the reference.
- **Room tone**: "thin (0.0s clean air)" is common on produced interviews and OK —
  pauses fall back to synthetic tone. Only chase a better tone source if the user
  asked for authentic room tone specifically.

## Step 4b — Write the transcript sidecar (mandatory)

Write `voices/<name>.ref_text.txt` by hand with the exact words spoken in the
banked reference, using the whisper transcript as a starting point but
correcting it by ear. Do NOT leave this to the engine's whisper fallback: on
a reference with trailing air, whisper invents a phrase over the silence, and
a transcript that claims words the audio lacks makes the clone speak them as
a preamble on every take (Carnegie/Spader 2026-09-04, 13 hours lost). Stage 5
now runs a preflight (`pipeline/voice_preflight.py`) that halts a build on
this, but the sidecar is what makes it pass.

## Step 5 — Report

Tell the user: source video (title + URL), the window used, the QC transcript
snippet proving it's the right speaker alone, and whether room tone was usable
or will fall back to synthetic.

## Step 6 — Distill (optional but recommended)

A generated reference beats a real one: no room noise, no interviewer, no
mumble. After banking real audio, synthesize ~6 candidate takes of neutral
public-domain narration (~70 words) with the fresh clone, then QC each:

- `pipeline.voice_intake.audit_reference` (pitch spread, onset blob)
- whisper transcript must START at the target text's first phrase and END at
  its last (verify by phrase, not word count — whisper writes "one hundred
  and thirty" as "130")
- resemblyzer similarity back to the real reference (keep the highest)

Run `python3 main.py voice-distill <name>` — it generates candidates, QCs
them, snips hallucinated preambles, and banks the winner automatically.
Bank the winner as `voices/<name>.wav`, keep the real audio as
`<name>.orig.wav`, write the text to `<name>.ref_text.txt`.

Two structural rules learned the hard way:
- The reference (raw or distilled) must END on a COMPLETE SENTENCE. A ref
  cut mid-thought makes every take hallucinate a "completion" preamble.
- The reference should OPEN with ~0.5s of soft air, not tight on the first
  word — tight onsets teach the clone clicky take starts.

A distilled reference usually has too little inter-sentence air to harvest a
room-tone bed (VAD pool < 3 frames): assembly falls back to the synthetic
bed automatically, which suits a clean synthetic voice.
