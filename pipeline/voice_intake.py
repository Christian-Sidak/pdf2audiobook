"""Streamlined voice + room-tone intake.

Point at a YouTube URL (or local audio) and it auto-extracts:
  - a clean, sentence-bounded ~25s speech reference (a stronger voiceprint
    than a short clip; Ware's 30s beat Maya's 8.5s), and
  - a same-session room-tone sample (the clean non-speech air the pause
    harvester needs).

Both are banked into voices/<name>.wav and voices/<name>.roomtone.wav and
validated (warns if the session lacks usable clean air).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from rich.console import Console

from pipeline.config import ROOT, SAMPLE_RATE

console = Console()

YTDLP = os.environ.get("YTDLP", "yt-dlp")  # override with YTDLP=/path/to/yt-dlp
REF_SECONDS = 25.0
TONE_SECONDS = 3.0


def _ytdlp() -> str:
    return YTDLP


def fetch_audio(source: str, start: str | None, end: str | None) -> Path:
    """Return a 24kHz mono WAV of the source (URL via yt-dlp, else local),
    optionally trimmed to [start, end] (MM:SS or seconds)."""
    tmp = Path(tempfile.mkdtemp(prefix="voice_grab_"))
    raw = tmp / "raw.wav"
    if source.startswith(("http://", "https://", "www.")):
        section = None
        if start or end:
            section = f"*{start or '0'}-{end or 'inf'}"
        cmd = [_ytdlp(), "-x", "--audio-format", "wav", "-o", str(tmp / "dl.%(ext)s")]
        if section:
            cmd += ["--download-sections", section]
        cmd.append(source)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {r.stderr[-300:]}")
        dl = next(tmp.glob("dl.*"))
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(dl),
                        "-ac", "1", "-ar", str(SAMPLE_RATE), str(raw)], check=True)
    else:
        trim = ([] if not start else ["-ss", start]) + ([] if not end else ["-to", end])
        subprocess.run(["ffmpeg", "-y", "-v", "error", *trim, "-i", source,
                        "-ac", "1", "-ar", str(SAMPLE_RATE), str(raw)], check=True)
    return raw


def auto_reference(wav: Path, seconds: float = REF_SECONDS) -> tuple[float, float]:
    """Pick a sentence-bounded, gap-free ~`seconds` window using Whisper
    segment boundaries. Prefers continuous speech away from the clip edges."""
    import mlx_whisper

    from evals.audioutil import WHISPER_MODEL

    r = mlx_whisper.transcribe(str(wav), path_or_hf_repo=WHISPER_MODEL)
    segs = [s for s in r["segments"] if s["text"].strip()]
    if not segs:
        return (0.0, seconds)

    # Greedy: for each segment start, extend through consecutive segments until
    # we reach ~seconds, rejecting windows with a >0.8s internal gap (likely a
    # speaker change or edit). Score by closeness to target length.
    best, best_score = None, -1.0
    for i in range(len(segs)):
        j, gap_ok = i, True
        while j + 1 < len(segs) and segs[j]["end"] - segs[i]["start"] < seconds:
            if segs[j + 1]["start"] - segs[j]["end"] > 0.8:
                gap_ok = False
                break
            j += 1
        length = segs[j]["end"] - segs[i]["start"]
        if not gap_ok or length < seconds * 0.6:
            continue
        # prefer target length and mid-clip position
        score = -abs(length - seconds) - 0.5 * (i == 0)
        if score > best_score:
            best, best_score = (segs[i]["start"], segs[j]["end"]), score
    return best or (segs[0]["start"], min(segs[-1]["end"], segs[0]["start"] + seconds))


def auto_roomtone(wav: Path, want: float = TONE_SECONDS) -> tuple[float, float] | None:
    """Longest clean non-speech span (silero-VAD), padded away from speech."""
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    audio, sr = sf.read(str(wav), dtype="float32")
    model = load_silero_vad()
    vad_sr = 16000
    if sr != vad_sr:
        import torchaudio.functional as AF
        w16 = AF.resample(torch.from_numpy(audio), sr, vad_sr)
    else:
        w16 = torch.from_numpy(audio)
    speech = get_speech_timestamps(w16, model, sampling_rate=vad_sr)
    dur = len(audio) / sr
    bounds = [0.0] + [t for s in speech for t in (s["start"] / vad_sr, s["end"] / vad_sr)] + [dur]
    gaps = [(bounds[k], bounds[k + 1]) for k in range(1, len(bounds) - 1, 2)]
    guard = 0.25
    gaps = [(s + guard, e - guard) for s, e in gaps if (e - guard) - (s + guard) > 0.4]
    if not gaps:
        return None
    s, e = max(gaps, key=lambda g: g[1] - g[0])
    mid = (s + e) / 2
    return (max(s, mid - want / 2), min(e, mid + want / 2))


PICK_SCHEMA = {
    "type": "object",
    "properties": {
        "start": {"type": "number"},
        "end": {"type": "number"},
        "reason": {"type": "string"},
        "alternates": {"type": "array", "items": {
            "type": "object",
            "properties": {"start": {"type": "number"}, "end": {"type": "number"}},
            "required": ["start", "end"]}},
    },
    "required": ["start", "end", "reason", "alternates"],
}

PICK_PROMPT = """You are choosing the IDEAL 25-40 second window from a recording to serve \
as a voice-cloning reference for audiobook narration. The TTS model will imitate \
everything in the window: the speaker's fluency, pacing, and prosody included. Below is \
the timestamped transcript.

Choose the window with:
- one single speaker throughout, in fluent CONNECTED prose: complete sentences, \
even flowing delivery, the kind of stretch that would read beautifully on the page
- NO stammers, false starts, repeated words ("very, very"), self-corrections, or \
trailing rewrites ("in that... at different, by turns")
- minimal filler ("you know", "I mean", "sort of", "um")
- no second voice: no interviewer questions, no back-and-forth, no embedded clips \
from films or shows (acted dialogue, emotional outbursts)
- calm declarative register, not shouting, laughing, or performing characters
- window boundaries at sentence starts and ends

TRANSCRIPT (seconds):
{transcript}

Return JSON: {{"start": <s>, "end": <s>, "reason": "one sentence on why this stretch", \
"alternates": [two backup windows as {{"start","end"}}]}}"""


def llm_pick_window(segments: list[dict], model: str | None = None) -> dict | None:
    """LLM chooses the reference window from a timestamped transcript.
    Fluency and prosody are judgment calls: acoustic gap-picking cannot rank
    them, so the model reads the transcript and picks; deterministic audits
    validate afterward. Returns {start, end, reason, alternates} or None when
    Ollama is unreachable (caller falls back to the acoustic picker)."""
    from pipeline.config import CFG
    from pipeline.ollama_client import OllamaError, chat_json

    model = model or CFG["narration"]["judge_model"]
    lines = "\n".join(f"{s['start']:7.1f}-{s['end']:7.1f}  {s['text'].strip()}"
                      for s in segments)
    try:
        r = chat_json(model, [{"role": "user", "content":
                               PICK_PROMPT.format(transcript=lines[:24000])}],
                      PICK_SCHEMA, temperature=0.0, num_ctx=16384)
    except OllamaError:
        return None
    try:
        if float(r["end"]) - float(r["start"]) < 15:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return r


def audit_reference(ref_path: Path) -> list[str]:
    """Post-bank QC on a voice reference. Returns human-readable warnings.

    - pitch spread: windowed median F0; p90-p10 spread over 0.35x the median
      means the window likely contains a second speaker (an interviewer bleed
      polluted a banked reference exactly this way) OR in-character dialogue,
      which also weakens the voiceprint. Either way: re-window.
    - onset blob: a short energy burst then a gap before speech (breath/grunt
      or a clipped word onset) is reproduced by the clone at take starts.
    Noisy vintage sources (78rpm) break the pitch tracker; treat warnings on
    those as advisory."""
    import librosa
    import numpy as np
    import soundfile as sf

    warnings: list[str] = []
    a, sr = sf.read(str(ref_path), dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    dur = len(a) / sr
    body = float(np.sqrt((a ** 2).mean())) or 1e-12

    meds = []
    for st in np.arange(0, dur - 2.0, 1.0):
        seg = a[int(st * sr):int((st + 2.0) * sr)]
        f0, _, _ = librosa.pyin(seg, fmin=60, fmax=400, sr=sr, frame_length=1024)
        f0v = f0[~np.isnan(f0)]
        if len(f0v) > 5:
            meds.append(float(np.median(f0v)))
    if len(meds) > 3:
        meds_a = np.array(meds)
        spread = (np.percentile(meds_a, 90) - np.percentile(meds_a, 10)) / np.median(meds_a)
        if spread > 0.35:
            warnings.append(
                f"pitch spread {spread:.2f} (>0.35): possible second speaker or "
                "in-character dialogue — LISTEN and re-window if polluted")

    n = int(0.02 * sr)
    nf = len(a) // n
    if nf > 30:
        act = np.sqrt((a[:nf * n].reshape(nf, n) ** 2).mean(1)) > 0.1 * body
        regs, s0 = [], None
        for i, v in enumerate(act):
            if v and s0 is None:
                s0 = i
            if not v and s0 is not None:
                regs.append((s0 * 0.02, i * 0.02))
                s0 = None
            if len(regs) >= 2:
                break
        if len(regs) >= 2:
            (s1, e1), (s2, _) = regs[0], regs[1]
            if s1 < 0.15 and (e1 - s1) <= 0.40 and (s2 - e1) >= 0.20:
                warnings.append(
                    f"onset blob: {e1 - s1:.2f}s burst then {s2 - e1:.2f}s gap at the "
                    "head — the clone will reproduce it at take starts; trim the head")
    return warnings


def grab(name: str, source: str, start: str | None = None, end: str | None = None,
         ref_seconds: float = REF_SECONDS) -> bool:
    raw = fetch_audio(source, start, end)
    import soundfile as sf

    ref_dir = ROOT / "voices"
    ref_dir.mkdir(exist_ok=True)
    ref_path = ref_dir / f"{name}.wav"
    tone_path = ref_dir / f"{name}.roomtone.wav"

    # LLM window selection: fluency and prosody are judgment calls the
    # acoustic gap-picker cannot rank. Whisper supplies the timestamped
    # transcript, the judge model picks the ideal stretch, the acoustic
    # picker sentence-snaps inside it, and audit_reference validates. The
    # LLM's alternate windows serve as retries when the audit complains.
    import mlx_whisper

    from evals.audioutil import WHISPER_MODEL
    tr = mlx_whisper.transcribe(str(raw), path_or_hf_repo=WHISPER_MODEL)
    pick = llm_pick_window(tr["segments"])
    windows: list[tuple[float, float] | None] = [None]
    if pick:
        console.print(f"[cyan]llm window:[/cyan] {pick['start']:.0f}-{pick['end']:.0f}s "
                      f"[dim]({pick['reason'][:100]})[/dim]")
        windows = ([(float(pick["start"]), float(pick["end"]))]
                   + [(float(a["start"]), float(a["end"]))
                      for a in pick.get("alternates", [])[:2]]
                   + [None])  # final fallback: acoustic pick over the whole clip

    warnings: list[str] = []
    for w in windows:
        pick_src = raw
        if w is not None:
            pick_src = raw.parent / "picked.wav"
            subprocess.run(["ffmpeg", "-y", "-v", "error",
                            "-ss", str(max(0.0, w[0] - 0.5)), "-to", str(w[1] + 0.5),
                            "-i", str(raw), str(pick_src)], check=True)
        rs, re = auto_reference(pick_src, ref_seconds)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(rs), "-to", str(re),
                        "-i", str(pick_src), str(ref_path)], check=True)
        warnings = audit_reference(ref_path)
        if not warnings:
            break
        if w is not None:
            console.print(f"[yellow]window {w[0]:.0f}-{w[1]:.0f}s failed audit "
                          f"({warnings[0][:60]}...), trying next[/yellow]")
    console.print(f"[green]reference:[/green] {re - rs:.1f}s sentence-bounded speech "
                  f"[dim](clip {rs:.0f}-{re:.0f}s of picked window)[/dim]")

    tone = auto_roomtone(raw)
    if tone:
        ts, te = tone
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(ts), "-to", str(te),
                        "-i", str(raw), str(tone_path)], check=True)
        # Validate harvestable air.
        from pipeline.roomtone import _vad_nonspeech_pool
        pool, _ = _vad_nonspeech_pool(str(tone_path), 500)
        air = len(pool) * 0.06
        if air >= 1.0:
            console.print(f"[green]room tone:[/green] {te - ts:.1f}s, {air:.1f}s clean air")
        else:
            console.print(f"[yellow]room tone thin ({air:.1f}s clean air)[/yellow] "
                          f"— pauses may fall back to synthetic tone")
    else:
        tone_path.unlink(missing_ok=True)
        console.print("[yellow]no clean non-speech span found[/yellow] — no room-tone "
                      "sidecar; pauses will use synthetic tone")

    shutil.rmtree(raw.parent, ignore_errors=True)
    for msg in warnings:
        console.print(f"[red bold]REF QC:[/red bold] [yellow]{msg}[/yellow]")
    console.print(f"[bold green]banked {name}[/bold green] -> {ref_path.name}"
                  + (f" + {tone_path.name}" if tone_path.exists() else ""))
    return True


DISTILL_TEXT = (  # public-domain narration (Patterson, 1925) — phonetically broad
    "At the time of my arrival, railhead had just reached Tsavo, about one hundred and "
    "thirty miles from the coast. Here it was found that a river crossed the route, fed "
    "from the everlasting snows of Kilimanjaro. The bridge had to be built, and the camps "
    "moved steadily forward through the wilderness, mile after mile, under a sun that "
    "showed no mercy to man or beast.")


def distill_reference(name: str, text: str | None = None, n: int = 6) -> bool:
    """Replace a banked real-audio reference with a cleaner GENERATED one.

    Synthesizes n candidate takes of neutral narration with the fresh clone,
    QCs each (audit_reference; whisper transcript must start and end on the
    target text's phrases — never compare word counts, ASR rewrites numerals;
    speaker similarity back to the real audio), snips any hallucinated
    preamble at the whisper word boundary, and banks the winner as
    voices/<name>.wav. The real audio survives as <name>.orig.wav and the
    spoken text as <name>.ref_text.txt.

    Why: a distilled reference has no room noise, no interviewer bleed, and —
    because the text is chosen — provably ends on a complete sentence, which
    prevents the continuation-preamble failure a mid-thought reference causes.
    """
    import re

    import mlx_whisper
    import numpy as np
    import soundfile as sf

    from evals.audioutil import WHISPER_MODEL
    from pipeline.s5_render import SAMPLE_RATE, _make_engine

    text = text or DISTILL_TEXT
    ref = Path("voices") / f"{name}.wav"
    if not ref.exists():
        console.print(f"[red]no banked reference voices/{name}.wav[/red]")
        return False
    first = " ".join(text.split()[:3]).lower()
    last = " ".join(text.split()[-3:]).rstrip(".!?").lower()

    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        enc = VoiceEncoder(verbose=False)
        real_emb = enc.embed_utterance(preprocess_wav(str(ref)))
    except Exception:
        enc = None

    eng = _make_engine("qwen3tts", {"reference_wav": str(ref),
                                    "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                    "post": "fade3ms"})
    tmp = Path(tempfile.mkdtemp(prefix="distill_"))
    best = None  # (rank tuple, path)
    for i in range(n):
        a = eng.synthesize(text)
        p = tmp / f"cand{i}.wav"
        sf.write(p, a, SAMPLE_RATE)
        r = mlx_whisper.transcribe(str(p), path_or_hf_repo=WHISPER_MODEL,
                                   word_timestamps=True)
        heard = re.sub(r"[^a-z' ]", "", r["text"].lower())
        if last not in heard:
            console.print(f"  cand{i}: dropped (truncated or garbled ending)")
            continue
        # Snip any preamble before the target text's first word.
        words = [w for s in r["segments"] for w in s.get("words", [])]
        toks = first.split()
        onset = 0.0
        for j in range(len(words) - len(toks) + 1):
            if all(toks[k] in words[j + k]["word"].lower() for k in range(len(toks))):
                onset = words[j]["start"]
                break
        else:
            console.print(f"  cand{i}: dropped (target text start not found)")
            continue
        if onset > 0.05:
            a2, _ = sf.read(p, dtype="float32")
            a2 = a2[int(max(0.0, onset - 0.07) * SAMPLE_RATE):].copy()
            fade = int(0.010 * SAMPLE_RATE)
            a2[:fade] *= np.linspace(0, 1, fade)
            sf.write(p, a2, SAMPLE_RATE)
        warns = audit_reference(p)
        sim = float(np.dot(real_emb, enc.embed_utterance(preprocess_wav(str(p))))) \
            if enc else 0.0
        console.print(f"  cand{i}: snip {onset:.2f}s, warns {len(warns)}, sim {sim:.3f}")
        rank = (len(warns), -sim)
        if best is None or rank < best[0]:
            best = (rank, p, sim)
    if best is None:
        console.print("[red]no candidate survived QC — keeping the real reference[/red]")
        return False

    a, sr = sf.read(best[1], dtype="float32")
    # Open with ~0.5s of the take's own quietest air (tight onsets teach clicks).
    f = int(0.02 * sr)
    frames = np.sqrt((a[:len(a) // f * f].reshape(-1, f) ** 2).mean(1))
    w = 25
    q = min(range(max(1, len(frames) - w)), key=lambda i: frames[i:i + w].mean())
    airhead = a[q * f:(q + w) * f] * np.linspace(0.3, 1.0, w * f)
    out = np.concatenate([airhead, a])

    orig = ref.with_suffix(".orig.wav")
    if not orig.exists():
        shutil.copy(ref, orig)
    sf.write(ref, out, sr)
    ref.with_suffix(".ref_text.txt").write_text(text)
    # A distilled ref rarely has harvestable air; a stale real-audio tone
    # sidecar under synthetic-clean takes is worse than the synth fallback.
    tone = ref.with_suffix(".roomtone.wav")
    if tone.exists():
        tone.rename(ref.with_suffix(".roomtone.orig.wav"))
    shutil.rmtree(tmp, ignore_errors=True)
    console.print(f"[bold green]distilled {name}[/bold green] -> {ref.name} "
                  f"({len(out)/sr:.1f}s, sim {best[2]:.3f}; real audio kept as {orig.name})")
    return True
