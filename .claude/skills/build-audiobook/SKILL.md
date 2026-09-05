---
name: build-audiobook
description: Turn a PDF into a finished, QC-gated audiobook end to end — source vetting, extraction preview, chapter sidecars, cloned-voice render, QC triage, mastering, cover art, and delivery.
---

# build-audiobook

Drive the pdf2audiobook pipeline from a PDF to an M4B in Apple Books.

## 1. Vet the source

- Public-domain check for anything that leaves the machine: US rule is 95
  years from publication (in 2026, published ≤1930 is safe). archive.org and
  gutenberg.org are the source of choice; prefer scans with a text layer.
- Drop the PDF in `library/`. Copyrighted PDFs are fine for private use —
  they are gitignored — but never commit them or publish their renders.

## 2. Preview extraction

```bash
python3 main.py preview "library/<book>.pdf"
```

Inspect the chapter table and `artifacts/<book_id>/02_body.txt`. Common
defects and their fixes:

- **Front matter in the body / bad chapterization**: write a
  `library/<book>.chapters.yaml` sidecar (1-indexed start pages, optional
  `matter: front|back`). The sidecar beats every heuristic. Do NOT hand-edit
  `02_body.txt` — stage 3 reads the stage-1/2 JSON artifacts, not the txt.
- **Caption/plate junk inside sentences**: blank those page entries in
  `01_extract.json` (text + blocks), then re-run `--stages 2-3`.
- Verify with a stages-2-3 re-run until the chapter table is clean.

## 3. Render

```bash
# fast draft, preset voice
python3 main.py build "library/<book>.pdf" --fast
# cloned voice (bank one first with /grab-voice, distill with voice-distill)
python3 main.py build "library/<book>.pdf" --engine qwen3tts \
    --voice voices/<name>.wav --title "…" --author "…"
```

Run long builds in the background and watch the printed stage/QC lines.
Takes are content-hash cached: killing and relaunching a build never loses
finished work, and text edits only re-render the changed segments.

## 4. QC loop triage

Stage 5 gets a 4-attempt budget; failures feed back as re-renders, and from
attempt 2 stubborn takes get their text adjudicated for speakability by the
local LLM. Know the failure classes:

- Attempt summaries show TRUE violation counts per dimension and the retry
  loop is fed every fixable violation (the old 25-per-dimension cap made a
  14%-failure book unclearable; removed 2026-09-05). Expect attempt 1 to
  feed hundreds back on a long book; the count should fall steeply each
  attempt. A count that does not fall is a systematic fault (voice, text),
  not a backlog. The review queue and console still list at most 10 per
  dimension for readability.
- Text adjudication (the sanctioned rewrite for stubborn takes) now fires
  per segment after its second failure, not for every failure on attempt 2.
- **Whisper false positives**: homophones ("forefeet" heard "four feet"),
  proper nouns, foreign words. Verify by listening, not by transcript.
- **Onset artifacts (blob-then-gap)**: snip, don't re-roll — detect the
  pre-speech energy blob acoustically, whisper-transcribe head and tail
  (head must be contentless, tail must carry the full source text), then
  trim with a 10ms fade. Re-rolling burns attempts; snipping is exact.
- **Sub-second takes failing speaker similarity**: usually segmentation
  shrapnel upstream (standalone honorifics, punctuation-only fragments) —
  fix the narration segments, not the audio.
- If the run halts with residual flags: `python3 -m evals.review list`,
  play the flagged takes for a human verdict, then either fix targeted
  segments and re-run `--stages 5-6`, or assemble past accepted flags with
  `--keep-going`.

## 5. Master and iterate

```bash
python3 main.py reassemble <book_id>   # pause/tone/level changes only — no re-render
```

Pace: spoken wpm is gated to [130, 178]. A faithful clone of a fast talker
can exceed it; either accept deliberately (resolve the review item) or
stretch the master with sox `tempo -s 0.93` (pitch-preserving; ≤~7% is
inaudible, more smears).

## 6. Cover art

1. Generate textless art (mflux/Qwen-Image or any generator), square.
2. `export.cover.design_typography` picks a face/palette; `compose_cover`
   composites title/author/narrator (pass the narrator NAME only — the
   "Narrated by" prefix is added for you).
3. `python3 main.py cover <book_id> art.png` embeds it in the M4B + chapters.

## 7. Deliver

```bash
python3 main.py publish <book_id>    # Apple Books on this Mac
python3 -c "from export.icloud import export_icloud; ..."  # phone via iCloud/Files
```

Rights: clone voices you have rights to. Renders in the voice of a living
person stay private unless you have their consent.
