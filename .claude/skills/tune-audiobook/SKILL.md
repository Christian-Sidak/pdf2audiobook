---
name: tune-audiobook
description: Iterate a finished audiobook's pacing, pauses, room tone, and loudness without re-rendering a single take. Edit config.yaml knobs, reassemble (stage 6 only), listen, repeat. Use after a build sounds rushed, draggy, choppy, or hot.
---

# tune-audiobook

Everything below is an assembly-time decision. Takes are dry and immutable;
pauses, tone, and level live in stage 6, so each iteration costs seconds:

```bash
# edit config.yaml, then
python3 main.py reassemble <book_id>
# listen to output/<book_id>/chapters/, adjust, repeat
```

## The knobs (config.yaml)

Pacing feel comes from silence, not audio stretching:

- `pause_policy.sentence.gap`: the single biggest lever. 0.55s default;
  0.65 to 0.75 reads as measured or literary, 0.45 as brisk.
- `pause_policy.paragraph.before`: breath between paragraphs (0.5 default).
- `pause_policy.chapter_heading` / `section_heading`: the before/after air
  around announced headings. Raise `before` if chapters feel abrupt.
- `pause_policy.blockquote`: symmetric air that sets quotations apart.

Mastering:

- `mastering.lufs_target`: -18 LUFS default (ACX range is -23 to -18).
  Lower it if the book sounds hot next to commercial audiobooks.
- `mastering.room_tone_db`: level of the tone bed under pauses (-58 dB).
  Raise slightly (-52) if pauses sound dead against a noisy reference
  voice; the bed must be audible enough to hide the take boundaries.
- `mastering.head_tone_s` / `tail_tone_s`: leading/trailing air per chapter.
- `mastering.per_take.target_rms_dbfs`: per-take leveling before the bed.

## When reassembly is NOT enough

- **The voice itself is too fast or slow**: `tts.engines.qwen3tts.tempo`
  and `target_wpm` act at render time. Changing them re-renders (new cache
  key), so decide on a single chapter first: build with
  `--chapters 1 --target-wpm 120`, listen, then commit to the full run.
- **Small overall stretch**: sox `tempo -s 0.93` on the master is
  pitch-preserving and inaudible up to about 7 percent; beyond that it
  smears. Use it to nudge a finished book into the wpm gate rather than
  re-rendering.
- **A few bad takes**: that is QC territory, not tuning; see /check-build.

Title and author metadata survive reassembly (read back from the output
manifest). Cover art must be re-embedded after a reassemble:
`python3 main.py cover <book_id> <art.png>`.

Judge pacing by ear on real chapters, not on numbers: the spoken-wpm gate
is [130, 178], but a slow contemplative text at 135 and a field memoir at
170 can both be right.
