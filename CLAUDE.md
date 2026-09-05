# pdf2audiobook — project conventions

- **Takes are immutable assets.** Rendered takes are content-addressed
  (`segment_hash(engine, params, text)`) and one-of-a-kind stochastic
  performances. Never delete or overwrite a take a human has approved;
  invalidate through `pipeline/take_cache.py` rejection, or render new text.
- **Language judgment goes through the local LLM; deterministic code
  validates and falls back.** Number verbalization, take review verdicts,
  title dedupe, window picking, speakability adjudication — all LLM calls
  with schema-validated output and a mechanical fallback. Don't replace them
  with regex heuristics; don't let the LLM output past validation.
- **Never rewrite an author's prose to fix TTS** — pacing, retakes, and take
  selection first. The one sanctioned exception is the stage-5 adjudicator,
  which may split sentences and respell hard words for stubborn takes.
- **Voice references must end on a complete sentence** (a mid-thought ref
  makes every take hallucinate a continuation preamble) and should open with
  ~0.5s of soft air (tight onsets teach clicky take starts). Prefer
  `main.py voice-distill` references: generated, clean, provably bounded.
  **Every cloned voice needs a hand-written `voices/<name>.ref_text.txt`.**
  The whisper fallback hallucinates over a reference's silent tail and the
  clone then speaks that phrase as a preamble on every take. Stage 5 runs
  `pipeline/voice_preflight.py` (transcript-vs-audio, LLM-judged canary
  takes) before the first take and halts on failure; don't disable it.
- **Chapter structure overrides live in `library/<book>.chapters.yaml`**
  sidecars — never hand-edit derived artifacts to fix structure.
- Skills in `.claude/skills/` document the full workflows: `grab-voice`
  (find → scout → bank → QC → distill a narrator voice), `build-audiobook`
  (PDF → M4B end to end), `check-build` (monitor/resume/triage renders),
  `tune-audiobook` (assembly-time pacing and mastering iteration), and
  `run-evals` (eval matrix and baseline discipline).
