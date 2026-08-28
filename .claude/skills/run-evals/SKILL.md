---
name: run-evals
description: Run the eval matrix over the corpus, gate changes against the committed baseline, interpret regressions, and update the baseline deliberately. Use before committing any pipeline change and when adding a document to the eval corpus.
---

# run-evals

The pipeline is eval-gated: deterministic checks plus LLM-judged reviews
per stage, with a committed baseline. Treat evals like a test suite with a
golden-file discipline.

## Run

```bash
python3 -m evals.run --stages 1-3              # cheap text stages over the corpus
python3 -m evals.run --stages 4 --docs <id>    # one stage, one document
python3 -m evals.run --diff                    # gate against baseline; exit 1 on regression
python3 -m evals.run --sample-chapters 2       # cap chapters per doc for speed
python3 -m evals.run -v                        # per-check detail
```

(`python3 main.py evals ...` forwards to the same entry point.)

Stages 1-3 are fast and deterministic: run them on every pipeline change.
Stages 4-5 call the local LLM and TTS: run them targeted (`--docs`,
`--sample-chapters`) unless you changed narration or render code.

## Interpret

- A regression in `--diff` means a check that passed at the baseline now
  fails. Read the failing check's message before touching thresholds; the
  corpus assertions encode real defects seen in real books (running
  headers, hyphen breaks, footnote interleaving).
- LLM-judged checks are non-deterministic at the margin. Re-run a lone
  marginal failure once before investigating; a repeat failure is real.
- Deletion flags on a document you know is clean: inspect with
  `python3 -m evals.review show <n>`; if it is a legitimate edit (dropped
  running header, de-hyphenation), resolve with `--allow` to record it in
  that document's `deletion_allowlist` in `evals/corpus.yaml`.

## Update the baseline

```bash
python3 -m evals.run --update-baseline
```

Only after: (a) the change is intentional, (b) you have looked at every
diff line it blesses, and (c) targeted listening confirms audio-affecting
changes. Commit `evals/baselines/baseline.json` in the same commit as the
code change that moved it, with the reason in the commit message.

## Add a corpus document

Add an entry in `evals/corpus.yaml` (id, pdf path, per-stage assertions;
PDFs stay untracked). Seed assertions from a real run with
`python3 -m evals.seed`, then hand-verify each one before committing.
Golden text lives in `evals/golden/<id>/` and must be public domain.
