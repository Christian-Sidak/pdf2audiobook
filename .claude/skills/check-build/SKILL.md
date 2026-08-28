---
name: check-build
description: Check on a running or halted audiobook build. Show live progress with ETA, diagnose stalls, resume interrupted renders (takes are cached, resuming is always safe), and triage a halted build's review queue.
---

# check-build

Answer "how is the render going?" and get a stuck build moving again.

## Status

```bash
python3 main.py status                 # panel per book: take progress, pace, ETA
python3 main.py status <book>          # one book (substring match on book id)
python3 main.py status --watch         # refresh every 15s
```

The panel derives pace from takes rendered in the last 30 minutes and checks
for a live `main.py build` process. Read the states literally:

- **rendering, with ETA**: healthy. Long books are overnight jobs; do not
  restart a healthy render to "speed it up".
- **build process alive, between takes**: normal during stage-4 retries or
  TTS model load. Only investigate if it persists 15+ minutes; then look at
  the build's stdout for a stuck LLM call or a repeated retry loop.
- **no build running**: the process died or was killed. Resuming is free:
  re-run the exact same build command. Takes are content-hash cached, so it
  continues from where it stopped; never delete artifacts to "start clean".
- **all takes rendered, awaiting assembly**: run the same build command (it
  skips to stage 6) or `python3 main.py reassemble <book_id>`.

## Halted with QC flags

A build that halts at a stage gate prints `halted at stage N`. Triage:

```bash
python3 -m evals.review list           # numbered queue: book, stage, dimension
python3 -m evals.review show <n>       # full JSON for one item
```

Play the flagged takes (paths are in the item JSON) and give a human
verdict per item:

```bash
python3 -m evals.review resolve <n> --fixed     # you fixed the cause; re-run the stage
python3 -m evals.review resolve <n> --allow     # false positive; adds the context to the
                                                # doc's deletion_allowlist (--pattern to
                                                # override the auto-derived pattern)
```

Then re-run the halted stages (`--stages 5-6` typically), or accept the
remaining flags deliberately with `--keep-going`. The failure classes and
listen-and-waive judgment calls are documented in /build-audiobook step 4.

## Remote (vast.ai) builds

`scripts/vast/manage.sh status|pull` is the control plane. SSH from a Mac
drops on long-running renders; silence is not failure. Never stop an
instance on SSH silence alone; wait for the DONE sentinel or pull and
inspect artifacts.
