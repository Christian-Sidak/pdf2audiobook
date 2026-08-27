"""Agentic QC orchestrator: run stage -> run its checks -> feed fixable
violations back for bounded, unit-scoped retries -> queue residuals for
human review.

Usage:
    python -m evals.qc_loop <book_id> [--stages 1-6] [--keep-going]
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rich.console import Console

from evals import checks as checks_pkg
from evals.contracts import BookCtx, CheckResult, Violation
from evals.corpus import load_corpus
from evals.run import parse_stages, run_doc
from pipeline.run_stages import STAGE_MODULES

console = Console()

RETRY_BUDGET = {4: 2, 5: 3}  # stages 1-3 and 6 are deterministic: failure = code bug
REVIEW_QUEUE = Path(__file__).resolve().parent / "review_queue.jsonl"


@dataclass
class QCResult:
    book_id: str
    completed_stages: list[int] = field(default_factory=list)
    halted_at: int | None = None
    queued: int = 0


def enqueue_review(book_id: str, stage: int, results: dict[str, CheckResult],
                   attempts: int) -> int:
    n = 0
    with open(REVIEW_QUEUE, "a", encoding="utf-8") as f:
        for dim, r in results.items():
            if r.status not in ("fail", "error"):
                continue
            for v in (r.violations or [Violation(message=r.details.get("traceback", "error"))])[:10]:
                f.write(json.dumps({
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "book": book_id, "stage": stage, "dimension": dim,
                    "unit_id": v.unit_id, "message": v.message,
                    "context": v.context, "attempts": attempts,
                }) + "\n")
                n += 1
    return n


def qc_run(book_id: str, stages: set[int] | None = None, keep_going: bool = False,
           pdf: str | None = None, config: dict | None = None) -> QCResult:
    checks_pkg.load_all()
    corpus = load_corpus()
    try:
        doc = corpus.doc(book_id)
    except KeyError:
        if not pdf:
            raise
        from evals.corpus import DocSpec
        doc = DocSpec(id=book_id, pdf=Path(pdf), tier="full")
    book = BookCtx.for_book(doc.id, doc.pdf, config=config)
    stages = sorted(stages or set(range(1, 7)))
    result = QCResult(book_id=book_id)

    for stage in stages:
        runner = importlib.import_module(STAGE_MODULES[stage])
        console.print(f"[bold blue]QC stage {stage}[/bold blue] ({runner.__name__})")
        try:
            runner.run(book)
        except Exception as e:
            import traceback
            console.print(f"[red bold]stage {stage} crashed:[/red bold] {e}")
            enqueue_review(book_id, stage, {"stage_crash": CheckResult(
                "stage_crash", stage, "error",
                details={"traceback": traceback.format_exc(limit=8)})}, attempts=1)
            result.halted_at = stage
            result.queued += 1
            return result

        budget = RETRY_BUDGET.get(stage, 0)
        for attempt in range(budget + 1):
            results = run_doc(doc, {stage}, corpus.defaults)
            failures = {d: r for d, r in results.items() if r.status in ("fail", "error")}
            if not failures:
                console.print(f"[green]stage {stage} checks green[/green]")
                break

            fixable = [v for r in failures.values() for v in r.violations if v.fixable]
            can_retry = fixable and attempt < budget and hasattr(runner, "retry")
            summary = ", ".join(f"{d} ({len(r.violations)})" for d, r in failures.items())
            console.print(f"[yellow]stage {stage} attempt {attempt + 1}: {summary}[/yellow]")

            if not can_retry:
                queued = enqueue_review(book_id, stage, failures, attempt + 1)
                result.queued += queued
                console.print(f"[red]stage {stage}: {queued} violations queued for review[/red]")
                if not keep_going:
                    result.halted_at = stage
                    return result
                break

            console.print(f"[dim]feeding {len(fixable)} fixable violations back to stage {stage}[/dim]")
            runner.retry(book, fixable)

        result.completed_stages.append(stage)

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("book_id")
    ap.add_argument("--stages", default="1-6")
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args()

    r = qc_run(args.book_id, parse_stages(args.stages), keep_going=args.keep_going)
    if r.halted_at:
        console.print(f"[red bold]halted at stage {r.halted_at}; "
                      f"{r.queued} items in review queue[/red bold]")
        sys.exit(1)
    console.print(f"[green bold]QC complete: stages {r.completed_stages}, "
                  f"{r.queued} review items[/green bold]")


if __name__ == "__main__":
    main()
