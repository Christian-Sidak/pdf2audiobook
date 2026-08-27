"""Matrix runner: run registered checks over the corpus, report, gate.

Usage:
    python -m evals.run                       # all docs, all stages with artifacts
    python -m evals.run --stages 1-3 --docs origins_of_the_kabbalah
    python -m evals.run --stage 4 --pass-k 3
    python -m evals.run --diff                # exit 1 on pass->fail vs baseline
    python -m evals.run --update-baseline
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from evals import checks as checks_pkg
from evals.artifacts import ArtifactSet
from evals.contracts import ArtifactMissing, CheckResult
from evals.corpus import Corpus, DocSpec, load_corpus
from pipeline.config import ARTIFACTS_DIR, ROOT

console = Console()

EVALS_DIR = Path(__file__).resolve().parent
BASELINE_PATH = EVALS_DIR / "baselines" / "baseline.json"
REPORTS_DIR = EVALS_DIR / "reports"


def parse_stages(spec: str | None) -> set[int]:
    if not spec:
        return set(range(1, 7))
    out: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _requirement_met(doc: DocSpec, req: str) -> bool:
    if req.startswith("golden."):
        p = doc.golden_path(req.split(".", 1)[1])
        return p is not None and p.exists()
    if req.startswith("expect."):
        return doc.expect.get(req.split(".", 1)[1]) is not None
    return True


def run_doc(doc: DocSpec, stages: set[int], cfg: dict) -> dict[str, CheckResult]:
    art = ArtifactSet(ARTIFACTS_DIR / doc.id)
    results: dict[str, CheckResult] = {}
    for rc in checks_pkg.REGISTRY:
        if rc.stage not in stages:
            continue
        if rc.dimension in doc.skip_checks:
            results[rc.dimension] = CheckResult.skipped(rc.dimension, rc.stage, "skip_checks in corpus.yaml")
            continue
        missing = [r for r in rc.requires if not _requirement_met(doc, r)]
        if missing:
            results[rc.dimension] = CheckResult.skipped(rc.dimension, rc.stage, f"requires {', '.join(missing)}")
            continue
        try:
            results[rc.dimension] = rc.fn(doc, art, cfg)
        except ArtifactMissing as e:
            results[rc.dimension] = CheckResult.skipped(rc.dimension, rc.stage, f"stage not run ({e.name})")
        except Exception:
            results[rc.dimension] = CheckResult(rc.dimension, rc.stage, "error",
                                                details={"traceback": traceback.format_exc(limit=6)})
    return results


def run_matrix(corpus: Corpus, stages: set[int], doc_ids: list[str] | None,
               pass_k: int | None = None, sample_chapters: int = 0) -> dict:
    docs = [d for d in corpus.documents if not doc_ids or d.id in doc_ids]
    cfg = corpus.defaults
    matrix: dict[str, dict[str, CheckResult]] = {}

    for doc in docs:
        if pass_k and pass_k > 1 and 4 in stages:
            matrix[doc.id] = _run_pass_k(doc, stages, cfg, pass_k, sample_chapters)
        else:
            matrix[doc.id] = run_doc(doc, stages, cfg)

    return {"matrix": matrix, "docs": [d.id for d in docs], "stages": sorted(stages)}


def _run_pass_k(doc: DocSpec, stages: set[int], cfg: dict, k: int,
                sample_chapters: int = 0) -> dict[str, CheckResult]:
    """pass^k for stage 4: regenerate the narration k times; a stage-4 cell
    passes only if every generation passes. Non-stage-4 checks run once.
    sample_chapters > 0 regenerates only the first N body chapters
    (deterministic subset) to bound cost."""
    from evals.contracts import BookCtx

    try:
        from pipeline import s4_narration
    except ImportError:
        res = run_doc(doc, stages, cfg)
        for dim, r in res.items():
            if r.stage == 4:
                res[dim] = CheckResult(dim, 4, "error", details={"reason": "pass-k requested but pipeline.s4_narration not available"})
        return res

    book = BookCtx.for_book(doc.id, doc.pdf)
    if sample_chapters:
        import json as _json
        chapters_file = book.artifacts_dir / "03_chapters.json"
        body = [c["id"] for c in _json.loads(chapters_file.read_text())["chapters"]
                if not c.get("front_matter")]
        book.config["only_chapters"] = body[:sample_chapters]
    merged: dict[str, CheckResult] = run_doc(doc, stages - {4}, cfg)
    for i in range(k):
        console.print(f"[dim]pass^k: generation {i + 1}/{k} for {doc.id}[/dim]")
        s4_narration.run(book)
        round_results = run_doc(doc, {4}, cfg)
        for dim, r in round_results.items():
            prev = merged.get(dim)
            if prev is None or prev.status == "pass":
                merged[dim] = r  # any non-pass round sticks
            if r.status != "pass" and prev and prev.status != "pass":
                merged[dim].violations.extend(r.violations)
        for dim, r in merged.items():
            if r.stage == 4:
                r.details.setdefault("pass_k", k)
    return merged


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def build_report(run: dict, cfg: dict, baseline: dict | None) -> dict:
    cells = {}
    summary = {"pass": 0, "fail": 0, "skip": 0, "error": 0}
    regressions = []
    for doc_id, results in run["matrix"].items():
        cells[doc_id] = {dim: r.to_dict() for dim, r in results.items()}
        for dim, r in results.items():
            summary[r.status] += 1
            key = f"{doc_id}:{dim}"
            if baseline and baseline.get(key) == "pass" and r.status in ("fail", "error"):
                regressions.append(key)
    return {
        "run_id": datetime.now().isoformat(timespec="seconds"),
        "git_rev": _git_rev(),
        "config": {k: v for k, v in cfg.items()},
        "stages": run["stages"],
        "matrix": cells,
        "summary": summary,
        "regressions": sorted(regressions),
    }


def render_terminal(run: dict, verbose: bool) -> None:
    style = {"pass": "[green]PASS[/green]", "fail": "[red]FAIL[/red]",
             "skip": "[dim]skip[/dim]", "error": "[magenta]ERR[/magenta]"}
    by_stage: dict[int, list[str]] = {}
    for results in run["matrix"].values():
        for dim, r in results.items():
            by_stage.setdefault(r.stage, [])
            if dim not in by_stage[r.stage]:
                by_stage[r.stage].append(dim)

    for stage in sorted(by_stage):
        table = Table(title=f"Stage {stage}", show_lines=False)
        table.add_column("document")
        for dim in by_stage[stage]:
            table.add_column(dim, justify="center")
        for doc_id, results in run["matrix"].items():
            row = [doc_id]
            for dim in by_stage[stage]:
                r = results.get(dim)
                row.append(style[r.status] if r else "[dim]-[/dim]")
            table.add_row(*row)
        console.print(table)

    if verbose:
        from rich.markup import escape
        for doc_id, results in run["matrix"].items():
            for dim, r in results.items():
                for v in r.violations[:10]:
                    console.print(f"  [red]{doc_id}:{dim}[/red] {escape(v.message)}"
                                  + (f" [dim]({v.unit_id})[/dim]" if v.unit_id else ""))
                if r.status == "error":
                    console.print(f"  [magenta]{doc_id}:{dim}[/magenta] {r.details.get('traceback', r.details.get('reason', ''))}")


def flatten_statuses(run: dict) -> dict[str, str]:
    return {f"{doc_id}:{dim}": r.status
            for doc_id, results in run["matrix"].items() for dim, r in results.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the eval matrix")
    ap.add_argument("--stages", "--stage", dest="stages", help="e.g. 1-3 or 4 or 2,5")
    ap.add_argument("--docs", help="comma-separated doc ids")
    ap.add_argument("--pass-k", type=int, default=None)
    ap.add_argument("--sample-chapters", type=int, default=0,
                    help="pass^k: regenerate only the first N body chapters")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--diff", action="store_true", help="gate against baseline; exit 1 on regression")
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args(argv)

    checks_pkg.load_all()
    if not checks_pkg.REGISTRY:
        console.print("[yellow]No checks registered yet.[/yellow]")

    corpus = load_corpus()
    stages = parse_stages(args.stages)
    doc_ids = args.docs.split(",") if args.docs else None

    run = run_matrix(corpus, stages, doc_ids, pass_k=args.pass_k,
                     sample_chapters=args.sample_chapters)

    baseline = None
    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text())

    report = build_report(run, corpus.defaults, baseline)
    render_terminal(run, args.verbose)

    REPORTS_DIR.mkdir(exist_ok=True)
    out_dir = REPORTS_DIR / report["run_id"].replace(":", "-")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "matrix.json").write_text(json.dumps(report, indent=2))

    s = report["summary"]
    console.print(f"\n[bold]pass {s['pass']}  fail {s['fail']}  skip {s['skip']}  error {s['error']}[/bold]"
                  f"  [dim]-> {out_dir / 'matrix.json'}[/dim]")

    if args.update_baseline:
        BASELINE_PATH.parent.mkdir(exist_ok=True)
        current = flatten_statuses(run)
        if baseline:
            baseline.update(current)
            current = baseline
        BASELINE_PATH.write_text(json.dumps(current, indent=2, sort_keys=True))
        console.print(f"[green]Baseline updated:[/green] {BASELINE_PATH}")

    if report["regressions"]:
        console.print(f"[red bold]Regressions:[/red bold] {', '.join(report['regressions'])}")
        if args.diff:
            return 1
    elif args.diff:
        console.print("[green]No regressions vs baseline.[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
