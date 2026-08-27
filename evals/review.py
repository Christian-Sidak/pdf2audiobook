"""Review queue CLI for residual QC failures.

Usage:
    python -m evals.review list
    python -m evals.review show 3
    python -m evals.review resolve 3 --fixed
    python -m evals.review resolve 3 --allow   # adds to the doc's deletion_allowlist
"""
from __future__ import annotations

import argparse
import json

from rich.console import Console
from rich.markup import escape

from evals.corpus import load_corpus, save_corpus
from evals.qc_loop import REVIEW_QUEUE

console = Console()


def _load() -> list[dict]:
    if not REVIEW_QUEUE.exists():
        return []
    return [json.loads(l) for l in REVIEW_QUEUE.read_text().splitlines() if l.strip()]


def _save(items: list[dict]) -> None:
    REVIEW_QUEUE.write_text("".join(json.dumps(i) + "\n" for i in items))


def cmd_list(args) -> None:
    items = _load()
    if not items:
        console.print("[green]review queue empty[/green]")
        return
    for i, item in enumerate(items):
        console.print(f"  [{i}] {item['book']} s{item['stage']} {item['dimension']} "
                      f"{escape(item['message'][:90])}")


def cmd_show(args) -> None:
    item = _load()[args.n]
    console.print_json(json.dumps(item, indent=2))


def cmd_resolve(args) -> None:
    items = _load()
    item = items.pop(args.n)
    if args.allow:
        corpus = load_corpus()
        for d in corpus.raw["documents"]:
            if d["id"] == item["book"]:
                allow = d.setdefault("assertions", {}).setdefault("deletion_allowlist", [])
                pattern = args.pattern or (item.get("context") or item["message"])[:80]
                allow.append(pattern)
                save_corpus(corpus)
                console.print(f"[green]added to {item['book']} deletion_allowlist:[/green] {escape(pattern)}")
                break
    _save(items)
    console.print(f"[green]resolved[/green] ({len(items)} remaining)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.set_defaults(fn=cmd_list)
    p = sub.add_parser("show"); p.add_argument("n", type=int); p.set_defaults(fn=cmd_show)
    p = sub.add_parser("resolve")
    p.add_argument("n", type=int)
    p.add_argument("--fixed", action="store_true")
    p.add_argument("--allow", action="store_true")
    p.add_argument("--pattern")
    p.set_defaults(fn=cmd_resolve)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
