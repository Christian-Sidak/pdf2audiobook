"""Seed per-document assertions into corpus.yaml from pipeline artifacts.

Usage:
    python -m evals.seed suggest-forbidden <book_id> [--auto --top 8]
    python -m evals.seed span <book_id> --pages 41-42 [--auto | --text "..."]
    python -m evals.seed chapters <book_id> [--from-golden-txt path] [--write-expect]
    python -m evals.seed show <book_id> --page 41
"""
from __future__ import annotations

import argparse
import json
import re
import sys

from rich.console import Console

from evals.artifacts import ArtifactSet
from evals.corpus import CORPUS_PATH, load_corpus, save_corpus
from evals.textutil import normalize, repeated_lines
from pipeline.config import ARTIFACTS_DIR, ROOT

console = Console()

GOLDEN_DIR = CORPUS_PATH.parent / "golden"


def _raw_doc(corpus, book_id: str):
    for d in corpus.raw["documents"]:
        if d["id"] == book_id:
            return d
    sys.exit(f"unknown doc id: {book_id}")


def _assertions(raw_doc, key: str):
    a = raw_doc.setdefault("assertions", {})
    return a.setdefault(key, [])


def cmd_suggest_forbidden(args) -> None:
    corpus = load_corpus()
    art = ArtifactSet(ARTIFACTS_DIR / args.book_id)
    pages = [p["text"] for p in art.extract["pages"]]
    candidates = sorted(repeated_lines(pages, min_fraction=0.15).items(), key=lambda kv: -kv[1])

    if not candidates:
        console.print("[yellow]No repeated-line candidates found.[/yellow]")
        return

    picked = []
    for line, count in candidates[: args.top]:
        console.print(f"  [{count:4d} pages] {line[:90]!r}")
        if args.auto:
            picked.append((line, count))

    if not args.auto:
        console.print("[dim]Re-run with --auto to write the listed candidates.[/dim]")
        return

    raw_doc = _raw_doc(corpus, args.book_id)
    existing = {e["text"] for e in _assertions(raw_doc, "must_not_contain")}
    added = 0
    for i, (line, count) in enumerate(picked):
        if line in existing:
            continue
        _assertions(raw_doc, "must_not_contain").append({
            "id": f"header_{i:02d}", "text": line, "match": "line",
            "note": f"appears on {count} pages (suggest-forbidden)",
        })
        added += 1
    save_corpus(corpus)
    console.print(f"[green]Added {added} must_not_contain entries for {args.book_id}[/green]")


def _page_text(art: ArtifactSet, number: int) -> str:
    for p in art.extract["pages"]:
        if p["number"] == number:
            return p["text"]
    sys.exit(f"page {number} not found")


def _auto_span(art: ArtifactSet, a: int, b: int) -> str | None:
    """Reconstruct the sentence crossing the a->b page break from raw text."""
    pages = [p["text"] for p in art.extract["pages"]]
    junk = set(repeated_lines(pages, min_fraction=0.15))

    def clean_lines(text: str) -> list[str]:
        out = []
        for line in text.splitlines():
            n = normalize(line, casefold=True)
            if not n or re.fullmatch(r"[\divxlc]+", n) or n in junk:
                continue
            out.append(line.strip())
        return out

    tail_txt = " ".join(clean_lines(_page_text(art, a)))
    head_txt = " ".join(clean_lines(_page_text(art, b)))

    m = re.search(r"[.!?][\"')\]]*\s+(?=[A-Z\"'(\[])(?!.*[.!?][\"')\]]*\s+[A-Z\"'(\[])", tail_txt)
    tail = tail_txt[m.end():] if m else tail_txt[-300:]
    m2 = re.search(r"[.!?][\"')\]]*(\s|$)", head_txt)
    head = head_txt[: m2.end()] if m2 else head_txt[:300]

    joined = f"{tail} {head}".strip()
    joined = re.sub(r"(\w)-\s+(\w)", r"\1\2", joined)  # dehyphenate the break
    joined = normalize(joined)
    return joined if len(joined.split()) >= 6 else None


def cmd_span(args) -> None:
    corpus = load_corpus()
    art = ArtifactSet(ARTIFACTS_DIR / args.book_id)
    a, b = (int(x) - 1 for x in args.pages.split("-"))  # 1-indexed input

    if args.text:
        sentence = normalize(args.text)
    elif args.auto:
        sentence = _auto_span(art, a, b)
        if not sentence:
            sys.exit("could not auto-reconstruct span; pass --text")
        console.print(f"[bold]Reconstructed:[/bold] {sentence}")
    else:
        console.print(f"[bold]--- end of page {a + 1} ---[/bold]")
        console.print(_page_text(art, a)[-400:])
        console.print(f"[bold]--- start of page {b + 1} ---[/bold]")
        console.print(_page_text(art, b)[:400])
        console.print("[dim]Re-run with --auto or --text to record the crossing sentence.[/dim]")
        return

    raw_doc = _raw_doc(corpus, args.book_id)
    _assertions(raw_doc, "must_contain").append({
        "id": f"span_p{a + 1}_{b + 1}", "text": sentence,
        "note": f"sentence crossing pages {a + 1}-{b + 1}",
    })
    save_corpus(corpus)
    console.print(f"[green]Recorded span_p{a + 1}_{b + 1} for {args.book_id}[/green]")


def cmd_chapters(args) -> None:
    corpus = load_corpus()
    titles: list[str] = []

    if args.from_golden_txt:
        text = (ROOT / args.from_golden_txt).read_text(encoding="utf-8")
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if re.fullmatch(r"Chapter\s+(?:[A-Z]+|\d+)", line.strip()):
                title = normalize(line)
                # Attach the all-caps subtitle if one follows within 3 lines.
                for nxt in lines[i + 1: i + 4]:
                    nxt = nxt.strip()
                    if not nxt or re.fullmatch(r"=+", nxt):
                        continue
                    if nxt.isupper() and 3 < len(nxt) < 80:
                        title += f": {normalize(nxt)}"
                    break
                titles.append(title)
    else:
        art = ArtifactSet(ARTIFACTS_DIR / args.book_id)
        titles = [c["title"] for c in art.chapters["chapters"] if not c.get("front_matter")]

    for i, t in enumerate(titles):
        console.print(f"  {i + 1}. {t}")

    out = GOLDEN_DIR / args.book_id / "chapters.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"titles": titles}, indent=2, ensure_ascii=False))
    console.print(f"[green]Wrote {out} ({len(titles)} titles)[/green]")

    if args.write_expect:
        raw_doc = _raw_doc(corpus, args.book_id)
        raw_doc.setdefault("expect", {})["chapter_count"] = len(titles)
        save_corpus(corpus)
        console.print(f"[green]expect.chapter_count = {len(titles)}[/green]")


def cmd_show(args) -> None:
    art = ArtifactSet(ARTIFACTS_DIR / args.book_id)
    console.print(_page_text(art, args.page - 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("suggest-forbidden")
    p.add_argument("book_id")
    p.add_argument("--auto", action="store_true")
    p.add_argument("--top", type=int, default=8)
    p.set_defaults(fn=cmd_suggest_forbidden)

    p = sub.add_parser("span")
    p.add_argument("book_id")
    p.add_argument("--pages", required=True, help="e.g. 41-42 (1-indexed)")
    p.add_argument("--auto", action="store_true")
    p.add_argument("--text")
    p.set_defaults(fn=cmd_span)

    p = sub.add_parser("chapters")
    p.add_argument("book_id")
    p.add_argument("--from-golden-txt")
    p.add_argument("--write-expect", action="store_true")
    p.set_defaults(fn=cmd_chapters)

    p = sub.add_parser("show")
    p.add_argument("book_id")
    p.add_argument("--page", type=int, required=True)
    p.set_defaults(fn=cmd_show)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
