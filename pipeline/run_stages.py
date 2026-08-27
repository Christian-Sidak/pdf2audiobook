"""Run pipeline stages for corpus docs (or any PDF).

Usage:
    python -m pipeline.run_stages --stages 1-3 --docs origins_of_the_kabbalah
    python -m pipeline.run_stages --stages 1-3          # all corpus docs
    python -m pipeline.run_stages --pdf path/to/book.pdf --stages 1-4
"""
from __future__ import annotations

import argparse
import importlib
import time

from evals.contracts import BookCtx
from evals.corpus import load_corpus
from evals.run import parse_stages
from pipeline.config import book_id_for

STAGE_MODULES = {
    1: "pipeline.s1_extract",
    2: "pipeline.s2_structural",
    3: "pipeline.s3_chapterize",
    4: "pipeline.s4_narration",
    5: "pipeline.s5_render",
    6: "pipeline.s6_assemble",
}


def run_stages(book: BookCtx, stages: set[int]) -> None:
    for n in sorted(stages):
        mod = importlib.import_module(STAGE_MODULES[n])
        t0 = time.time()
        out = mod.run(book)
        print(f"[{book.book_id}] stage {n} -> {out.name} ({time.time() - t0:.1f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="1-3")
    ap.add_argument("--docs", help="comma-separated corpus doc ids (default: all)")
    ap.add_argument("--pdf", help="run on an arbitrary PDF instead of corpus docs")
    ap.add_argument("--engine", help="TTS engine override (e.g. qwen3tts)")
    ap.add_argument("--voice", help="voice: kokoro preset or reference WAV path")
    args = ap.parse_args()

    stages = parse_stages(args.stages)
    if args.pdf:
        config = {k: v for k, v in
                  (("tts_engine", args.engine), ("voice", args.voice)) if v}
        book = BookCtx.for_book(book_id_for(args.pdf), args.pdf, config=config)
        run_stages(book, stages)
        return

    corpus = load_corpus()
    wanted = set(args.docs.split(",")) if args.docs else None
    for doc in corpus.documents:
        if wanted and doc.id not in wanted:
            continue
        if not doc.pdf.exists():
            print(f"SKIP {doc.id}: missing {doc.pdf}")
            continue
        run_stages(BookCtx.for_book(doc.id, doc.pdf), stages)


if __name__ == "__main__":
    main()
