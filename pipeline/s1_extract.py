"""Stage 1: PDF extraction with layout blocks, embedded TOC, and OCR gate.

Artifact: 01_extract.json
    {source, pdf, page_count, toc, ocr_pages, ocr_unavailable?, pages: [
        {number, text, blocks: [{bbox, text, size}], page_height,
         chars, dict_word_ratio, garbage_density}]}
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import fitz

from evals.contracts import ARTIFACT_FILES, BookCtx
from pipeline.textquality import page_metrics

from pipeline.config import CFG

MIN_CHARS_PER_PAGE = int(CFG["extraction"]["min_chars_per_page"])
OCR_DOC_FRACTION = float(CFG["extraction"]["ocr_doc_fraction"])


import re

# Private-use-area glyphs (custom math/ligature fonts) and control chars carry
# no readable value; scrub at extraction. U+FFFD is kept: it signals real
# extraction failure and feeds garbage_density.
_SCRUB = re.compile("[\ue000-\uf8ff\u0000-\u0008\u000b\u000c\u000e-\u001f]")


def _scrub(text: str) -> str:
    return _SCRUB.sub("", text)


def _page_record(page: fitz.Page, number: int) -> dict:
    text = _scrub(page.get_text())
    blocks = []
    d = page.get_text("dict")
    for block in d.get("blocks", []):
        if block.get("type") != 0:  # text blocks only
            continue
        sizes: list[float] = []
        lines = []
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            sizes.extend(s["size"] for s in spans)
            lines.append("".join(s["text"] for s in spans))
        btext = _scrub("\n".join(lines)).strip()
        if not btext:
            continue
        sizes.sort()
        blocks.append({
            "bbox": [round(v, 1) for v in block["bbox"]],
            "text": btext,
            "size": round(sizes[len(sizes) // 2], 1) if sizes else 0.0,
        })
    return {
        "number": number,
        "text": text,
        "blocks": blocks,
        "page_height": round(page.rect.height, 1),
        **page_metrics(text),
    }


def _ocr_pdf(pdf_path: Path) -> Path | None:
    """Run ocrmypdf over the whole document; returns the OCR'd copy or None."""
    if not shutil.which("ocrmypdf"):
        return None
    out = Path(tempfile.mkdtemp(prefix="pdf2audiobook_ocr_")) / "ocr.pdf"
    cmd = ["ocrmypdf", "--skip-text", "--optimize", "0", "--quiet", str(pdf_path), str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return out if result.returncode == 0 and out.exists() else None


def run(book: BookCtx) -> Path:
    doc = fitz.open(str(book.pdf_path))
    pages = [_page_record(p, i) for i, p in enumerate(doc)]
    toc = [{"level": l - 1, "title": t.strip(), "page": p - 1 if p > 0 else None}
           for l, t, p in doc.get_toc()]
    doc.close()

    near_empty = [p["number"] for p in pages if p["chars"] < MIN_CHARS_PER_PAGE]
    ocr_pages: list[int] = []
    ocr_unavailable = False

    if len(pages) > 3 and len(near_empty) / len(pages) > OCR_DOC_FRACTION:
        ocr_copy = _ocr_pdf(book.pdf_path)
        if ocr_copy is None:
            ocr_unavailable = True
        else:
            ocr_doc = fitz.open(str(ocr_copy))
            redone = [_page_record(p, i) for i, p in enumerate(ocr_doc)]
            if not toc:
                toc = [{"level": l - 1, "title": t.strip(), "page": p - 1 if p > 0 else None}
                       for l, t, p in ocr_doc.get_toc()]
            ocr_doc.close()
            ocr_pages = [p["number"] for p in pages
                         if p["chars"] < MIN_CHARS_PER_PAGE and redone[p["number"]]["chars"] >= MIN_CHARS_PER_PAGE]
            pages = redone

    artifact = {
        "source": "pipeline",
        "pdf": str(book.pdf_path),
        "page_count": len(pages),
        "toc": toc,
        "ocr_pages": ocr_pages,
        "ocr_unavailable": ocr_unavailable,
        "pages": pages,
    }
    out = book.artifacts_dir / ARTIFACT_FILES["extract"]
    out.write_text(json.dumps(artifact, indent=1, ensure_ascii=False), encoding="utf-8")
    return out
