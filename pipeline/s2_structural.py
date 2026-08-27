"""Stage 2: structural cleaning.

Removes running headers/footers (positional + cross-page repetition), page
numbers, and footnotes (font-size based); rejoins hyphenated words with a
dictionary guard; re-flows paragraphs within and across page boundaries so no
sentence is split by a page turn.

Artifacts: 02_structural.json (provenance), 02_body.txt (continuous text).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from evals.contracts import ARTIFACT_FILES, BookCtx
from pipeline.textquality import _dictionary

from pipeline.config import CFG

HEADER_ZONE = float(CFG["structural"]["header_zone"])
FOOTER_ZONE = float(CFG["structural"]["footer_zone"])
REPEAT_MIN_COUNT = int(CFG["structural"]["repeat_min_count"])
FOOTNOTE_SIZE_RATIO = float(CFG["structural"]["footnote_size_ratio"])
_TERMINAL = tuple('.!?:;"”’)')
_PAGE_NUM = re.compile(r"^[\d\s]{1,6}$|^[ivxlcIVXLC]{1,7}$")


def _unspace_letters(s: str) -> str:
    """Collapse letter-spaced typography: 'K A B B A L A H' -> 'KABBALAH'."""
    return re.sub(r"\b(?:[A-Za-z] ){2,}[A-Za-z]\b", lambda m: m.group(0).replace(" ", ""), s)


def _norm_template(line: str) -> str:
    """Normalize a margin line into a header template: casefold, digits out,
    letter-spacing collapsed."""
    t = _unspace_letters(line)
    t = re.sub(r"\d+", "#", t).casefold()
    return re.sub(r"\s+", " ", t).strip()


def _margin_lines(page: dict) -> list[tuple[str, str]]:
    """(zone, line) pairs for blocks in the header/footer zones."""
    h = page["page_height"] or 1.0
    out = []
    for b in page.get("blocks", []):
        y0, y1 = b["bbox"][1], b["bbox"][3]
        zone = "header" if y1 < h * HEADER_ZONE else "footer" if y0 > h * FOOTER_ZONE else None
        if zone:
            for line in b["text"].splitlines():
                if line.strip():
                    out.append((zone, line.strip()))
    return out


def _detect_templates(pages: list[dict]) -> tuple[set[str], set[str]]:
    headers: Counter[str] = Counter()
    footers: Counter[str] = Counter()
    for p in pages:
        seen: set[tuple[str, str]] = set()
        for zone, line in _margin_lines(p):
            t = _norm_template(line)
            if len(t) >= 2 and (zone, t) not in seen:
                seen.add((zone, t))
                (headers if zone == "header" else footers)[t] += 1
    # Absolute threshold: per-chapter running headers (chapter title on recto
    # pages) repeat only over one chapter's span and would evade any
    # whole-document fraction. Margin-zone restriction keeps this safe.
    threshold = REPEAT_MIN_COUNT
    return ({t for t, c in headers.items() if c >= threshold},
            {t for t, c in footers.items() if c >= threshold})


def _body_font_size(pages: list[dict]) -> float:
    weighted: Counter[float] = Counter()
    for p in pages:
        for b in p.get("blocks", []):
            if b["size"] > 0:
                weighted[b["size"]] += len(b["text"])
    if not weighted:
        return 10.0
    return weighted.most_common(1)[0][0]


def _join_wrapped(lines: list[str], dictionary: frozenset[str]) -> str:
    """Join hard-wrapped lines into one paragraph, repairing hyphen breaks."""
    out = ""
    for raw in lines:
        s = re.sub(r"\s+", " ", raw).strip()
        # Soft hyphen: acts as a hyphen at line end, invisible elsewhere.
        s = re.sub(r"­+$", "-", s).replace("­", "")
        if not s:
            continue
        if not out:
            out = s
            continue
        if out.endswith("-"):
            out = _dehyphenate_join(out, s, dictionary)
        else:
            out = out + " " + s
    return out


def _dehyphenate_join(a: str, b: str, dictionary: frozenset[str]) -> str:
    """Join 'liter-' + 'ature'. Keep the hyphen only for true compounds:
    both fragments are standalone words and their fusion is not a word
    (self- + consciousness). Proper nouns and ordinary wraps fuse plain
    (Eli- + ezer -> Eliezer, knowl- + edge -> knowledge)."""
    head = a[:-1]
    m = re.search(r"([A-Za-z]+)$", head)
    n = re.match(r"([A-Za-z]+)", b)
    if not m or not n:
        return head + b
    left, right = m.group(1).lower(), n.group(1).lower()
    fused = left + right
    if fused in dictionary:
        return head + b
    if left in dictionary and right in dictionary and len(left) >= 4:
        return a + b  # keep hyphen: real compound broken at its own hyphen
    return head + b


_NOTE_LINE = re.compile(r"^\s*(\d{1,3}\.\s|\*|†)")
_CAPTION = re.compile(r"^\s*(Figure|Fig\.|Table|Map|Plate|Illustration|Photo)\s+\d+[.:]", re.IGNORECASE)


def strip_superscript_markers(para: str) -> str:
    """Remove inline superscript footnote markers fused to punctuation or
    words: 'universe."62 ', 'desire].67', 'Orient19', and trailing
    '...Habakkuk 3:4.197' at paragraph end. Real numbers survive: '1,204',
    'War of 1812', 'x2' (math), decimals mid-sentence."""
    para = re.sub(r'([A-Za-z][\.\"\'”’\)\],;:!\?]{1,3})\d{1,3}(?=\s|$)', r"\1", para)
    para = re.sub(r"(?<=[a-z][a-z][a-z])\d{1,3}(?=[\s,;:\.\)\]]|$)", "", para)
    para = re.sub(r"([.!?\]\)\"'”’])\d{1,3}\s*$", r"\1", para)
    return para


def _is_footnote_block(b: dict, body_size: float, page_height: float) -> bool:
    """Block quotations also use a smaller font in scholarly books, so 'small
    font' alone only counts near the page bottom; higher up it must also look
    like a note ('12. Text...')."""
    y0 = b["bbox"][1]
    small_font = 0 < b["size"] < body_size * FOOTNOTE_SIZE_RATIO
    slightly_small = 0 < b["size"] < body_size * 0.97
    starts_like_note = bool(_NOTE_LINE.match(b["text"]))
    if small_font and y0 > page_height * 0.62:
        return True
    if slightly_small and starts_like_note and y0 > page_height * 0.4:
        return True
    lines = b["text"].splitlines()
    note_lines = sum(1 for l in lines if _NOTE_LINE.match(l))
    if y0 > page_height * 0.4 and note_lines >= max(1, len(lines) // 2) and b["size"] <= body_size:
        return True
    return False


_TOC_LINE_DOT = re.compile(r"(?:\.\s*){4,}\d*\s*$")  # '.... 27' and '. . . .'
_TOC_LINE_TRAIL = re.compile(r"\s(\d{1,4}|[ivxlc]{1,7})\s*$")
_OUTLINE_NUM = re.compile(r"^\s*\d{1,2}\.\s+\S")


def _toc_like(lines: list[str]) -> bool:
    """TOC signature over raw lines, BEFORE paragraph joining destroys them."""
    lines = [l for l in lines if l.strip()]
    if not lines:
        return False
    if re.fullmatch(r"contents|table of contents", lines[0].strip(), re.IGNORECASE):
        return True
    if sum(1 for l in lines if _TOC_LINE_DOT.search(l)) >= 3:
        return True
    # NOTE: no numbered-outline page signature here: chapter openers carry
    # their own section outlines and would be swallowed. Outline paragraphs
    # are dropped per-paragraph at stage 3 instead.
    trailing = sum(1 for l in lines if _TOC_LINE_TRAIL.search(l) or "•" in l)
    return trailing >= 4 and trailing / len(lines) >= 0.3


def _clean_page(page: dict, headers: set[str], footers: set[str], body_size: float,
                dictionary: frozenset[str], removed: dict, ocr_page: bool = False) -> list[str]:
    """Return the page's body paragraphs in reading order."""
    h = page["page_height"] or 1.0
    paragraphs: list[str] = []
    blocks = sorted(page.get("blocks", []), key=lambda b: (round(b["bbox"][1]), b["bbox"][0]))

    for b in blocks:
        y0, y1 = b["bbox"][1], b["bbox"][3]
        in_header = y1 < h * HEADER_ZONE
        in_footer = y0 > h * FOOTER_ZONE

        if _is_footnote_block(b, body_size, h):
            removed["footnotes"].append(b["text"][:200])
            continue
        if _CAPTION.match(b["text"]):
            removed["captions"].append(b["text"][:200])
            continue

        kept_lines: list[str] = []
        for line in b["text"].splitlines():
            s = line.strip()
            if not s:
                continue
            if _PAGE_NUM.fullmatch(s):
                removed["page_numbers"] += 1
                continue
            t = _norm_template(s)
            if (in_header or in_footer) and (t in headers or t in footers):
                removed["matched_margin_lines"] += 1
                continue
            # Repeated header text occasionally lands just outside the zone,
            # but a large-font match is a real display title ('CHAPTER ONE'
            # heading vs the same words as a running head), never a header.
            if (t in headers and len(kept_lines) == 0 and not paragraphs
                    and b["size"] <= body_size * 1.15):
                removed["matched_margin_lines"] += 1
                continue
            kept_lines.append(s)

        if not kept_lines:
            continue
        para = strip_superscript_markers(_join_wrapped(kept_lines, dictionary))
        if para:
            paragraphs.append(para)

    # Merge blocks that PDF layout split mid-sentence within the page. OCR
    # text layers fragment into line-level blocks, so OCR'd pages merge as
    # aggressively as page boundaries do.
    merged: list[str] = []
    for para in paragraphs:
        if merged and _continues(merged[-1], para, cross_page=ocr_page):
            merged[-1] = _join_hyphen_aware(merged[-1], para, dictionary)
        else:
            merged.append(para)
    return merged


def _continues(prev: str, nxt: str, cross_page: bool = False) -> bool:
    """Does `nxt` continue a sentence `prev` left unfinished?"""
    if prev.endswith(_TERMINAL):
        return False
    if prev.endswith("-") or prev.endswith(","):
        return True
    if cross_page:
        # An unfinished paragraph at a page end virtually always continues,
        # even when the next word is a capitalized proper noun. The exception
        # is a chapter/section display title opening the next page.
        return not _looks_heading(nxt)
    if nxt[:1].islower() or nxt[:1] in "\"'([":
        return True
    # Pre-1800 typography capitalizes common Nouns mid-sentence, so the
    # lowercase test misses breaks like 'afford but a forry / Sketch of
    # Guinea'. A paragraph ending on a lowercase word with no terminal
    # punctuation still continues into a capitalized word, headings aside.
    return (prev[-1:].islower() and prev[-1:].isalpha()
            and not _looks_heading(prev) and not _looks_heading(nxt))


_HEADING_SHAPE = re.compile(
    r"^(?:chapter|part|book|section|introduction|preface|foreword|prologue|"
    r"epilogue|conclusion|appendix)\b", re.IGNORECASE)


def _looks_heading(para: str) -> bool:
    p = para.strip()
    if len(p) > 90 or not p:
        return False
    if p.isupper():
        return True
    return bool(_HEADING_SHAPE.match(p)) and p[-1:] not in ".!?,;"


def _join_hyphen_aware(a: str, b: str, dictionary: frozenset[str]) -> str:
    if a.endswith("-"):
        return _dehyphenate_join(a, b, dictionary)
    return a + " " + b


def run(book: BookCtx) -> Path:
    extract = json.loads((book.artifacts_dir / ARTIFACT_FILES["extract"]).read_text(encoding="utf-8"))
    pages = extract["pages"]
    dictionary = _dictionary()

    headers, footers = _detect_templates(pages)
    body_size = _body_font_size(pages)

    removed = {"headers": sorted(headers), "footers": sorted(footers),
               "page_numbers": 0, "matched_margin_lines": 0, "footnotes": [], "captions": []}

    ocr_pages = set(extract.get("ocr_pages", []))
    page_paragraphs: list[tuple[int, list[str]]] = []
    toc_pages: set[int] = set()
    for p in pages:
        raw_lines = [l for b in p.get("blocks", []) for l in b["text"].splitlines()]
        if _toc_like(raw_lines):
            toc_pages.add(p["number"])
        page_paragraphs.append((p["number"], _clean_page(p, headers, footers, body_size,
                                                         dictionary, removed,
                                                         ocr_page=p["number"] in ocr_pages)))

    # Cross-page re-flow: a page ending mid-sentence continues onto the next.
    # Look back past pages whose content was entirely removed (headers,
    # footnotes) so continuity survives them.
    flowed: list[tuple[int, list[str]]] = []
    for number, paras in page_paragraphs:
        if paras:
            prev_paras = next((pp for _, pp in reversed(flowed) if pp), None)
            if prev_paras and _continues(prev_paras[-1], paras[0], cross_page=True):
                prev_paras[-1] = _join_hyphen_aware(prev_paras[-1], paras[0], dictionary)
                paras = paras[1:]
        flowed.append((number, paras))

    # Repair in-line 'self- designation' style breaks; leave suspended
    # compounds ('eighteenth- and nineteenth-century') alone.
    def _fix_spaced_hyphen(m: re.Match) -> str:
        left, right = m.group(1), m.group(2)
        if (left + right).lower() in dictionary:
            return left + right
        return f"{left}-{right}"

    spaced = re.compile(r"\b([A-Za-z]{2,})- (?!(?:and|or|nor|und|oder|to)\b)([a-z]{2,})\b")
    out_pages = [{"number": n, "toc_like": n in toc_pages,
                  "text": spaced.sub(_fix_spaced_hyphen, "\n\n".join(paras))}
                 for n, paras in flowed]
    body = "\n\n".join(p["text"] for p in out_pages if p["text"])

    structural = {
        "source": "pipeline",
        "body_font_size": body_size,
        "removed": {**removed, "footnotes": removed["footnotes"][:500]},
        "footnote_count": len(removed["footnotes"]),
        "pages": out_pages,
    }
    (book.artifacts_dir / ARTIFACT_FILES["structural"]).write_text(
        json.dumps(structural, indent=1, ensure_ascii=False), encoding="utf-8")
    out = book.artifacts_dir / ARTIFACT_FILES["body"]
    out.write_text(body, encoding="utf-8")
    return out
