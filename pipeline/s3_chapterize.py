"""Stage 3: chapterize.

Builds the chapter tree from the embedded PDF outline when it is trustworthy
(junk outlines full of printing filenames are rejected), falling back to
heading heuristics: explicit chapter patterns first, then large-font heading
blocks. Printed TOC pages and front/back matter are classified and flagged so
downstream stages never narrate them.

Artifact: 03_chapters.json
    {chapters: [{id, title, level, source, matter, front_matter,
                 start_page, end_page, text}]}
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from evals.contracts import ARTIFACT_FILES, BookCtx

_DOT_LEADER = re.compile(r"\.{3,}\s*\d+\s*$")

FRONT_TITLES = re.compile(
    r"^(contents|table of contents|copyright|title page|half title|dedication|"
    r"acknowledg\w*|list of (figures|tables|maps|illustrations|abbreviations)|"
    r"figures|illustrations|maps|plates|tables|(a )?note on .*|"
    r"abbreviations|about the author|epigraph)\s*$", re.IGNORECASE)
BACK_TITLES = re.compile(
    r"^(bibliograph\w*|references|works cited|index(es)?|notes|endnotes|glossary|"
    r"appendix\w*|further reading|about the author|also by)\b", re.IGNORECASE)
BODY_SPECIAL = re.compile(
    r"^(introduction|preface|foreword|prologue|epilogue|conclusion)\b", re.IGNORECASE)

_FRONT_CONTENT = re.compile(
    r"all rights reserved|library of congress|isbn[\s:0-9\-]|printed in the united states",
    re.IGNORECASE)

_JUNK_TITLE = re.compile(
    r"\.indd|\.pdf|^\d+_|^cover\b|amazon|-print$|^untitled$|^blank\b", re.IGNORECASE)

_NUM_WORDS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
              "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
              "seventeen", "eighteen", "nineteen", "twenty")
_CHAPTER_HEAD = re.compile(
    rf"^(?:chapter|part|book)\s+(\d+|[ivxlc]+|{'|'.join(_NUM_WORDS)})\b\s*[:.\-]?\s*(.{{0,60}})$",
    re.IGNORECASE)


def _unspace_letters(s: str) -> str:
    return re.sub(r"\b(?:[A-Za-z] ){2,}[A-Za-z]\b", lambda m: m.group(0).replace(" ", ""), s)


def _matter_for(title: str, position: float) -> str:
    if FRONT_TITLES.match(title.strip()):
        return "front"
    if BACK_TITLES.match(title.strip()) and position > 0.5:
        return "back"
    return "body"


_TRAILING_PAGE = re.compile(r"\s(\d{1,4}|[ivxlc]{1,7})\s*$")


def _is_toc_page(text: str) -> bool:
    """Fallback for joined text; the reliable signal is the stage-2 per-page
    `toc_like` flag, computed on raw lines before paragraph joining."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    if re.fullmatch(r"contents|table of contents", lines[0].strip(), re.IGNORECASE):
        return True
    if sum(1 for l in lines if _DOT_LEADER.search(l)) >= 3:
        return True
    trailing = sum(1 for l in lines if _TRAILING_PAGE.search(l) or "•" in l)
    return trailing >= 4 and trailing / len(lines) >= 0.3


def _page_is_toc(p: dict) -> bool:
    return bool(p.get("toc_like")) or _is_toc_page(p["text"])


def _heading_of(paragraph: str) -> str | None:
    """A short standalone paragraph that reads as a chapter heading."""
    p = _unspace_letters(paragraph.strip())
    if not (3 <= len(p) <= 90) or "\n" in p:
        return None
    if _DOT_LEADER.search(p) or "•" in p:
        return None  # printed-TOC line, not a heading
    if re.search(r"\s\d{1,4}$", p) and not re.fullmatch(
            r"(?:chapter|part|book)\s+\d{1,4}", p, re.IGNORECASE):
        return None  # trailing page number: TOC or list entry
    if re.search(r"[-–]\s*\d{1,4}\b", p) or p.casefold().endswith("contents"):
        return None  # embedded page reference or TOC tail: a contents line
    if _CHAPTER_HEAD.match(p):
        return p
    if BODY_SPECIAL.match(p) and len(p) < 40 and (p.isupper() or p.istitle()):
        return p
    return None


def _title_case(s: str) -> str:
    s = _unspace_letters(s).strip().rstrip(",;:.- ")
    return " ".join(w if w.isupper() and len(w) <= 3 else w.capitalize()
                    for w in s.split()) if s.isupper() else s


def _pick_outline_level(toc: list[dict]) -> list[dict] | None:
    entries = [t for t in toc if t.get("page") is not None]
    by_level: dict[int, list[dict]] = {}
    for t in entries:
        by_level.setdefault(t["level"], []).append(t)
    for level in sorted(by_level):
        items = by_level[level]
        junky = sum(1 for t in items if _JUNK_TITLE.search(t["title"]))
        if len(items) < 2 or junky / len(items) > 0.3:
            continue
        if len(items) <= 80:
            return [t for t in items if not _JUNK_TITLE.search(t["title"])]
    return None


def _chapters_from_sidecar(book: BookCtx, pages: list[dict]) -> list[dict] | None:
    """Manual chapter declaration: <book>.chapters.yaml next to the PDF wins
    over every heuristic. The escape hatch for typography no detector should
    be contorted around (display-font homoglyphs, subtitle-smaller-than-body).

        chapters:
          - {title: "First Essay. Historical Criticism: Theory of Modes",
             start_page: 50}          # 1-indexed
          - {title: Notes and Index, start_page: 372, matter: back}
    """
    sidecar = book.pdf_path.with_suffix(".chapters.yaml")
    if not sidecar.exists():
        return None
    from ruamel.yaml import YAML

    spec = YAML(typ="safe").load(sidecar.open())
    entries = spec.get("chapters") or []
    if not entries:
        return None
    entries = sorted(entries, key=lambda e: int(e["start_page"]))
    n_pages = len(pages)
    chapters = []
    first = int(entries[0]["start_page"]) - 1
    if first > 0:
        chapters.append({"title": "Front Matter", "matter": "front",
                         "start_page": 0, "end_page": first - 1, "source": "sidecar"})
    for i, e in enumerate(entries):
        start = int(e["start_page"]) - 1
        end = int(entries[i + 1]["start_page"]) - 2 if i + 1 < len(entries) else n_pages - 1
        chapters.append({"title": str(e["title"]),
                         "matter": e.get("matter", "body"),
                         "start_page": start, "end_page": max(end, start),
                         "source": "sidecar"})
    return chapters


def _chapters_from_outline(toc: list[dict], pages: list[dict]) -> list[dict] | None:
    top = _pick_outline_level(toc)
    if not top or len(top) < 2:
        return None

    top = sorted(top, key=lambda t: t["page"])
    n_pages = len(pages)
    chapters = []
    if top[0]["page"] > 0:
        chapters.append({"title": "Front Matter", "matter": "front",
                         "start_page": 0, "end_page": top[0]["page"] - 1, "source": "outline"})
    for i, entry in enumerate(top):
        end = top[i + 1]["page"] - 1 if i + 1 < len(top) else n_pages - 1
        end = max(end, entry["page"])
        title = _title_case(entry["title"].strip())
        chapters.append({
            "title": title,
            "matter": _matter_for(title, entry["page"] / max(1, n_pages)),
            "start_page": entry["page"], "end_page": end, "source": "outline",
        })
    return chapters


def _reads_as_title(text: str) -> bool:
    """OCR noise in large type must not become a chapter ('that—the other
    side had aes it ee a'): titles start upper and read as language."""
    from pipeline.textquality import dict_word_ratio

    t = text.strip()
    if not t or not t[0].isupper():
        return False
    return dict_word_ratio(t) >= 0.5


def _font_boundaries(extract: dict, pages_by_number: dict[int, dict]) -> list[tuple[int, int, str]]:
    """Large-font short blocks near the top of a page: heading candidates for
    books whose chapters are not 'Chapter N' formatted."""
    sizes: Counter[float] = Counter()
    for p in extract["pages"]:
        for b in p.get("blocks", []):
            if b["size"] > 0:
                sizes[b["size"]] += len(b["text"])
    if not sizes:
        return []
    body_size = sizes.most_common(1)[0][0]

    boundaries = []
    for p in extract["pages"]:
        h = p.get("page_height") or 1.0
        for b in p.get("blocks", []):
            text = _unspace_letters(re.sub(r"\s+", " ", b["text"]).strip())
            if (b["size"] >= body_size * 1.35 and 4 <= len(text) <= 90
                    and b["bbox"][1] < h * 0.45 and not text.isdigit()
                    and not _DOT_LEADER.search(text)
                    and _reads_as_title(text)
                    and pages_by_number.get(p["number"], {}).get("text")):
                boundaries.append((p["number"], 0, _title_case(text)))
                break
    return boundaries


def _chapters_from_headings(pages: list[dict], extract: dict) -> list[dict]:
    pages_by_number = {p["number"]: p for p in pages}
    boundaries: list[tuple[int, int, str]] = []
    for p in pages:
        if _page_is_toc(p):
            continue  # printed contents pages produce fake headings
        paras = p["text"].split("\n\n") if p["text"] else []
        for i, para in enumerate(paras):
            head = _heading_of(para)
            if head:
                title = _title_case(head)
                if i + 1 < len(paras):
                    nxt = _unspace_letters(paras[i + 1].strip())
                    if (nxt.isupper() and 3 < len(nxt) < 80 and "\n" not in nxt
                            and re.fullmatch(r"(?:chapter|part|book)\s+\S+\s*[:.\-]?\s*",
                                             head, re.IGNORECASE)):
                        title = f"{title.rstrip(' :.-')}: {_title_case(nxt)}"
                boundaries.append((p["number"], i, title))
                break

    if len(boundaries) < 2:
        font_bounds = _font_boundaries(extract, pages_by_number)
        if len(font_bounds) >= max(2, len(boundaries) + 1):
            boundaries = font_bounds

    chapters: list[dict] = []
    if not boundaries:
        return [{"title": "Full Text", "matter": "body", "start_page": pages[0]["number"],
                 "end_page": pages[-1]["number"], "source": "heuristic"}]
    first_page = boundaries[0][0]
    if first_page > pages[0]["number"]:
        chapters.append({"title": "Front Matter", "matter": "front",
                         "start_page": pages[0]["number"], "end_page": first_page - 1,
                         "source": "heuristic"})
    for i, (page, para_idx, title) in enumerate(boundaries):
        end = boundaries[i + 1][0] - 1 if i + 1 < len(boundaries) else pages[-1]["number"]
        chapters.append({"title": title, "matter": _matter_for(title, page / max(1, len(pages))),
                         "start_page": page, "end_page": max(end, page),
                         "source": "heuristic", "_para_idx": para_idx})
    return chapters


def _fill_text(chapters: list[dict], pages: list[dict], excluded: list[dict]) -> None:
    by_number = {p["number"]: p for p in pages}
    total = max(ch["end_page"] for ch in chapters) + 1 if chapters else 1
    for ch in chapters:
        paras: list[str] = []
        for n in range(ch["start_page"], ch["end_page"] + 1):
            p = by_number.get(n)
            if not p or not p["text"]:
                continue
            if n != ch["start_page"] and _page_is_toc(p):
                # Mini-TOC page inside a chapter: never narrated.
                excluded.append({"page": n, "reason": "mini_toc", "chars": len(p["text"])})
                continue
            page_paras = p["text"].split("\n\n")
            kept = []
            for q in page_paras:
                if _is_listing_para(q):
                    excluded.append({"page": n, "reason": "listing", "chars": len(q)})
                else:
                    kept.append(q)
            page_paras = kept
            if n == ch["start_page"]:
                skip = ch.pop("_para_idx", 0)
                page_paras = page_paras[skip:]
                while page_paras and _consumed_by_title(page_paras[0], ch["title"]):
                    excluded.append({"page": n, "reason": "title", "chars": len(page_paras[0])})
                    page_paras = page_paras[1:]
                # Chapter-opening section outlines and part-opener mini-TOCs
                # are navigation, not narration.
                while page_paras and _is_section_outline(page_paras[0]):
                    excluded.append({"page": n, "reason": "section_outline", "chars": len(page_paras[0])})
                    page_paras = page_paras[1:]
                if page_paras and (p.get("toc_like") or _is_toc_page("\n".join(page_paras))):
                    excluded.append({"page": n, "reason": "mini_toc", "chars": sum(len(x) for x in page_paras)})
                    page_paras = []
            paras.extend(page_paras)
        text = "\n\n".join(paras)
        head = text[:3000]
        early = ch["start_page"] / total < 0.15
        if ch["matter"] == "body" and _is_toc_page(head):
            ch["matter"] = "front"
            ch["title"] = ch["title"] if FRONT_TITLES.match(ch["title"]) else "Contents"
        elif ch["matter"] == "body" and early and _FRONT_CONTENT.search(head):
            ch["matter"] = "front"
        ch["text"] = text


_OUTLINE_LINE = re.compile(r"^\s*\d{1,2}\.\s+\S")
_SPACED_DOTS = re.compile(r"(?:\.\s*){4,}\d*\s*$")


def _is_listing_para(paragraph: str) -> bool:
    """Embedded TOC/listing paragraphs anywhere in a chapter: a bare
    'Contents' heading or dot-leader lines ('Introduction . . . . 27')."""
    p = paragraph.strip()
    if p.casefold() in ("contents", "table of contents"):
        return True
    lines = [l for l in p.splitlines() if l.strip()]
    if not lines:
        return False
    dotted = sum(1 for l in lines if _SPACED_DOTS.search(l))
    return dotted >= 2 or dotted == len(lines)


def _is_section_outline(paragraph: str) -> bool:
    """A chapter-opening list of numbered section titles (no sentences)."""
    p = paragraph.strip()
    if p.casefold() in ("contents", "table of contents"):
        return True
    lines = [l for l in p.splitlines() if l.strip()]
    if len(lines) < 2:
        return bool(_OUTLINE_LINE.match(p)) and not p.rstrip().endswith(".") and len(p) < 200
    outline = sum(1 for l in lines if _OUTLINE_LINE.match(l) or _TRAILING_PAGE.search(l))
    return outline / len(lines) >= 0.6


def _consumed_by_title(paragraph: str, title: str) -> bool:
    p = re.sub(r"\W+", " ", _unspace_letters(paragraph)).casefold().strip()
    t = re.sub(r"\W+", " ", title).casefold().strip()
    return bool(p) and len(p) < 90 and (p in t or t.startswith(p) or t.endswith(p))


def run(book: BookCtx) -> Path:
    extract = json.loads((book.artifacts_dir / ARTIFACT_FILES["extract"]).read_text(encoding="utf-8"))
    structural = json.loads((book.artifacts_dir / ARTIFACT_FILES["structural"]).read_text(encoding="utf-8"))
    pages = structural["pages"]

    chapters = _chapters_from_sidecar(book, pages)
    if chapters is None:
        chapters = _chapters_from_outline(extract.get("toc", []), pages)
    if chapters is None:
        chapters = _chapters_from_headings(pages, extract)
    excluded: list[dict] = []
    _fill_text(chapters, pages, excluded)

    final = []
    for i, ch in enumerate(c for c in chapters if c["text"].strip()):
        final.append({
            "id": f"ch{i:02d}",
            "title": ch["title"],
            "level": 0,
            "source": ch["source"],
            "matter": ch["matter"],
            "front_matter": ch["matter"] != "body",
            "start_page": ch["start_page"],
            "end_page": ch["end_page"],
            "text": ch["text"],
        })

    out = book.artifacts_dir / ARTIFACT_FILES["chapters"]
    out.write_text(json.dumps({"source": "pipeline", "chapters": final, "excluded": excluded},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    return out
