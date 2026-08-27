"""Stage 1 (extract) checks: schema, OCR gate, embedded TOC capture."""
from __future__ import annotations

import jsonschema

from evals.artifacts import ArtifactSet
from evals.checks import check
from evals.contracts import CheckResult, Violation
from evals.corpus import DocSpec

EXTRACT_SCHEMA = {
    "type": "object",
    "required": ["pages", "toc", "ocr_pages", "page_count"],
    "properties": {
        "page_count": {"type": "integer", "minimum": 1},
        "ocr_pages": {"type": "array", "items": {"type": "integer"}},
        "toc": {"type": "array", "items": {
            "type": "object",
            "required": ["level", "title"],
            "properties": {"level": {"type": "integer"}, "title": {"type": "string"}},
        }},
        "pages": {"type": "array", "minItems": 1, "items": {
            "type": "object",
            "required": ["number", "text", "dict_word_ratio", "garbage_density"],
        }},
    },
}


@check(stage=1, dimension="extract_schema_valid")
def extract_schema_valid(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    try:
        jsonschema.validate(art.extract, EXTRACT_SCHEMA)
    except jsonschema.ValidationError as e:
        return CheckResult.failed("extract_schema_valid", 1, [Violation(message=e.message[:300])])
    return CheckResult.passed("extract_schema_valid", 1)


@check(stage=1, dimension="ocr_gate_flags")
def ocr_gate_flags(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    ratio_min = cfg.get("dict_word_ratio_min", 0.85)
    garbage_max = cfg.get("garbage_char_density_max", 0.02)
    violations: list[Violation] = []

    flagged = set(art.extract.get("ocr_pages", []))
    expected_spec = doc.expect.get("ocr_expected_pages")
    if expected_spec == "scan":
        # Whole document is scanned: OCR must have run and produced text.
        if art.extract.get("ocr_unavailable"):
            violations.append(Violation(message="scanned document but OCR unavailable (install ocrmypdf)"))
        elif not flagged:
            violations.append(Violation(message="scanned document but no pages were OCR-routed"))
    else:
        expected = set(expected_spec or [])
        if flagged != expected:
            violations.append(Violation(
                message=f"OCR-routed pages {sorted(flagged)} != expected {sorted(expected)}"))

    # Page-level scoring here only catches SYSTEMIC extraction failure; foreign
    # front matter (abbreviation lists, copyright) legitimately scores low and
    # is excluded at stage 3, where body_chapter_quality re-checks what will
    # actually be narrated.
    # Recompute from text: stored metrics may predate metric fixes.
    from pipeline.textquality import dict_word_ratio, garbage_density

    pages = art.extract["pages"]
    substantial = [p for p in pages if p["chars"] > 200]
    garbage_pages = [(p, garbage_density(p["text"])) for p in substantial]
    garbage_pages = [(p, g) for p, g in garbage_pages if g > garbage_max]
    for p, g in garbage_pages[:10]:
        violations.append(Violation(
            message=f"garbage characters after gate: density={g:.4f}",
            unit_id=f"p{p['number'] + 1}"))
    severe = [p for p in substantial if dict_word_ratio(p["text"]) < 0.6]
    if substantial and len(severe) / len(substantial) > 0.2:
        violations.append(Violation(
            message=f"{len(severe)}/{len(substantial)} substantial pages below 0.6 dictionary-word "
                    f"ratio: systemic OCR/extraction failure"))
    bad_quality = garbage_pages

    # A document whose pages are mostly near-empty was never extracted at all
    # (scan without a text layer and no OCR ran).
    near_empty = sum(1 for p in pages if p["chars"] < 100)
    if len(pages) > 5 and near_empty / len(pages) > 0.3:
        violations.append(Violation(
            message=f"{near_empty}/{len(pages)} pages near-empty: scanned PDF without OCR"))

    if violations:
        return CheckResult.failed("ocr_gate_flags", 1, violations,
                                  bad_quality_pages=len(bad_quality), near_empty_pages=near_empty)
    return CheckResult.passed("ocr_gate_flags", 1, pages=len(pages))


@check(stage=1, dimension="toc_captured", requires=("expect.has_embedded_toc",))
def toc_captured(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    has_toc = bool(art.extract.get("toc"))
    expected = bool(doc.expect["has_embedded_toc"])
    if has_toc != expected:
        return CheckResult.failed("toc_captured", 1, [Violation(
            message=f"embedded TOC present={has_toc}, expected={expected}")])
    return CheckResult.passed("toc_captured", 1, entries=len(art.extract.get("toc", [])))
