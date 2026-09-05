"""Stage 2 (structural clean) checks, run against 02_body.txt."""
from __future__ import annotations

import re

from evals.artifacts import ArtifactSet
from evals.checks import check
from evals.contracts import CheckResult, Violation
from evals.corpus import DocSpec, SeededString
from evals.textutil import contains_normalized, excerpt, normalize, normalize_line, repeated_lines

_PAGE_NUM = re.compile(r"^\s*\d{1,4}\s*$")
_ROMAN = re.compile(r"^\s*[ivxlc]{1,7}\s*$", re.IGNORECASE)


def _seeded_hits(body: str, seeded: SeededString) -> bool:
    if seeded.match == "regex":
        return re.search(seeded.text, body, re.MULTILINE | re.IGNORECASE) is not None
    if seeded.match == "line":
        target = normalize_line(seeded.text)
        return any(normalize_line(l) == target for l in body.splitlines())
    return contains_normalized(body, seeded.text)


@check(stage=2, dimension="no_page_numbers")
def no_page_numbers(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    body = art.body_text
    violations = []
    for i, line in enumerate(body.splitlines()):
        if _PAGE_NUM.match(line) or (_ROMAN.match(line) and len(line.strip()) < 8):
            violations.append(Violation(message=f"standalone page number line: {line.strip()!r}",
                                        unit_id=f"line{i + 1}"))
    if violations:
        return CheckResult.failed("no_page_numbers", 2, violations[:20], total=len(violations))
    return CheckResult.passed("no_page_numbers", 2)


@check(stage=2, dimension="no_running_headers")
def no_running_headers(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    fraction = cfg.get("header_page_fraction", 0.2)
    pages = [p["text"] for p in art.extract["pages"]]
    candidates = repeated_lines(pages, min_fraction=fraction)
    body_lines = {}
    for line in art.body_text.splitlines():
        n = normalize_line(line)
        if n:
            body_lines[n] = body_lines.get(n, 0) + 1

    violations = []
    for cand, page_count in sorted(candidates.items(), key=lambda kv: -kv[1]):
        # A legitimate heading may appear once or twice; a surviving running
        # header appears many times.
        if body_lines.get(cand, 0) >= 3:
            violations.append(Violation(
                message=f"repeated line survives in body {body_lines[cand]}x (on {page_count} pages): {cand!r}"))
    if violations:
        return CheckResult.failed("no_running_headers", 2, violations[:20], total=len(violations))
    return CheckResult.passed("no_running_headers", 2, candidates=len(candidates))


@check(stage=2, dimension="seeded_forbidden")
def seeded_forbidden(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    if not doc.must_not_contain:
        return CheckResult.skipped("seeded_forbidden", 2, "no must_not_contain seeded")
    body = art.body_text
    violations = []
    for s in doc.must_not_contain:
        if _seeded_hits(body, s):
            violations.append(Violation(message=f"forbidden [{s.id}] present ({s.match}): {s.text[:80]!r}",
                                        context=excerpt(body, s.text[:40])))
    if violations:
        return CheckResult.failed("seeded_forbidden", 2, violations)
    return CheckResult.passed("seeded_forbidden", 2, seeded=len(doc.must_not_contain))


@check(stage=2, dimension="removed_text_absent")
def removed_text_absent(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    """Whatever stage 2 says it removed must actually be gone from the body:
    footnote blocks, captions, and non-prose pages (diagrams, family trees,
    charts). A page dropped by the judge that still leaks a line into the
    narration is a real defect, not a threshold question. Lines under 25
    characters are too short to test without false hits on ordinary words."""
    # Compare against NARRATED text (body chapters) with whitespace
    # normalized, and require a substantial run of the removed block (its
    # first 120 chars, blocks under 60 chars skipped): a single footnote line
    # legitimately recurs in a glossary or a list of names, a whole block
    # does not.
    from evals.textutil import normalize

    narrated = normalize(" ".join(c["text"] for c in art.chapters["chapters"]
                                  if c.get("matter", "body") == "body"))
    removed = art.structural.get("removed", {})
    violations = []
    kinds = {"footnotes": removed.get("footnotes", []), "captions": removed.get("captions", []),
             "non_prose_pages": [p.get("text", "") for p in removed.get("non_prose_pages", [])]}
    for kind, items in kinds.items():
        for item in items:
            probe = normalize(item)
            if len(probe) < 60:
                continue
            if probe[:120] in narrated:
                violations.append(Violation(message=f"{kind}: removed block present in narration: {probe[:80]!r}"))
    details = {k: len(v) for k, v in kinds.items()}
    if violations:
        return CheckResult.failed("removed_text_absent", 2, violations, **details)
    return CheckResult.passed("removed_text_absent", 2, **details)


@check(stage=2, dimension="punctuation_spacing")
def punctuation_spacing(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    """Sentence punctuation followed directly by a capital ("earth.Varnashram")
    fuses sentences into one take and defeats every sentence-aligned check.
    Clean PDFs run under 0.1 per 1,000 chars; Joothan's extraction ran 1.6
    before stage 2 learned to repair it."""
    body = art.body_text
    if len(body) < 5000:
        return CheckResult.skipped("punctuation_spacing", 2, "too little text")
    hits = re.findall(r"[a-z0-9][.!?][A-Z]|[a-z][,;:][A-Za-z]", body)
    per_k = 1000 * len(hits) / len(body)
    limit = cfg.get("punct_spacing_max_per_kchar", 0.3)
    if per_k > limit:
        return CheckResult.failed("punctuation_spacing", 2, [Violation(
            message=f"{per_k:.2f} fused punctuation per 1,000 chars > {limit} (e.g. {hits[:5]})")],
            per_kchar=round(per_k, 3))
    return CheckResult.passed("punctuation_spacing", 2, per_kchar=round(per_k, 3))


@check(stage=2, dimension="no_footnotes_inline")
def no_footnotes_inline(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    """Runs on NARRATED text (body chapters): bibliographies and notes
    sections legitimately contain footnote-shaped lines and are excluded as
    back matter, not cleaned line by line."""
    body = "\n\n".join(c["text"] for c in art.chapters["chapters"]
                       if not c.get("front_matter"))
    violations = []

    # Numbered lines are footnotes only when they carry citation apparatus;
    # numbered SECTION HEADINGS ('1. The State of Research') and content lists
    # are legitimate narration.
    numbered = re.findall(r"^\d{1,3}\.\s+.{10,}$", body, re.MULTILINE)
    citation = re.compile(r"cf\.|ibid|op\. cit|pp?\.\s*\d|\(\d{4}\)|\d{4}\),|[A-Z][a-z]+,\s+[A-Z][a-z]+.*\d{4}")
    footnote_lines = [l for l in numbered if citation.search(l)]
    if len(footnote_lines) >= 5:
        violations.append(Violation(
            message=f"{len(footnote_lines)} footnote-style lines ('N. Text...') in body",
            context=footnote_lines[0][:120]))

    remnants = re.findall(r"[a-z][.,;:\"]\d{1,3}\s", body)
    if len(remnants) >= 8:
        violations.append(Violation(
            message=f"{len(remnants)} superscript footnote-marker remnants (word.12 patterns)"))

    if violations:
        return CheckResult.failed("no_footnotes_inline", 2, violations,
                                  footnote_lines=len(footnote_lines), remnants=len(remnants))
    return CheckResult.passed("no_footnotes_inline", 2)


@check(stage=2, dimension="sentence_integrity_across_pages")
def sentence_integrity(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    if not doc.must_contain:
        return CheckResult.skipped("sentence_integrity_across_pages", 2, "no must_contain seeded")
    body = art.body_text
    violations = []
    for s in doc.must_contain:
        if not contains_normalized(body, s.text):
            violations.append(Violation(
                message=f"seeded sentence [{s.id}] not found whole",
                context=s.text[:160], fixable=False))
    if violations:
        return CheckResult.failed("sentence_integrity_across_pages", 2, violations)
    return CheckResult.passed("sentence_integrity_across_pages", 2, seeded=len(doc.must_contain))


@check(stage=2, dimension="dehyphenation")
def dehyphenation(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    body = art.body_text
    violations = []

    newline_breaks = re.findall(r"[a-z]-\s*\n\s*[a-z]", body)
    # Suspended compounds ('eighteenth- and nineteenth-century') are legal.
    space_breaks = re.findall(r"\b[a-z]{2,}- (?!(?:and|or|nor|und|oder|to)\b)[a-z]{2,}\b", body)
    if newline_breaks or len(space_breaks) >= 3:
        violations.append(Violation(
            message=f"hyphenation artifacts remain: {len(newline_breaks)} line-break, {len(space_breaks)} spaced"))

    for word in doc.keep_hyphens:
        if word.lower() not in body.lower():
            violations.append(Violation(
                message=f"legitimate hyphenated word lost (over-joined?): {word!r}"))

    if violations:
        return CheckResult.failed("dehyphenation", 2, violations)
    return CheckResult.passed("dehyphenation", 2)


@check(stage=2, dimension="paragraph_reflow")
def paragraph_reflow(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    """Runs on narrated text: front/back matter lists legitimately end lines
    without punctuation."""
    max_broken = doc.expect.get("paragraph_broken_max") or cfg.get("paragraph_broken_max", 0.05)
    body = "\n\n".join(c["text"] for c in art.chapters["chapters"]
                       if not c.get("front_matter"))
    paragraphs = [p.strip() for p in body.split("\n\n") if len(p.strip()) > 200]
    if not paragraphs:
        return CheckResult.skipped("paragraph_reflow", 2, "no long paragraphs found")
    broken = [p for p in paragraphs if normalize(p)[-1] not in ".!?\"'):;"]
    frac = len(broken) / len(paragraphs)
    if frac > max_broken:
        return CheckResult.failed("paragraph_reflow", 2, [Violation(
            message=f"{len(broken)}/{len(paragraphs)} paragraphs end mid-sentence ({frac:.1%})",
            context=broken[0][-120:])], broken_fraction=round(frac, 3))
    return CheckResult.passed("paragraph_reflow", 2, broken_fraction=round(frac, 3))
