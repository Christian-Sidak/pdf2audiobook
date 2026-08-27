"""Stage 3 (chapterize) checks, run against 03_chapters.json."""
from __future__ import annotations

import json
import re

from evals.artifacts import ArtifactSet
from evals.checks import check
from evals.contracts import CheckResult, Violation
from evals.corpus import DocSpec
from evals.textutil import normalize

_DOT_LEADER = re.compile(r"\.{3,}\s*\d+\s*$")
_FRONT_MATTER_MARKERS = [
    "all rights reserved", "library of congress", "isbn", "printed in the united states",
]


def _body_chapters(art: ArtifactSet) -> list[dict]:
    return [c for c in art.chapters["chapters"] if not c.get("front_matter")]


@check(stage=3, dimension="no_toc_leakage")
def no_toc_leakage(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    violations = []
    for ch in _body_chapters(art):
        dot_lines = [l for l in ch["text"].splitlines() if _DOT_LEADER.search(l)]
        if len(dot_lines) >= 3:
            violations.append(Violation(
                message=f"{len(dot_lines)} dot-leader TOC lines inside chapter {ch['title']!r}",
                unit_id=ch["id"], context=dot_lines[0][:120]))
        if any(normalize(l, casefold=True) == "contents" for l in ch["text"].splitlines()):
            violations.append(Violation(message=f"'Contents' heading inside chapter {ch['title']!r}",
                                        unit_id=ch["id"]))
    if violations:
        return CheckResult.failed("no_toc_leakage", 3, violations)
    return CheckResult.passed("no_toc_leakage", 3)


@check(stage=3, dimension="chapter_count", requires=("expect.chapter_count",))
def chapter_count(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    got = len(_body_chapters(art))
    want = int(doc.expect["chapter_count"])
    if got != want:
        titles = [c["title"] for c in _body_chapters(art)]
        return CheckResult.failed("chapter_count", 3, [Violation(
            message=f"{got} chapters, expected {want}: {titles[:12]}")])
    return CheckResult.passed("chapter_count", 3, count=got)


@check(stage=3, dimension="chapter_titles_vs_golden", requires=("golden.chapters",))
def chapter_titles_vs_golden(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    golden = json.loads(doc.golden_path("chapters").read_text())
    want = [normalize(t, casefold=True) for t in golden["titles"]]
    got = [normalize(c["title"], casefold=True) for c in _body_chapters(art)]
    if got != want:
        violations = []
        for i in range(max(len(got), len(want))):
            g = got[i] if i < len(got) else "<missing>"
            w = want[i] if i < len(want) else "<extra>"
            if g != w:
                violations.append(Violation(message=f"position {i}: got {g!r}, want {w!r}"))
        return CheckResult.failed("chapter_titles_vs_golden", 3, violations[:15])
    return CheckResult.passed("chapter_titles_vs_golden", 3, count=len(got))


@check(stage=3, dimension="front_matter_excluded")
def front_matter_excluded(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    violations = []
    for ch in _body_chapters(art):
        low = ch["text"].lower()
        for marker in _FRONT_MATTER_MARKERS:
            if marker in low:
                violations.append(Violation(
                    message=f"front-matter marker {marker!r} inside chapter {ch['title']!r}",
                    unit_id=ch["id"]))
    if violations:
        return CheckResult.failed("front_matter_excluded", 3, violations[:10])
    return CheckResult.passed("front_matter_excluded", 3)


@check(stage=3, dimension="body_chapter_quality")
def body_chapter_quality(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    """Narrated text must read as language: catches OCR garbage that survived
    into body chapters (front matter legitimately scores low and is excluded)."""
    from pipeline.textquality import dict_word_ratio, garbage_density

    ratio_min = doc.expect.get("body_dict_word_ratio_min") or cfg.get("body_dict_word_ratio_min", 0.75)
    garbage_max = cfg.get("garbage_char_density_max", 0.02)
    violations = []
    for ch in _body_chapters(art):
        garbage = garbage_density(ch["text"])
        # Short chapters get a laxer dictionary gate rather than none: a
        # skipped check let an OCR title page ship as narrated chapter zero.
        ratio = dict_word_ratio(ch["text"])
        effective_min = ratio_min if len(ch["text"]) >= 1500 else min(ratio_min, 0.55)
        if ratio < effective_min or garbage > garbage_max:
            violations.append(Violation(
                message=f"chapter {ch['title']!r}: dict ratio {ratio:.2f}, garbage {garbage:.3f}",
                unit_id=ch["id"]))
    if violations:
        return CheckResult.failed("body_chapter_quality", 3, violations)
    return CheckResult.passed("body_chapter_quality", 3)


@check(stage=3, dimension="chapter_text_coverage")
def chapter_text_coverage(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    minimum = cfg.get("chapter_text_coverage_min", 0.97)
    body_len = len(normalize(art.body_text))
    if body_len == 0:
        return CheckResult.failed("chapter_text_coverage", 3, [Violation(message="empty body text")])
    # Include front-matter chapters and deliberately excluded navigation text
    # (mini-TOCs, section outlines): classification is not text loss.
    chapters_len = sum(len(normalize(c["text"])) for c in art.chapters["chapters"])
    chapters_len += sum(e.get("chars", 0) for e in art.chapters.get("excluded", []))
    coverage = chapters_len / body_len
    if coverage < minimum:
        return CheckResult.failed("chapter_text_coverage", 3, [Violation(
            message=f"chapters cover {coverage:.1%} of body text (min {minimum:.0%})")],
            coverage=round(coverage, 4))
    return CheckResult.passed("chapter_text_coverage", 3, coverage=round(coverage, 4))
