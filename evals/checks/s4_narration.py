"""Stage 4 (narration rewrite) checks."""
from __future__ import annotations

import json
import re

from rapidfuzz import fuzz, process

from evals.artifacts import ArtifactSet
from evals.checks import check
from evals.contracts import CheckResult, Violation
from evals.corpus import DocSpec
from evals.textutil import sentence_split
from pipeline.config import PAUSE_POLICY, SEGMENT_LENGTH_LIMITS

# Deletions the rewrite is always allowed to make, independent of per-doc
# allowlists: citation apparatus and reference debris.
DEFAULT_DELETION_PATTERNS = [
    r"^\(?(ibid|op\.\s*cit|loc\.\s*cit)\b.*\)?$",
    r"^\[?\d+([,\s\-]+\d+)*\]?$",
    r"^\(?(see|cf)\b[^.]{0,80}\)?\.?$",
    r"^\(?pp?\.\s*\d.*\)?$",
]


def _script(art: ArtifactSet) -> dict:
    return art.narration


def _body_chapters(art: ArtifactSet) -> list[dict]:
    return [c for c in art.chapters["chapters"] if not c.get("front_matter")]


@check(stage=4, dimension="narration_schema_valid")
def narration_schema_valid(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    from pipeline.ir import NarrationScript
    script = _script(art)  # ArtifactMissing propagates -> skip, not fail
    try:
        NarrationScript.model_validate(script)
    except Exception as e:
        return CheckResult.failed("narration_schema_valid", 4, [Violation(message=str(e)[:400])])
    return CheckResult.passed("narration_schema_valid", 4, segments=len(_script(art)["segments"]))


@check(stage=4, dimension="no_surviving_digits")
def no_surviving_digits(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    violations = []
    for seg in _script(art)["segments"]:
        # Vulgar-fraction glyphs are numerals for the ear even though they
        # are not \d ('Platform 9¾').
        digits = re.findall(r"\d+|[¼½¾⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞]", seg["text"])
        if digits:
            violations.append(Violation(
                message=f"digits {digits[:5]} in segment: {seg['text'][:90]!r}",
                unit_id=seg["id"], context=seg["text"][:200], fixable=True))
    if violations:
        return CheckResult.failed("no_surviving_digits", 4, violations[:20], total=len(violations))
    return CheckResult.passed("no_surviving_digits", 4)


@check(stage=4, dimension="segment_text_quality")
def segment_text_quality(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    """Narration must read as language. Passthrough-fallback windows can ship
    raw OCR debris that faithfulness cannot flag (it is faithful to the
    garbage); the dictionary gate catches it."""
    from pipeline.textquality import dict_word_ratio

    ratio_min = cfg.get("segment_dict_word_ratio_min", 0.55)
    violations = []
    for seg in _script(art)["segments"]:
        if seg["type"] == "chapter_heading" or len(seg["text"]) < 40:
            continue
        ratio = dict_word_ratio(_numeric_blind(seg["text"]))
        if ratio < ratio_min:
            violations.append(Violation(
                message=f"unreadable narration (dict ratio {ratio:.2f}): {seg['text'][:80]!r}",
                unit_id=seg["id"], context=seg["text"][:200], fixable=True))
    if violations:
        return CheckResult.failed("segment_text_quality", 4, violations[:20], total=len(violations))
    return CheckResult.passed("segment_text_quality", 4)


_ARITH_YEAR = re.compile(
    r"\bone thousand,?(?: and)? (?:\w+ hundred|(?:twenty|thirty|forty|fifty|sixty|"
    r"seventy|eighty|ninety)[- ]\w+)\b", re.IGNORECASE)
_SRC_ERA_YEAR = re.compile(r"\b(?:AD|A\.D\.|BC|B\.C\.|CE|C\.E\.)\s?(1\d{3})\b")
_SRC_COMMA_QTY = re.compile(r"\b(\d{1,3},\d{3})\b")


@check(stage=4, dimension="number_style")
def number_style(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    """Year/quantity style conventions in narration.

    (a) Invariant: arithmetic year forms ('one thousand four hundred...')
    must never survive; the deterministic fixer guarantees it, this verifies.
    (b) Contextual: source orthography classifies numerals (era marker ->
    spoken year pairing; thousands-comma -> arithmetic quantity); each mention
    is traced through source_span to its narration segment and the style
    checked there."""
    script = _script(art)
    violations = []

    for seg in script["segments"]:
        m = _ARITH_YEAR.search(seg["text"])
        if m:
            violations.append(Violation(
                message=f"arithmetic year style {m.group(0)!r}: {seg['text'][:80]!r}",
                unit_id=seg["id"], context=seg["text"][:200], fixable=True))

    checked = 0
    chapters = {c["id"]: c["text"] for c in _body_chapters(art)}
    segs_by_ch: dict[str, list[dict]] = {}
    for s in script["segments"]:
        segs_by_ch.setdefault(s["chapter_id"], []).append(s)

    def covering(ch_id: str, pos: int) -> dict | None:
        for s in segs_by_ch.get(ch_id, []):
            span = s.get("source_span")
            if span and span[0] <= pos < span[1]:
                return s
        return None

    for ch_id, text in chapters.items():
        for m in _SRC_ERA_YEAR.finditer(text):
            seg = covering(ch_id, m.start())
            if seg is None:
                continue
            checked += 1
            if re.search(r"\bthousand\b", seg["text"], re.IGNORECASE):
                violations.append(Violation(
                    message=f"era-marked year {m.group(0)!r} rendered arithmetically",
                    unit_id=seg["id"], context=seg["text"][:200], fixable=True))
        for m in _SRC_COMMA_QTY.finditer(text):
            seg = covering(ch_id, m.start())
            if seg is None:
                continue
            checked += 1
            # A comma quantity must NOT be paired like a year ('ten sixty-six').
            if re.search(r"\b(?:ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
                         r"seventeen|eighteen|nineteen) (?:oh-\w+|(?:twenty|thirty|forty|"
                         r"fifty|sixty|seventy|eighty|ninety))", seg["text"], re.IGNORECASE):
                violations.append(Violation(
                    message=f"comma quantity {m.group(0)!r} rendered as paired year",
                    unit_id=seg["id"], context=seg["text"][:200], fixable=True))

    if violations:
        return CheckResult.failed("number_style", 4, violations[:20],
                                  total=len(violations), contextual_checked=checked)
    return CheckResult.passed("number_style", 4, contextual_checked=checked)


@check(stage=4, dimension="no_bare_roman_numerals")
def no_bare_roman_numerals(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    """Roman numerals evade digit checks and TTS reads 'III' as letters."""
    from pipeline.s4_narration import _ROMAN_TOKEN

    pattern = re.compile(rf"\b(?:[A-Z][a-z]{{2,}}|Chapter|Part|Volume|Book|Act|Section)"
                         rf"\s+(?:{_ROMAN_TOKEN})\b(?!\.)")
    violations = []
    for seg in _script(art)["segments"]:
        m = pattern.search(seg["text"])
        if m:
            violations.append(Violation(
                message=f"bare roman numeral {m.group(0)!r} in segment: {seg['text'][:80]!r}",
                unit_id=seg["chapter_id"], context=seg["text"][:200], fixable=True))
    if violations:
        return CheckResult.failed("no_bare_roman_numerals", 4, violations[:20], total=len(violations))
    return CheckResult.passed("no_bare_roman_numerals", 4)


@check(stage=4, dimension="segment_length_limits")
def segment_length_limits(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    violations = []
    for seg in _script(art)["segments"]:
        lo, hi = SEGMENT_LENGTH_LIMITS.get(seg["type"], (1, 900))
        if not (lo <= len(seg["text"]) <= hi):
            violations.append(Violation(
                message=f"{seg['type']} length {len(seg['text'])} outside [{lo},{hi}]: {seg['text'][:80]!r}",
                unit_id=seg["id"], fixable=True))
    if violations:
        return CheckResult.failed("segment_length_limits", 4, violations[:20], total=len(violations))
    return CheckResult.passed("segment_length_limits", 4)


# pause_policy_bounds retired with IR v2: pauses are no longer segment data;
# silence_gaps_vs_policy (stage 6) verifies the assembled timeline against
# the live policy instead.


def _allowed_deletion(sentence: str, doc: DocSpec) -> bool:
    s = sentence.strip()
    for pat in DEFAULT_DELETION_PATTERNS + doc.deletion_allowlist:
        if re.match(pat, s, re.IGNORECASE):
            return True
    return False


_NUM_WORDS_RE = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|"
    r"point|percent|first|second|third|fourth|fifth|\d+)\b", re.IGNORECASE)


def _numeric_blind(text: str) -> str:
    """Verbalized numbers ('zero point eight six') cannot fuzzy-match their
    digit source ('0.86'); compare the prose skeleton instead."""
    return _NUM_WORDS_RE.sub(" ", text)


def _chapter_rewrite_sentences(script: dict, chapter_id: str) -> list[str]:
    texts = [s["text"] for s in script["segments"]
             if s["chapter_id"] == chapter_id and s["type"] in ("paragraph", "blockquote")]
    return sentence_split(" ".join(texts))


@check(stage=4, dimension="faithfulness_aligned_diff")
def faithfulness_aligned_diff(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    min_sim = cfg.get("faithfulness_min_similarity", 60)
    min_words = cfg.get("faithfulness_min_words", 8)
    script = _script(art)
    violations = []
    ambiguous: list[tuple[str, str, str]] = []  # (chapter_id, src, best_match) for the judge

    narrated = {s["chapter_id"] for s in script["segments"]}
    skipped_chapters = []
    for ch in _body_chapters(art):
        if ch["id"] not in narrated:
            skipped_chapters.append(ch["id"])  # partial build: chapter not narrated
            continue
        rewrite_sents = _chapter_rewrite_sentences(script, ch["id"])
        if not rewrite_sents:
            violations.append(Violation(message=f"chapter {ch['title']!r} has no rewrite output",
                                        unit_id=ch["id"], fixable=True))
            continue
        blind_rewrites = [_numeric_blind(s) for s in rewrite_sents]
        for src in sentence_split(ch["text"]):
            if len(src.split()) <= min_words or _allowed_deletion(src, doc):
                continue
            best = process.extractOne(_numeric_blind(src), blind_rewrites,
                                      scorer=fuzz.token_set_ratio, score_cutoff=min_sim)
            if best is None:
                violations.append(Violation(
                    message=f"source sentence dropped: {src[:100]!r}",
                    unit_id=ch["id"], context=src[:300], fixable=True))
            elif best[1] < 85:
                ambiguous.append((ch["id"], src, best[0]))

    details = {"ambiguous_pairs": len(ambiguous), "not_narrated": skipped_chapters}
    if violations:
        return CheckResult.failed("faithfulness_aligned_diff", 4, violations[:25],
                                  total=len(violations), **details)
    return CheckResult.passed("faithfulness_aligned_diff", 4, **details)


@check(stage=4, dimension="no_hallucination_judge", deterministic=False)
def no_hallucination_judge(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    """Judge rewrite sentences with no plausible source: hallucination hunt."""
    from evals.llm_judge import judge, judge_available

    min_sim = cfg.get("faithfulness_min_similarity", 60)
    script = _script(art)

    candidates: list[tuple[str, str, str]] = []
    for ch in _body_chapters(art):
        src_sents = sentence_split(ch["text"])
        if not src_sents:
            continue
        blind_srcs = [_numeric_blind(s) for s in src_sents]
        for rw in _chapter_rewrite_sentences(script, ch["id"]):
            if len(rw.split()) <= 15:
                continue
            best = process.extractOne(_numeric_blind(rw), blind_srcs,
                                      scorer=fuzz.token_set_ratio, score_cutoff=min_sim)
            if best is None:
                candidates.append((ch["id"], rw, ch["text"][:1500]))

    if not candidates:
        return CheckResult.passed("no_hallucination_judge", 4, judged=0)
    if not judge_available():
        return CheckResult.skipped("no_hallucination_judge", 4, "ollama unreachable")

    violations = []
    for ch_id, rw, src_context in candidates[:10]:
        verdict = judge(source=src_context, rewrite=rw,
                        question="Does the rewrite passage contain content not present in the source (a hallucination)?")
        if verdict["verdict"] == "hallucination":
            violations.append(Violation(
                message=f"judged hallucination: {rw[:100]!r} ({verdict['reason'][:120]})",
                unit_id=ch_id, context=rw[:300], fixable=True))
    if violations:
        return CheckResult.failed("no_hallucination_judge", 4, violations,
                                  judged=min(10, len(candidates)), candidates=len(candidates))
    return CheckResult.passed("no_hallucination_judge", 4,
                              judged=min(10, len(candidates)), candidates=len(candidates))
