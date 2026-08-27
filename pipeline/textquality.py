"""Per-page text quality metrics for the OCR gate.

Shared by stage 1 (to decide which pages need OCR) and the evals
(ocr_gate_flags / body_chapter_quality checks). Uses the macOS system
dictionary to avoid a model dependency.
"""
from __future__ import annotations

import re
from functools import lru_cache

DICT_PATHS = ["/usr/share/dict/words"]

# True extraction-failure markers. Scholarly text legitimately contains
# Hebrew, Arabic, Greek, math, and transliteration marks, so garbage is an
# explicit denylist (replacement/control/private-use chars), not the
# complement of an ASCII whitelist.
_GARBAGE = re.compile(
    "[\ufffd"                                  # replacement character
    "\u0000-\u0008\u000b\u000c\u000e-\u001f"  # control chars (not tab/newline)
    "\ue000-\uf8ff]"                          # private use area
)
_TOKEN = re.compile(r"[A-Za-z]{2,}")


@lru_cache(maxsize=1)
def _dictionary() -> frozenset[str]:
    words: set[str] = set()
    for p in DICT_PATHS:
        try:
            with open(p, encoding="utf-8", errors="ignore") as f:
                words.update(w.strip().lower() for w in f)
        except OSError:
            continue
    return frozenset(words)


# /usr/share/dict/words holds base forms only ('tear', 'stream', 'approach'),
# so inflected everyday prose scores as non-words without suffix stripping.
_SUFFIXES = ("s", "es", "ed", "d", "ing", "ly")


def is_word(token: str, d: frozenset[str] | None = None) -> bool:
    d = d if d is not None else _dictionary()
    if token in d:
        return True
    for suf in _SUFFIXES:
        if not token.endswith(suf):
            continue
        base = token[: -len(suf)]
        if len(base) < 2:
            continue
        if base in d or (base + "e") in d:
            return True
        # doubled final consonant: stopped -> stopp -> stop
        if len(base) >= 3 and base[-1] == base[-2] and base[:-1] in d:
            return True
        # y -> i mutation: carried -> carri -> carry
        if base.endswith("i") and (base[:-1] + "y") in d:
            return True
    return False


def dict_word_ratio(text: str) -> float:
    """Fraction of alphabetic tokens found in the dictionary. 1.0 for empty."""
    tokens = [t.lower() for t in _TOKEN.findall(text)]
    if not tokens:
        return 1.0
    d = _dictionary()
    return sum(1 for t in tokens if is_word(t, d)) / len(tokens)


def garbage_density(text: str) -> float:
    """Fraction of characters that are extraction-failure markers."""
    if not text:
        return 0.0
    return len(_GARBAGE.findall(text)) / len(text)


def page_metrics(text: str) -> dict:
    return {
        "chars": len(text),
        "dict_word_ratio": round(dict_word_ratio(text), 4),
        "garbage_density": round(garbage_density(text), 4),
    }
