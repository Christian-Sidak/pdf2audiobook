"""Text normalization, sentence splitting, alignment, and WER."""
from __future__ import annotations

import re

_QUOTE_MAP = str.maketrans({
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "–": "-", "—": "-", "−": "-",
    " ": " ", "ﬁ": "fi", "ﬂ": "fl",
})


def normalize(text: str, casefold: bool = False) -> str:
    """Whitespace-collapse and unify quotes/dashes/ligatures for comparison."""
    text = text.translate(_QUOTE_MAP)
    text = text.replace("­", "")  # soft hyphen: invisible for comparison
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold() if casefold else text


def normalize_line(line: str) -> str:
    return normalize(line, casefold=True)


def contains_normalized(haystack: str, needle: str) -> bool:
    return normalize(needle, casefold=True) in normalize(haystack, casefold=True)


_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")

_ABBREV = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|vs|etc|cf|ed|vol|no|pp|ca|e\.g|i\.e)\.$", re.IGNORECASE
)


def sentence_split(text: str) -> list[str]:
    """Regex sentence splitter, good enough for alignment (not for TTS)."""
    parts = _SENT_SPLIT.split(normalize(text))
    # Rejoin splits caused by common abbreviations.
    out: list[str] = []
    for part in parts:
        if out and _ABBREV.search(out[-1]):
            out[-1] = out[-1] + " " + part
        else:
            out.append(part)
    return [p.strip() for p in out if p.strip()]


def word_tokens(text: str) -> list[str]:
    # Fold diacritics: ASR spells transliterations without them ('Hicrī' vs
    # 'Hicri'); the audio is not wrong.
    import unicodedata
    folded = unicodedata.normalize("NFKD", normalize(text, casefold=True))
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9']+", folded)


_SMALL_NUMS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen",
    "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen",
    "19": "nineteen", "20": "twenty",
}


def asr_normalize(text: str) -> str:
    """Normalize ASR output for WER: small digits to words ('Chapter 1' vs
    'Chapter One' is not an audio error)."""
    return re.sub(r"\b(\d{1,2})\b", lambda m: _SMALL_NUMS.get(m.group(1), m.group(1)), text)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate via Levenshtein distance over normalized word tokens."""
    ref = word_tokens(reference)
    hyp = word_tokens(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def repeated_lines(pages: list[str], min_fraction: float = 0.2, min_len: int = 4) -> dict[str, int]:
    """Normalized lines that occur on >= min_fraction of pages.

    Candidate running headers/footers. Lines that are pure page numbers are
    excluded (they have their own check).
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    for page in pages:
        seen = set()
        for line in page.splitlines():
            n = normalize_line(line)
            if len(n) >= min_len and not re.fullmatch(r"[\divxlc]+", n):
                seen.add(n)
        counts.update(seen)
    threshold = max(2, int(len(pages) * min_fraction))
    return {line: c for line, c in counts.items() if c >= threshold}


def excerpt(text: str, needle: str, radius: int = 120) -> str:
    """Return +-radius chars around the first occurrence of needle."""
    idx = text.find(needle)
    if idx < 0:
        return needle[:radius]
    return text[max(0, idx - radius): idx + len(needle) + radius].replace("\n", " ")
