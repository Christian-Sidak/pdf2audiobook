"""Stage 4: narration rewrite via local Qwen (Ollama).

Converts each body chapter into the typed narration script IR. The LLM
rewrites windows of paragraphs for the ear (verbalized numbers and dates,
expanded abbreviations, dropped citations) and tags segment types. Chapter
headings are generated deterministically from the chapter tree; pauses come
from the policy table, never from the model.

Artifact: 04_narration.json
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from evals.contracts import ARTIFACT_FILES, BookCtx, Violation
from pipeline.config import PAUSE_POLICY, REWRITE_MODEL, ROOT
from pipeline.ir import NarrationScript, Segment
from pipeline.ollama_client import chat_json

from pipeline.config import CFG

WINDOW_PARAGRAPHS = int(CFG["narration"]["window_paragraphs"])
WINDOW_RETRIES = int(CFG["narration"]["window_retries"])
MAX_SEGMENT_CHARS = int(CFG["narration"]["max_segment_chars"])

REWRITE_SCHEMA = {
    "type": "object",
    "required": ["segments"],
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "text"],
                "properties": {
                    "type": {"type": "string",
                             "enum": ["section_heading", "paragraph", "blockquote"]},
                    "text": {"type": "string"},
                },
            },
        },
    },
}

SYSTEM_PROMPT = """You are preparing scholarly book text for audiobook narration. Rewrite the given paragraphs into narration segments.

Rules:
- PRESERVE all narrative content, names, arguments, and quotations. Do not summarize, do not skip sentences, do not add anything.
- Verbalize ALL digits and numbers as words ("1204" -> "twelve oh four" for years, "twelve hundred and four" for quantities; "3rd" -> "third"; "70%" -> "seventy percent"). No digit characters may remain.
- Expand abbreviations for the ear: "e.g." -> "for example", "i.e." -> "that is", "cf." -> "compare", "ca." -> "around", "St." -> "Saint", "MS" -> "manuscript".
- Roman numerals become words: regnal names as ordinals ("Selim III" -> "Selim the Third"), structural references as cardinals ("Part II" -> "Part Two").
- Years use the spoken pairing convention: "1066" -> "ten sixty-six", "1431" -> "fourteen thirty-one", "1200" -> "twelve hundred", "AD 1001" -> "AD ten oh-one" (never "one thousand and..." for years).
- Numbers with thousands-commas are QUANTITIES, never years: "1,001 nights" -> "one thousand and one nights"; "2,500 men" -> "two thousand five hundred men".
- Drop pure citation apparatus: parenthetical references like "(ibid.)", "(op. cit.)", bracketed reference numbers, page-number references. Keep substantive parenthetical asides, folding them into the sentence naturally.
- Mark a segment "section_heading" only if it is clearly a heading (short title line, numbered section). Mark long verbatim quotations as "blockquote". Everything else is "paragraph".
- Keep paragraphs under 900 characters; split long paragraphs at sentence boundaries.
- Foreign terms and transliterations: keep them, spelled as-is.

Return JSON: {"segments": [{"type": ..., "text": ...}, ...]} in reading order."""

_NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
              "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
              "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}


def _int_words(n: int) -> str:
    if n < 20:
        return _NUM_WORDS[n]
    if n < 100:
        return _TENS[n // 10] + ("-" + _NUM_WORDS[n % 10] if n % 10 else "")
    return str(n)  # headings never need more


def _int_full(n: int) -> str:
    """Any integer to words. Passthrough windows skip the LLM, so this is the
    deterministic replacement for its number verbalization."""
    if n < 100:
        return _int_words(n)
    if n >= 10 ** 12:  # digit-by-digit beyond common magnitudes
        return " ".join(_NUM_WORDS[int(c)] for c in str(n))
    for div, name in ((10 ** 9, "billion"), (10 ** 6, "million"),
                      (1000, "thousand"), (100, "hundred")):
        if n >= div:
            rem = n % div
            head = f"{_int_full(n // div)} {name}"
            return head + (f" {_int_full(rem)}" if rem else "")
    return _int_words(n)


def _year_pair_words(n: int) -> str:
    """Spoken year convention: 1961 -> 'nineteen sixty-one', 1905 ->
    'nineteen oh-five', 1900 -> 'nineteen hundred', 2005 -> 'two thousand
    five'."""
    hi, lo = divmod(n, 100)
    if hi == 20:
        return "two thousand" + (f" {_int_words(lo)}" if lo else "")
    if lo == 0:
        return f"{_int_words(hi)} hundred"
    if lo < 10:
        return f"{_int_words(hi)} oh-{_NUM_WORDS[lo]}"
    return f"{_int_words(hi)} {_int_words(lo)}"


_TENS_ORDINAL = {2: "twentieth", 3: "thirtieth", 4: "fortieth", 5: "fiftieth",
                 6: "sixtieth", 7: "seventieth", 8: "eightieth", 9: "ninetieth"}


def _ordinal_words(n: int) -> str:
    if n <= 20:
        return _ORDINALS[n].lower()
    if n < 100:
        tens, unit = divmod(n, 10)
        if unit == 0:
            return _TENS_ORDINAL[tens]
        return f"{_TENS[tens]}-{_ORDINALS[unit].lower()}"
    return f"{_int_full(n)}th"  # rare in prose; still digit-free


def verbalize_digits(text: str) -> str:
    """Deterministic digits-to-words for passthrough windows: money, ordinals,
    decimals, comma groups, year-shaped numbers, then bare integers. The
    no_surviving_digits gate requires narration to be digit-free."""
    text = re.sub(r"\$\s?(\d[\d,]*)(?:\.(\d\d))?",
                  lambda m: _int_full(int(m.group(1).replace(",", ""))) + " dollars"
                  + (f" and {_int_full(int(m.group(2)))} cents" if m.group(2) else ""),
                  text)
    text = re.sub(r"\b(\d{1,3})(?:st|nd|rd|th)\b",
                  lambda m: _ordinal_words(int(m.group(1))), text)
    text = re.sub(r"\b(\d+)\.(\d+)\b",
                  lambda m: _int_full(int(m.group(1))) + " point "
                  + " ".join(_NUM_WORDS[int(c)] for c in m.group(2)), text)
    text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b",
                  lambda m: _int_full(int(m.group(0).replace(",", ""))), text)
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)  # '18ff' -> '18 ff'
    text = re.sub(r"\b(1[0-9]\d\d|20\d\d)\b",
                  lambda m: _year_pair_words(int(m.group(0))), text)
    return re.sub(r"\d+", lambda m: _int_full(int(m.group(0))), text)


_NUMBER_TOKEN_WORDS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty "
    "sixty seventy eighty ninety hundred thousand million billion point oh and "
    "dollars cents first second third fourth fifth sixth seventh eighth ninth tenth "
    "eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth "
    "eighteenth nineteenth twentieth thirtieth fortieth fiftieth sixtieth "
    "seventieth eightieth ninetieth".split())

VERBALIZE_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}},
                    "required": ["text"]}

VERBALIZE_PROMPT = """You are preparing text for audiobook narration. Replace every \
numeral in the passage with the words a narrator would speak, reading each number as \
its context demands: a year like 1810 as 'eighteen ten', a quantity like 1810 soldiers \
as 'one thousand eight hundred and ten', money, ordinals, page references, decimals \
each in their natural spoken form. Change NOTHING else: every word that is not a \
number must remain exactly as written, in the same order.

PASSAGE:
{text}

Return JSON: {{"text": "the passage with numbers spoken out"}}"""


def _non_number_skeleton(text: str) -> list[str]:
    """Token sequence with numerals and number-words removed: two texts that
    differ only in how numbers are written share a skeleton."""
    toks = re.findall(r"[A-Za-z]+|\d+", text.lower().replace("-", " "))
    return [t for t in toks if not t.isdigit() and t not in _NUMBER_TOKEN_WORDS]


_UNIT_VALUES = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS_VALUES = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
                "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALE_VALUES = {"hundred": 100, "thousand": 1000,
                 "million": 10 ** 6, "billion": 10 ** 9}
@lru_cache(maxsize=1)
def _ordinal_values() -> dict[str, int]:
    # Built lazily: _ORDINALS and _TENS_ORDINAL are defined further down.
    return {**{w.lower(): i for i, w in enumerate(_ORDINALS) if w},
            **{w: t * 10 for t, w in _TENS_ORDINAL.items()}}


@lru_cache(maxsize=1)
def _run_tokens() -> frozenset[str]:
    return frozenset(set(_UNIT_VALUES) | set(_TENS_VALUES) | set(_SCALE_VALUES)
                     | set(_ordinal_values()) | {"oh", "point", "and"})


def _small_value(t: str) -> int | None:
    if t == "oh":
        return 0
    return _UNIT_VALUES.get(t, _TENS_VALUES.get(t, _ordinal_values().get(t)))


def _parse_spoken_run(toks: list[str]) -> list[str]:
    """One contiguous run of spoken-number words -> canonical value strings.
    Handles both scale form ('one thousand eight hundred and ten' -> 1810)
    and pair form ('eighteen ten' -> 1810, 'nineteen oh five' -> 1905)."""
    toks = [t for t in toks if t != "and"]
    if not toks:
        return []
    if "point" in toks:  # decimal: '<int> point <digit> <digit>'
        i = toks.index("point")
        head = _parse_spoken_run(toks[:i])
        tail = [_small_value(t) for t in toks[i + 1:]]
        if len(head) == 1 and tail and all(v is not None and v < 10 for v in tail):
            return [head[0] + "." + "".join(str(v) for v in tail)]
        return head + [str(v) for v in tail if v is not None]

    if any(t in _SCALE_VALUES for t in toks):  # standard two-register parse
        total, cur = 0, 0
        for t in toks:
            if t == "hundred":
                cur = max(cur, 1) * 100
            elif t in _SCALE_VALUES:
                total += max(cur, 1) * _SCALE_VALUES[t]
                cur = 0
            else:
                cur += _small_value(t) or 0
        return [str(total + cur)]

    values, cur = [], None  # scale-free: units, tens compounds, year pairs
    for t in toks:
        v = _small_value(t)
        if v is None:
            continue
        if cur is None:
            cur = v
        elif t == "oh" and 10 <= cur <= 99:
            cur *= 100                      # 'nineteen oh' ...
        elif cur >= 100 and v < 100 and \
                (cur % 100 == 0 or (cur % 10 == 0 and v <= 9)):
            cur += v            # 'nineteen oh' + 'five'; 'nineteen sixty' + 'one'
        elif cur in _TENS_VALUES.values() and 1 <= v <= 9:
            cur += v                        # 'twenty' + 'one'
        elif 10 <= cur <= 99 and 10 <= v <= 99:
            cur = cur * 100 + v             # 'eighteen' + 'ten' -> 1810
        else:
            values.append(str(cur))
            cur = v
    if cur is not None:
        values.append(str(cur))
    return values


def _number_values(text: str) -> list[str]:
    """Numbers in a text, in order, as canonical strings: digits ('1961',
    '3.5') and spoken forms ('nineteen sixty-one', 'three point five') yield
    the same values, so an LLM verbalization can be checked for value errors
    against its source."""
    tokens = re.findall(r"\d[\d,]*(?:\.\d+)?|[a-z]+", text.lower().replace("-", " "))
    values: list[str] = []
    run: list[str] = []
    for t in tokens:
        if t[0].isdigit():
            values.extend(_parse_spoken_run(run))
            run = []
            values.append(t.replace(",", ""))
        elif t in _run_tokens():
            run.append(t)
        else:
            values.extend(_parse_spoken_run(run))
            run = []
    values.extend(_parse_spoken_run(run))
    return values


OPENING_DEDUPE_SCHEMA = {
    "type": "object",
    "properties": {"duplicates": {"type": "array", "items": {"type": "integer"}}},
    "required": ["duplicates"],
}

OPENING_DEDUPE_PROMPT = (
    "An audiobook chapter opens with these narration segments, in reading "
    "order:\n\n{listing}\n\n"
    "Segment 0 is the announced chapter title. Printed pages often repeat "
    "the title (half-title lines, running heads, OCR-damaged variants), so "
    "the same title would be spoken twice in a row with a pause between. "
    "Which segments after 0 are merely a repeat of the chapter title, adding "
    "no content of their own? OCR damage counts as a repeat if a listener "
    "would hear it as the title again. "
    'Reply as JSON: {{"duplicates": [indices]}} — an empty list if none.'
)


def dedupe_opening_titles(segments: list[dict], model: str) -> list[dict]:
    """Drop chapter-opening headings that just re-speak the announced title.
    Whether a mangled heading IS the title again is a judgment call (OCR
    damage defeats string equality), so the LLM decides and deterministic
    code only validates: only short heading-like segments among a chapter's
    first few may be dropped, never the announcement itself, never a
    paragraph with real content. Fallback: normalized exact-match dedupe."""
    from pipeline.ollama_client import OllamaError, chat_json

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", s.casefold()).strip()

    drop: set[int] = set()
    i = 0
    while i < len(segments):
        if segments[i]["type"] != "chapter_heading":
            i += 1
            continue
        ch_id = segments[i]["chapter_id"]
        head_end = i + 1
        while head_end < len(segments) and head_end - i < 5 \
                and segments[head_end]["chapter_id"] == ch_id:
            head_end += 1
        head = segments[i:head_end]
        # Droppable: heading-type or a title-short paragraph, never index 0.
        candidates = {j for j in range(1, len(head))
                      if head[j]["type"] in ("chapter_heading", "section_heading")
                      or len(head[j]["text"]) < 90}
        if candidates:
            listing = "\n".join(f"{j}. [{s['type']}] {s['text']}"
                                for j, s in enumerate(head))
            try:
                result = chat_json(
                    model,
                    [{"role": "user",
                      "content": OPENING_DEDUPE_PROMPT.format(listing=listing)}],
                    OPENING_DEDUPE_SCHEMA, temperature=0.0)
                picked = {j for j in result.get("duplicates", [])
                          if isinstance(j, int)}
            except OllamaError:
                picked = {j for j in range(1, len(head))
                          if norm(head[j]["text"]) == norm(head[0]["text"])}
            drop.update(i + j for j in picked & candidates)
        i = head_end
    return [s for j, s in enumerate(segments) if j not in drop]


def llm_verbalize_numbers(text: str, model: str) -> str:
    """Context-aware number verbalization: how a number is read is a judgment
    call ('in 1810' vs '1810 soldiers'), so the LLM decides and deterministic
    code only validates: output must be digit-free and word-identical outside
    the numbers, else fall back to the mechanical verbalizer."""
    from pipeline.ollama_client import OllamaError, chat_json

    if not re.search(r"\d", text):
        return text
    fallback = verbalize_digits(verbalize_fractions(text))
    try:
        result = chat_json(model, [{"role": "user", "content":
                                    VERBALIZE_PROMPT.format(text=text)}],
                           VERBALIZE_SCHEMA, temperature=0.0)
    except OllamaError:
        return fallback
    out = re.sub(r"\s+", " ", str(result.get("text", ""))).strip()
    if not out or re.search(r"\d", out) \
            or _non_number_skeleton(out) != _non_number_skeleton(text) \
            or _number_values(out) != _number_values(text):
        return fallback
    return out


def _roman_to_int(s: str) -> int | None:
    s = s.lower()
    if not s or any(c not in _ROMAN for c in s):
        return None
    total = 0
    for i, c in enumerate(s):
        v = _ROMAN[c]
        total += -v if i + 1 < len(s) and _ROMAN[s[i + 1]] > v else v
    return total


_VULGAR_FRACTIONS = {
    "¼": "one quarter", "½": "one half", "¾": "three quarters",
    "⅓": "one third", "⅔": "two thirds",
    "⅕": "one fifth", "⅖": "two fifths", "⅗": "three fifths", "⅘": "four fifths",
    "⅙": "one sixth", "⅚": "five sixths",
    "⅛": "one eighth", "⅜": "three eighths", "⅝": "five eighths", "⅞": "seven eighths",
}
_FRACTION_WORDS = {2: "half", 3: "third", 4: "quarter", 5: "fifth", 6: "sixth",
                   7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"}


def verbalize_fractions(text: str) -> str:
    """'Platform 9¾' -> 'Platform nine and three quarters'; '2½' -> 'two and a
    half'; standalone '¾' -> 'three quarters'; ASCII '9 3/4' likewise.
    Fraction glyphs are not digits, so digit checks alone never catch them."""

    def unicode_frac(m: re.Match) -> str:
        whole, glyph = m.group(1), m.group(2)
        frac = _VULGAR_FRACTIONS[glyph]
        if not whole:
            return frac
        whole = whole.strip()
        return f"{_int_words(int(whole))} and {frac}" if int(whole) < 100 else f"{whole} and {frac}"

    text = re.sub(rf"(\d+\s?)?([{''.join(_VULGAR_FRACTIONS)}])", unicode_frac, text)

    def ascii_frac(m: re.Match) -> str:
        whole, num, den = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if den not in _FRACTION_WORDS or whole >= 100 or num >= den:
            return m.group(0)
        unit = _FRACTION_WORDS[den] + ("s" if num > 1 else "")
        num_word = "a" if num == 1 else _int_words(num)
        return f"{_int_words(whole)} and {num_word} {unit}"

    return re.sub(r"\b(\d+)\s(\d)/(\d\d?)\b", ascii_frac, text)


_ORDINALS = ["", "First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh",
             "Eighth", "Ninth", "Tenth", "Eleventh", "Twelfth", "Thirteenth",
             "Fourteenth", "Fifteenth", "Sixteenth", "Seventeenth", "Eighteenth",
             "Nineteenth", "Twentieth"]

_ROMAN_TOKEN = (r"II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX")


_TEENS = {"one": "eleven", "two": "twelve", "three": "thirteen", "four": "fourteen",
          "five": "fifteen", "six": "sixteen", "seven": "seventeen",
          "eight": "eighteen", "nine": "nineteen"}
# LLM sometimes verbalizes years arithmetically; the spoken convention pairs
# the digits: 1066 -> 'ten sixty-six', 1431 -> 'fourteen thirty-one'.
_YEAR_HUNDRED = re.compile(
    r"\bone thousand,?(?: and)? (one|two|three|four|five|six|seven|eight|nine) hundred"
    r"(?:(?:,? and)?\s+((?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?"
    r"|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
    r"|one|two|three|four|five|six|seven|eight|nine))?\b", re.IGNORECASE)
_YEAR_BARE = re.compile(
    r"\bone thousand,?(?: and)? ((?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"[- ](?:one|two|three|four|five|six|seven|eight|nine))\b", re.IGNORECASE)


_UNITS_TEENS = (r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)")
# Era-marked years are unambiguous even in the 1001-1019 range where bare
# forms could be quantities ('1,001 Arabian Nights').
_YEAR_ERA = re.compile(
    rf"\b(AD|A\.D\.|BC|B\.C\.|CE|C\.E\.)\s+one thousand(?:,? and)? ({_UNITS_TEENS})\b")


def fix_year_style(text: str) -> str:
    """'one thousand four hundred thirty-one' -> 'fourteen thirty-one';
    'one thousand sixty-six' -> 'ten sixty-six'; 'AD one thousand and one'
    -> 'AD ten oh-one'. Deliberately conservative: bare 'one thousand and
    one' stays untouched (quantity reading is live: 1,001 nights); only
    tens-compounds, century forms, and era-marked forms convert."""
    def era(m: re.Match) -> str:
        tail = m.group(2).lower()
        units = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
        spoken = f"ten oh-{tail}" if tail in units else f"ten {tail}"
        return f"{m.group(1)} {spoken}"

    text = _YEAR_ERA.sub(era, text)

    def hundred(m: re.Match) -> str:
        century = _TEENS[m.group(1).lower()]
        tail = (m.group(2) or "hundred").lower()
        return f"{century} {tail}" if tail != "hundred" else f"{century} hundred"

    text = _YEAR_HUNDRED.sub(hundred, text)
    return _YEAR_BARE.sub(lambda m: f"ten {m.group(1).lower()}", text)


def verbalize_romans(text: str) -> str:
    """Roman numerals contain no digits, so digit checks never see them, and
    TTS reads bare 'III' as letters. Regnal numbers become ordinals
    ('Mehmed III' -> 'Mehmed the Third'); structural ones become cardinals
    ('Part II' -> 'Part Two')."""

    def structural(m: re.Match) -> str:
        n = _roman_to_int(m.group(2))
        return f"{m.group(1)} {_int_words(n).capitalize()}" if n and n <= 99 else m.group(0)

    text = re.sub(rf"\b(Chapter|Part|Volume|Book|Act|Section|War|Type|Class|Grade|"
                  rf"Phase|Mark|Title|Article|Appendix|Annex)\s+({_ROMAN_TOKEN})\b",
                  structural, text)

    def regnal(m: re.Match) -> str:
        n = _roman_to_int(m.group(2))
        if not n or n >= len(_ORDINALS):
            return m.group(0)
        return f"{m.group(1)} the {_ORDINALS[n]}"

    # A capitalized name followed by a roman numeral token (not 'I': pronoun).
    return re.sub(rf"\b([A-Z][a-z]{{2,}})\s+({_ROMAN_TOKEN})\b(?!\.)", regnal, text)


def spoken_heading(title: str) -> str:
    """'CHAPTER III: THE PROBLEM' -> 'Chapter Three. The Problem.'"""
    t = re.sub(r"\s+", " ", title).strip()

    def repl(m: re.Match) -> str:
        word, num = m.group(1), m.group(2)
        if num.isdigit():
            spoken = _int_words(int(num))
        else:
            r = _roman_to_int(num)
            spoken = _int_words(r) if r else num.lower()
        return f"{word.capitalize()} {spoken.capitalize()}"

    t = re.sub(r"\b(chapter|part|book|section|letter|volume)\s+([0-9]+|[ivxlcIVXLC]+)\b", repl, t, flags=re.IGNORECASE)
    t = re.sub(r"\s*[:–]\s*|\s+-\s*|\s*-\s+", ". ", t)
    t = " ".join(w.capitalize() if w.isupper() and len(w) >= 2 else w for w in t.split())
    t = t.rstrip(" ,;:.")
    if not t.endswith("."):
        t += "."
    return t


WINDOW_MAX_CHARS = int(CFG["narration"].get("window_max_chars", 3500))
MIN_WINDOW_OUTPUT_RATIO = float(CFG["narration"].get("min_window_output_ratio", 0.85))


def _windows(text: str, size: int = WINDOW_PARAGRAPHS,
             max_chars: int = WINDOW_MAX_CHARS) -> list[tuple[int, str]]:
    """(char_offset, window_text) tuples: up to `size` paragraphs AND at most
    `max_chars` characters per window. A paragraph longer than max_chars is
    split at sentence boundaries. Windows were paragraph-count only until
    2026-09-05: a memoir whose PDF paragraphs are indent-only came through
    stage 2 as 132 paragraphs of 2-12k chars, windows hit 25k chars, the
    model's output cap truncated the tail of every big window, and 27% of
    the book silently vanished from the narration."""
    paras = text.split("\n\n")
    # Flatten to (offset, piece) with oversize paragraphs pre-split.
    pieces: list[tuple[int, str]] = []
    offset = 0
    for p in paras:
        if len(p) <= max_chars:
            pieces.append((offset, p))
        else:
            sents = re.split(r"(?<=[.!?])\s+", p)
            cur, cur_off, pos = "", offset, offset
            for s in sents:
                if cur and len(cur) + len(s) + 1 > max_chars:
                    pieces.append((cur_off, cur))
                    cur, cur_off = "", pos
                cur = f"{cur} {s}" if cur else s
                pos += len(s) + 1
            if cur:
                pieces.append((cur_off, cur))
        offset += len(p) + 2
    out: list[tuple[int, str]] = []
    group: list[tuple[int, str]] = []
    glen = 0
    for off, piece in pieces:
        if group and (len(group) >= size or glen + len(piece) + 2 > max_chars):
            out.append((group[0][0], "\n\n".join(x for _, x in group)))
            group, glen = [], 0
        group.append((off, piece))
        glen += len(piece) + 2
    if group:
        out.append((group[0][0], "\n\n".join(x for _, x in group)))
    return [(o, c) for o, c in out if c.strip()]


def _local_violations(window_src: str, segments: list[dict]) -> list[str]:
    """Cheap immediate QC used for the bounded in-stage retry loop."""
    problems = []
    joined = " ".join(s["text"] for s in segments)
    digits = re.findall(r"\d+", joined)
    if digits:
        problems.append(f"digit strings remain in output: {digits[:8]}")
    # Faithful rewrites run 0.97-0.99 of source length (Carnegie p10 0.97);
    # the old 0.5 bar let a model that truncated the last fifth of every
    # window pass in-stage QC.
    if len(joined) < MIN_WINDOW_OUTPUT_RATIO * len(window_src):
        problems.append(f"output suspiciously short ({len(joined)} chars vs source {len(window_src)}, "
                        f"ratio {len(joined) / max(1, len(window_src)):.2f}): content was dropped")
    too_long = [s["text"][:60] for s in segments if len(s["text"]) > 900]
    if too_long:
        problems.append(f"segments over 900 chars: {too_long}")
    return problems


def _passthrough_window(window_src: str, model: str | None = None) -> list[dict]:
    """Last-resort fallback when the LLM cannot produce usable output for a
    window: deterministically cleaned source text. Never crashes the book;
    the stage-4 evals flag whatever the LLM would have fixed, and the QC
    retry loop re-narrates that chapter. Numbers still need spoken forms
    (no_surviving_digits gate): the LLM verbalizes them in context when a
    model is available, the mechanical verbalizer otherwise."""
    paras = [p.strip() for p in window_src.split("\n\n") if p.strip()]
    out = []
    for p in paras:
        p = verbalize_fractions(p)
        p = llm_verbalize_numbers(p, model) if model else verbalize_digits(p)
        out.append({"type": "paragraph", "text": re.sub(r"\s+", " ", p).strip()})
    return out


def _rewrite_window(window_src: str, model: str, note: str | None = None) -> list[dict]:
    from pipeline.ollama_client import OllamaError

    prompt = SYSTEM_PROMPT if not note else SYSTEM_PROMPT + (
        "\n\nIMPORTANT: a previous rewrite of this passage omitted or mangled the "
        "following. It MUST appear (rewritten for the ear) this time, in its "
        "original position:\n- " + note)
    messages = [{"role": "system", "content": prompt},
                {"role": "user", "content": window_src}]
    for attempt in range(WINDOW_RETRIES + 1):
        try:
            result = chat_json(model, messages, REWRITE_SCHEMA, temperature=0.2)
        except OllamaError as e:
            print(f"    WINDOW FALLBACK (passthrough): {e}", flush=True)
            return _passthrough_window(window_src, model)
        segments = [s for s in result.get("segments", []) if s.get("text", "").strip()]
        problems = _local_violations(window_src, segments)
        if not problems or attempt == WINDOW_RETRIES:
            return segments
        messages = messages + [
            {"role": "assistant", "content": json.dumps(result)[:4000]},
            {"role": "user", "content": "Fix these problems and return the corrected JSON. "
                                        "Do not change anything else:\n- " + "\n- ".join(problems)},
        ]
    return segments


def _load_chapters(book: BookCtx) -> list[dict]:
    data = json.loads((book.artifacts_dir / ARTIFACT_FILES["chapters"]).read_text(encoding="utf-8"))
    chapters = [c for c in data["chapters"] if not c.get("front_matter")]
    only = book.config.get("only_chapters")
    if only:  # pass^k cost bounding: regenerate a deterministic subset
        chapters = [c for c in chapters if c["id"] in only]
    return chapters


def _narrate_chapter(ch: dict, model: str, progress: dict | None = None,
                     progress_path: Path | None = None,
                     window_notes: dict[str, str] | None = None) -> list[dict]:
    """Returns raw segment dicts (id-less); ids assigned at assembly.

    Each completed window is checkpointed immediately: a crash nine hours
    into a book resumes instead of restarting."""
    import hashlib

    from pipeline.textquality import dict_word_ratio

    title = ch["title"]
    if dict_word_ratio(title) < 0.5:  # OCR-garbage title: announce neutrally
        ordinal = int(re.sub(r"\D", "", ch["id"]) or 0) + 1
        title = f"Chapter {ordinal}"
    segments: list[dict] = [{
        "chapter_id": ch["id"], "type": "chapter_heading",
        "text": spoken_heading(title),
        "source_span": None,
    }]
    for offset, window in _windows(ch["text"]):
        # Text-only key: survives chapter-boundary restructuring. Legacy
        # chapter|offset keys are honored on lookup.
        key = hashlib.sha256(window.encode()).hexdigest()[:16]
        legacy = hashlib.sha256(f"{ch['id']}|{offset}|{window}".encode()).hexdigest()[:16]
        if progress is not None and (key in progress or legacy in progress):
            raw_segs = progress.get(key) or progress[legacy]
        else:
            raw_segs = _rewrite_window(window, model,
                                       note=(window_notes or {}).get(key))
            if progress_path is not None:
                with open(progress_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "segments": raw_segs}) + "\n")
                if progress is not None:
                    progress[key] = raw_segs
        for seg in raw_segs:
            segments.append({
                "chapter_id": ch["id"], "type": seg["type"], "text": seg["text"].strip(),
                "source_span": (offset, offset + len(window)),
                "window_key": key,  # enables window-precise retry
            })
    return segments


def split_long_segments(segments: list[dict], max_chars: int = MAX_SEGMENT_CHARS) -> list[dict]:
    """Deterministic post-split: the model is unreliable about honoring the
    length limit, so long paragraphs split at sentence boundaries here."""
    out = []
    for seg in segments:
        if len(seg["text"]) <= max_chars or seg["type"] not in ("paragraph", "blockquote"):
            out.append(seg)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", seg["text"])
        # A single sentence over the limit splits at its last clause boundary.
        expanded: list[str] = []
        for s in sentences:
            while len(s) > max_chars:
                cut = max(s.rfind("; ", 0, max_chars), s.rfind(", ", 0, max_chars))
                if cut < max_chars // 3:
                    cut = s.rfind(" ", 0, max_chars)
                if cut <= 0:
                    break
                expanded.append(s[:cut + 1].strip())
                s = s[cut + 1:].strip()
            expanded.append(s)
        sentences = expanded
        chunk = ""
        for s in sentences:
            if chunk and len(chunk) + len(s) + 1 > max_chars:
                out.append({**seg, "text": chunk})
                chunk = s
            else:
                chunk = f"{chunk} {s}".strip() if chunk else s
        if chunk:
            out.append({**seg, "text": chunk})
    return out


def _map_label_like(text: str) -> bool:
    """Short standalone fragments dominated by out-of-dictionary tokens are
    map labels or figure debris that leaked past extraction, never prose."""
    from pipeline.textquality import _dictionary

    t = text.strip()
    if len(t) >= 60:
        return False
    tokens = re.findall(r"[A-Za-z]{2,}", t)
    if not tokens:
        return True
    d = _dictionary()
    oov = sum(1 for tok in tokens if tok.lower() not in d)
    return oov / len(tokens) >= 0.4


def _content_free(text: str) -> bool:
    """Segments that are mostly digits/punctuation (leaked page lists,
    reference runs) must never be narrated."""
    stripped = re.sub(r"\s", "", text)
    if len(stripped) < 20:
        return False
    alpha = sum(c.isalpha() for c in stripped)
    return alpha / len(stripped) < 0.5


# Honorifics whose trailing period is not a sentence boundary: "Mr. Whitehead"
# must not explode into a 0.3s "Mr." take (degenerate for TTS and for the
# speaker-similarity embedding). re lookbehinds are fixed-width, so one per
# abbreviation.
_ABBREV_GUARD = "".join(
    rf"(?<!\b{a}\.)" for a in
    ("Mr", "Mrs", "Ms", "Dr", "St", "Lieut", "Col", "Capt", "Sergt", "Prof",
     "Hon", "Rev", "Gen", "Maj", "No"))
_SENT_SPLIT = re.compile(rf"(?<=[.!?]){_ABBREV_GUARD}\s+")
# A bare dialogue attribution split off its quote ("I asked.", "exclaimed.")
# is not a speakable unit; rejoin it to the preceding sentence.
_ATTRIBUTION = re.compile(
    r"(?:(?:he|she|I|we|they)\s+)?"
    r"(?:asked|exclaimed|cried|said|replied|shouted|answered|muttered|whispered)[.!?]?$",
    re.IGNORECASE)


def _explode_sentences(segments: list[dict]) -> list[dict]:
    """Schema v2: the atomic (and cached) unit is the sentence. Paragraph
    membership survives via para_id so assembly knows sentence gaps from
    paragraph gaps."""
    out: list[dict] = []
    para_n = 0
    for seg in segments:
        para_n += 1
        para_id = f"para_{para_n:04d}"
        if seg["type"] in ("chapter_heading", "section_heading"):
            out.append({**seg, "para_id": para_id, "sentence_index": 0})
            continue
        pieces = [s for s in _SENT_SPLIT.split(seg["text"].strip()) if s.strip()]
        sentences: list[str] = []
        for s in pieces:
            bare = s.strip()
            degenerate = re.fullmatch(r"[\W_]+", bare) or _ATTRIBUTION.fullmatch(bare)
            if sentences and degenerate:
                sentences[-1] = f"{sentences[-1]} {bare}"
            elif re.fullmatch(r"[\W_]+", bare):
                continue  # leading punctuation-only shard: nothing to speak
            else:
                sentences.append(bare)
        for s in sentences or [seg["text"]]:
            out.append({**seg, "text": s, "para_id": para_id})
    return out


# Sentence-boundary adjudication. The regex splitter cannot tell "Owen D. |
# Young" (a middle initial) from "plan B. | Then" (a sentence end), and the
# costs are asymmetric: an over-joined take runs long and the length splitter
# trims it at a clause; an under-joined one is a 0.5s "Owen D." take that
# fails take review on every re-roll (Carnegie 2026-09-05: six such names).
# So deterministic code flags SUSPICIOUS boundaries (a small subset), the LLM
# decides join-or-split on just those, the answer is schema-validated and
# cached, and the fallback biases toward joining on the initial pattern.
_INITIAL_END = re.compile(r"\b[A-Z]\.$")
_ABBR_END = re.compile(r"\b(?:Jr|Sr|Inc|Ltd|Co|Bros|vs|etc|Mt|Ft)\.$")
SPLIT_CACHE = ROOT / "evals" / ".cache" / "split_adjudication"
SPLIT_PROMPT_VERSION = "v1"
SPLIT_SCHEMA = {
    "type": "object",
    "properties": {"decisions": {"type": "array", "items": {
        "type": "object",
        "properties": {"i": {"type": "integer"}, "join": {"type": "boolean"}},
        "required": ["i", "join"]}}},
    "required": ["decisions"],
}
SPLIT_PROMPT = """A narration script was split into sentences for text-to-speech, one take per \
sentence. Some splits are wrong: a period after a middle initial ("Owen D." + "Young, a lawyer"), \
an abbreviation ("Rockefeller, Jr." + "was..."), a list label, or a fragment that cannot be spoken \
as its own take. For each numbered pair decide whether LEFT and RIGHT are ONE sentence that was \
wrongly split (join: true) or genuinely two sentences (join: false). A short but complete sentence \
stays split. Do not rewrite anything.

{pairs}

Return JSON: {{"decisions": [{{"i": <pair number>, "join": true|false}}, ...]}}, one entry per pair."""


def _suspicious_boundary(left: str, right: str) -> bool:
    a, b = left.strip(), right.strip()
    return bool(_INITIAL_END.search(a) or _ABBR_END.search(a) or len(a) < 25
                or (b and b[0].islower()))


def _fallback_join(left: str, right: str) -> bool:
    a = left.strip()
    return bool(_INITIAL_END.search(a) or _ABBR_END.search(a))


def adjudicate_splits(segments: list[dict], model: str, batch: int = 20) -> list[dict]:
    """Join wrongly split sentences. Candidates are consecutive segments from
    the same paragraph whose boundary looks suspicious; the LLM rules on each
    candidate pair, decisions are cached by pair text, and a mechanical
    fallback (join on initial/abbreviation) covers an unreachable or
    malformed judge."""
    import hashlib

    # Candidates: same chapter, both prose. Within a paragraph any suspicious
    # boundary qualifies. ACROSS paragraphs only the strong signals do (an
    # initial or abbreviation on the left, a lowercase start on the right):
    # the LLM window rewrite splits "Owen D." from "Young" at a PDF line
    # break and emits them as separate paragraphs, so a same-paragraph rule
    # never sees the worst cases. A join keeps the left segment's paragraph.
    def _cand(i: int) -> bool:
        a, b = segments[i], segments[i + 1]
        if a.get("chapter_id") != b.get("chapter_id"):
            return False
        if a["type"] not in ("paragraph", "blockquote") or b["type"] not in ("paragraph", "blockquote"):
            return False
        at, bt = a["text"].strip(), b["text"].strip()
        strong = bool(_INITIAL_END.search(at) or _ABBR_END.search(at) or (bt and bt[0].islower()))
        if a.get("para_id") and a.get("para_id") == b.get("para_id"):
            return strong or _suspicious_boundary(at, bt)
        return strong

    cands = [i for i in range(len(segments) - 1) if _cand(i)]
    if not cands:
        return segments
    decisions: dict[int, bool] = {}
    uncached: list[tuple[int, Path]] = []
    for i in cands:
        key = hashlib.sha256(f"{SPLIT_PROMPT_VERSION}|{model}|{segments[i]['text']}|"
                             f"{segments[i + 1]['text']}".encode()).hexdigest()[:24]
        c = SPLIT_CACHE / f"{key}.json"
        if c.exists():
            decisions[i] = bool(json.loads(c.read_text())["join"])
        else:
            uncached.append((i, c))
    judged = fallback = 0
    for start in range(0, len(uncached), batch):
        chunk = uncached[start:start + batch]
        pairs = "\n".join(f"{n}. LEFT: {segments[i]['text']!r}\n   RIGHT: {segments[i + 1]['text']!r}"
                          for n, (i, _) in enumerate(chunk, 1))
        try:
            r = chat_json(model, [{"role": "user", "content": SPLIT_PROMPT.format(pairs=pairs)}],
                          SPLIT_SCHEMA, temperature=0.0, num_ctx=4096)
            got = {int(d["i"]): bool(d["join"]) for d in r.get("decisions", [])
                   if isinstance(d, dict) and "i" in d and "join" in d}
        except Exception:
            got = {}
        for n, (i, c) in enumerate(chunk, 1):
            if n in got:
                decisions[i] = got[n]
                SPLIT_CACHE.mkdir(parents=True, exist_ok=True)
                c.write_text(json.dumps({"join": got[n]}))
                judged += 1
            else:
                decisions[i] = _fallback_join(segments[i]["text"], segments[i + 1]["text"])
                fallback += 1
    out: list[dict] = []
    i = 0
    while i < len(segments):
        seg = dict(segments[i])
        j = i
        while decisions.get(j):
            seg["text"] = f"{seg['text']} {segments[j + 1]['text']}".strip()
            j += 1
        out.append(seg)
        i = j + 1
    joined = sum(1 for v in decisions.values() if v)
    print(f"  split adjudication: {len(cands)} suspicious boundaries, {joined} joined "
          f"({len(cands) - judged - fallback} cached, {judged} judged, {fallback} fallback)", flush=True)
    return out


def repair_splits(book: BookCtx) -> int:
    """Apply split adjudication to an existing narration script (for a book
    narrated before the adjudicator existed). Re-ids segments; run stages
    5-6 afterwards: only the joined segments re-render. Returns joins."""
    import shutil
    import time

    model = book.config.get("rewrite_model", REWRITE_MODEL)
    path = book.artifacts_dir / ARTIFACT_FILES["narration"]
    data = json.loads(path.read_text(encoding="utf-8"))
    before = len(data["segments"])
    segs = adjudicate_splits(data["segments"], model)
    segs = split_long_segments(segs)
    if len(segs) == before:
        print("  no joins; narration unchanged", flush=True)
        return 0
    shutil.copy2(path, path.with_name(f"04_narration.{int(time.time())}.presplitfix.bak.json"))
    para_counts: dict[str, int] = {}
    for seg in segs:
        pid = seg.get("para_id") or ""
        seg["sentence_index"] = para_counts.get(pid, 0)
        para_counts[pid] = seg["sentence_index"] + 1
    for i, seg in enumerate(segs):
        seg["id"] = f"seg_{i:04d}"
    data["segments"] = segs
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"  narration rewritten: {before} -> {len(segs)} segments", flush=True)
    return before - len(segs)


def _write_script(book: BookCtx, all_segments: list[dict], model: str,
                  input_hash: str | None = None) -> Path:
    all_segments = [s for s in all_segments
                    if s["type"] == "chapter_heading"
                    or not (_content_free(s["text"]) or _map_label_like(s["text"]))]
    all_segments = dedupe_opening_titles(all_segments, model)
    for s in all_segments:
        s["text"] = fix_year_style(
            verbalize_romans(verbalize_fractions(s["text"].replace("\\", " ")))).strip()
    all_segments = _explode_sentences(all_segments)
    all_segments = adjudicate_splits(all_segments, model)  # rejoin split initials etc.
    all_segments = split_long_segments(all_segments)  # clause-splits rare huge sentences
    para_counts: dict[str, int] = {}
    for seg in all_segments:
        pid = seg.get("para_id") or ""
        seg["sentence_index"] = para_counts.get(pid, 0)
        para_counts[pid] = seg["sentence_index"] + 1
    for i, seg in enumerate(all_segments):
        seg["id"] = f"seg_{i:04d}"
    allowed = set(Segment.model_fields)
    script = NarrationScript(book_id=book.book_id, model=model,
                             segments=[Segment(**{k: v for k, v in s.items() if k in allowed})
                                       for s in all_segments])
    payload = json.loads(script.model_dump_json())
    if input_hash:
        payload["input_hash"] = input_hash
    out = book.artifacts_dir / ARTIFACT_FILES["narration"]
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return out


def _input_hash(book: BookCtx, model: str) -> str:
    import hashlib
    chapters_bytes = (book.artifacts_dir / ARTIFACT_FILES["chapters"]).read_bytes()
    key = f"{model}|{sorted(book.config.get('only_chapters') or [])}".encode()
    return hashlib.sha256(chapters_bytes + key).hexdigest()[:16]


def run(book: BookCtx) -> Path:
    model = book.config.get("rewrite_model", REWRITE_MODEL)
    out_path = book.artifacts_dir / ARTIFACT_FILES["narration"]
    # A completed narration for unchanged inputs is authoritative: re-running
    # would produce slightly different text AND invalidate the render cache.
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            stamp = existing.get("input_hash")
            if stamp is None or stamp == _input_hash(book, model):
                print("  narration up to date; skipping (delete artifact to force)", flush=True)
                return out_path
        except (json.JSONDecodeError, OSError):
            pass  # torn artifact: regenerate
    progress_path = book.artifacts_dir / "04_narration.progress.jsonl"
    progress: dict = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                progress[row["key"]] = row["segments"]
            except (json.JSONDecodeError, KeyError):
                continue  # torn line from a crash mid-write
        if progress:
            print(f"  resuming: {len(progress)} windows already narrated", flush=True)

    segments: list[dict] = []
    for ch in _load_chapters(book):
        print(f"  narrating {ch['id']}: {ch['title']!r} ({len(ch['text']):,} chars)", flush=True)
        segments.extend(_narrate_chapter(ch, model, progress, progress_path))
    out = _write_script(book, segments, model, input_hash=_input_hash(book, model))
    progress_path.unlink(missing_ok=True)
    return out


def retry(book: BookCtx, feedback: list[Violation]) -> Path:
    """Re-narrate only what the violations name. Segment-level unit ids map
    to their source WINDOWS (via window_key, or source_span for legacy
    narrations), so fixing three bad windows never re-narrates a 160K-char
    chapter or invalidates its render cache. Chapter-level unit ids fall back
    to whole-chapter re-narration with feedback in the prompt."""
    import hashlib

    model = book.config.get("rewrite_model", REWRITE_MODEL)
    narration_path = book.artifacts_dir / ARTIFACT_FILES["narration"]
    existing = json.loads(narration_path.read_text(encoding="utf-8"))
    # A rewritten script orphans its recorded takes: never overwrite the only
    # copy. Timestamped backups keep every version assemblable.
    import time
    backup = narration_path.with_suffix(f".{int(time.time())}.bak.json")
    backup.write_text(json.dumps(existing, indent=1), encoding="utf-8")
    chapters = _load_chapters(book)
    ch_text = {c["id"]: c["text"] for c in chapters}

    def window_key_of(seg: dict) -> str | None:
        if seg.get("window_key"):
            return seg["window_key"]
        span = seg.get("source_span")
        text = ch_text.get(seg["chapter_id"])
        if span and text:
            return hashlib.sha256(text[span[0]:span[1]].encode()).hexdigest()[:16]
        return None

    seg_by_id = {s["id"]: s for s in existing["segments"]}
    bad_windows: set[str] = set()
    bad_chapters: set[str] = set()
    notes: dict[str, list[str]] = {}
    window_notes: dict[str, str] = {}
    for v in feedback:
        if not v.unit_id:
            continue
        if v.unit_id.startswith("seg_") and v.unit_id in seg_by_id:
            key = window_key_of(seg_by_id[v.unit_id])
            if key:
                bad_windows.add(key)
                continue
        # Chapter-level violation (e.g. a dropped source sentence): pin it to
        # the window that contains the offending passage, so one bad sentence
        # re-narrates one window, not the chapter. Injecting the MUST-appear
        # note chapter-wide makes every window insert the passage (duplicated
        # narration); the note goes only to the window it belongs in.
        def _fold(s: str) -> str:
            # The eval's sentence splitter straightens curly quotes; fold both
            # sides identically or probes never match the raw chapter text.
            s = s.translate(str.maketrans({"’": "'", "‘": "'",
                                           "“": '"', "”": '"'}))
            return re.sub(r"\s+", " ", s)

        probe = _fold(v.context or "").strip()[:120]
        text = ch_text.get(v.unit_id)
        pinned = False
        if text and len(probe) > 20:
            # A passage that straddles a window boundary (heading glued to the
            # following paragraph) matches no single window in full; its head
            # and tail pieces pin each window it touches.
            pieces = {probe}
            if len(probe) > 60:
                pieces.update((probe[:60], probe[-60:]))
            for _offset, window in _windows(text):
                w = _fold(window)
                if any(p in w for p in pieces):
                    key = hashlib.sha256(window.encode()).hexdigest()[:16]
                    bad_windows.add(key)
                    window_notes[key] = probe if key not in window_notes \
                        else window_notes[key] + "\n- " + probe
                    pinned = True
        if not pinned:
            bad_chapters.add(v.unit_id)
            notes.setdefault(v.unit_id, []).append(v.context or v.message)

    # Synthetic checkpoint table from the existing narration, minus everything
    # marked bad: _narrate_chapter regenerates exactly the gaps.
    progress: dict[str, list[dict]] = {}
    for s in existing["segments"]:
        if s["type"] == "chapter_heading" or s["chapter_id"] in bad_chapters:
            continue
        key = window_key_of(s)
        if key and key not in bad_windows:
            progress.setdefault(key, []).append({"type": s["type"], "text": s["text"]})

    # Re-rolling a window is a faithfulness lottery: each attempt can drop a
    # different sentence, so the QC loop never converges on stubborn windows.
    # From the second retry on, clamp them to deterministic passthrough
    # (cleaned source text, no LLM): guaranteed faithful, guaranteed to end.
    attempt = int(book.config.get("_narration_attempt", 0)) + 1
    book.config["_narration_attempt"] = attempt
    clamped = 0
    if attempt >= 2 and bad_windows:
        for ch in chapters:
            for _offset, window in _windows(ch["text"]):
                key = hashlib.sha256(window.encode()).hexdigest()[:16]
                if key in bad_windows and key not in progress:
                    progress[key] = _passthrough_window(window, model)
                    clamped += 1

    print(f"  window-precise retry: {len(bad_windows)} windows "
          f"({clamped} clamped to passthrough), "
          f"{len(bad_chapters)} whole chapters", flush=True)
    segments: list[dict] = []
    global SYSTEM_PROMPT
    old_prompt = SYSTEM_PROMPT
    for ch in chapters:
        if ch["id"] in bad_chapters and notes.get(ch["id"]):
            SYSTEM_PROMPT = old_prompt + (
                "\n\nIMPORTANT: your previous rewrite of this chapter omitted or mangled "
                "the following passages. They MUST appear (rewritten for the ear) this time:\n- "
                + "\n- ".join(notes[ch["id"]][:12]))
        try:
            segments.extend(_narrate_chapter(ch, model, progress, progress_path=None,
                                             window_notes=window_notes))
        finally:
            SYSTEM_PROMPT = old_prompt
    return _write_script(book, segments, model, input_hash=_input_hash(book, model))
