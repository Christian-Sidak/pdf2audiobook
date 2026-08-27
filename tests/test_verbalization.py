"""Golden table tests for the deterministic narration verbalizers.

Every listening bug becomes a row here. Run: python3 -m pytest tests/ -q
(or directly: python3 tests/test_verbalization.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.s4_narration import spoken_heading, verbalize_fractions, verbalize_romans

REGNAL_CASES = [
    # The 'ayayay' bug: sultans and monarchs read as letters.
    ("Sultan Mehmed III ascended the throne.", "Sultan Mehmed the Third ascended the throne."),
    ("Selim III focused on the new army.", "Selim the Third focused on the new army."),
    ("Mahmud II abolished the Janissaries.", "Mahmud the Second abolished the Janissaries."),
    ("Catherine II of Russia and Joseph II of Austria met.",
     "Catherine the Second of Russia and Joseph the Second of Austria met."),
    ("Louis XIV built Versailles.", "Louis the Fourteenth built Versailles."),
    ("Pope Benedict XVI resigned.", "Pope Benedict the Sixteenth resigned."),
    # Structural romans are cardinals, not ordinals.
    ("See Part II of this study.", "See Part Two of this study."),
    ("Chapter IV covers the reforms.", "Chapter Four covers the reforms."),
    ("World War II changed everything.", "World War Two changed everything."),
    # Never touched: pronoun 'I', middle initials, lone letters.
    ("I went to see him, and I stayed.", "I went to see him, and I stayed."),
    ("John V. Smith wrote the report.", "John V. Smith wrote the report."),
]

FRACTION_CASES = [
    # Numerals that are not digits: the check-evading class.
    ("The train leaves from Platform 9¾ at eleven.",
     "The train leaves from Platform nine and three quarters at eleven."),
    ("Add 2½ cups of flour.", "Add two and one half cups of flour."),
    ("About ¾ of the population agreed.", "About three quarters of the population agreed."),
    ("A ⅓ share went to each heir.", "A one third share went to each heir."),
    ("The plank was 9 3/4 inches wide.", "The plank was nine and three quarters inches wide."),
    ("He worked 12 1/2 hours.", "He worked twelve and a half hours."),
    # Not fractions: dates, ratios out of range stay untouched.
    ("The vote was 9/12 in favor.", "The vote was 9/12 in favor."),
]

HEADING_CASES = [
    ("CHAPTER III: THE PROBLEM", "Chapter Three. The Problem."),
    ("Chapter One: The Problem", "Chapter One. The Problem."),
    ("PART 2 - Initial Ottoman Responses", "Part Two. Initial Ottoman Responses."),
    ("Introduction", "Introduction."),
    ("Note on Transliteration,", "Note on Transliteration."),
]


def test_regnal_and_structural_romans():
    for src, want in REGNAL_CASES:
        got = verbalize_romans(src)
        assert got == want, f"{src!r}: got {got!r}, want {want!r}"


def test_fractions():
    for src, want in FRACTION_CASES:
        got = verbalize_fractions(src)
        assert got == want, f"{src!r}: got {got!r}, want {want!r}"


def test_spoken_headings():
    for src, want in HEADING_CASES:
        got = spoken_heading(src)
        assert got == want, f"{src!r}: got {got!r}, want {want!r}"



YEAR_STYLE_CASES = [
    # Medieval years pair the digits; arithmetic reading is a style error.
    ("The conquest of one thousand sixty-six changed England.",
     "The conquest of ten sixty-six changed England."),
    ("She was tried in one thousand four hundred thirty-one at Rouen.",
     "She was tried in fourteen thirty-one at Rouen."),
    ("Around one thousand two hundred the schools flourished.",
     "Around twelve hundred the schools flourished."),
    ("one thousand nine hundred and seventeen was a hard year.",
     "nineteen seventeen was a hard year."),
    # Era markers disambiguate the 1001-1019 range absolutely.
    ("The charter dates to AD one thousand and one.",
     "The charter dates to AD ten oh-one."),
    ("He died in AD one thousand and sixteen.",
     "He died in AD ten sixteen."),
    # Quantities must NEVER convert: the ambiguous bare form stays put.
    ("She told the one thousand and one Arabian nights.",
     "She told the one thousand and one Arabian nights."),
    ("An army of one thousand and one men marched.",
     "An army of one thousand and one men marched."),
]


def test_year_style():
    from pipeline.s4_narration import fix_year_style
    for src_, want in YEAR_STYLE_CASES:
        got = fix_year_style(src_)
        assert got == want, f"{src_!r}: got {got!r}, want {want!r}"

if __name__ == "__main__":
    test_regnal_and_structural_romans()
    test_fractions()
    test_spoken_headings()
    test_year_style()
    print(f"all {len(REGNAL_CASES) + len(FRACTION_CASES) + len(HEADING_CASES) + len(YEAR_STYLE_CASES)} verbalization cases pass")
