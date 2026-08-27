"""Golden table tests for stage-2 footnote handling.

Two layers: block classification (whole footnote blocks removed from the
page) and inline marker stripping (superscript numbers fused to words).
Run: python3 tests/test_structural.py  (or pytest)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.s2_structural import _is_footnote_block, strip_superscript_markers

PAGE_H = 700.0
BODY_SIZE = 10.0


def block(text: str, size: float, y0: float, y1: float | None = None) -> dict:
    return {"text": text, "size": size, "bbox": [50.0, y0, 550.0, y1 or y0 + 50]}


FOOTNOTE_BLOCKS = [
    # Classic bottom-of-page small-font note.
    block("58. On the aeons as treasures in the Bahir, see sections 96, 97.", 8.0, 620),
    # Multi-note block at the bottom, slightly small font.
    block("1. See, for instance, Sefaretname-i Rasih Efendi, IUL, no. 3887.\n"
          "2. Cf. the article in HTR 25 (1932), 129-134.", 9.5, 580),
    # Asterisk note, small font, lower half.
    block("* The dating of this manuscript is disputed.", 8.5, 400),
    # Numbered-note-majority block at body size, bottom half.
    block("12. Berakhoth 55a; cf. Jewish Gnosticism, 78-79.\n"
          "13. There is no compelling linguistic evidence.", 10.0, 500),
]

BODY_BLOCKS = [
    # Ordinary body paragraph.
    block("The question of the origin of the Kabbalah is difficult.", 10.0, 200),
    # Small-font BLOCK QUOTATION mid-page: scholarly quotes use smaller type
    # and must NOT be eaten as footnotes.
    block("As the chronicler wrote of those days, the land was quiet.", 8.5, 250),
    # Body text that happens to sit low on the page.
    block("The reforms continued into the next decade without pause.", 10.0, 600),
    # Numbered CONTENT list at body size, upper half (estate inventory style).
    block("1. Sheikh Abdulkerim Efendi, resident at the medrese.", 10.0, 150),
]

MARKER_CASES = [
    # Superscript debris fused to punctuation or words: strip.
    ('predicated on the "powers of the creator of the universe."62 This conforms',
     'predicated on the "powers of the creator of the universe." This conforms'),
    ("to confine this weakness as well as the desire].67", "to confine this weakness as well as the desire]."),
    ("attested among ancient sects in the Orient19 and beyond", "attested among ancient sects in the Orient and beyond"),
    ("composed between 1125 and 1240.263", "composed between 1125 and 1240."),
    # Real numbers survive.
    ("the sum of 1,204 akce was recorded", "the sum of 1,204 akce was recorded"),
    ("the War of 1812 ended", "the War of 1812 ended"),
    ("where x2 denotes the square", "where x2 denotes the square"),
    ("a ratio of 3.14 appears mid-sentence here", "a ratio of 3.14 appears mid-sentence here"),
]


def test_footnote_blocks_removed():
    for b in FOOTNOTE_BLOCKS:
        assert _is_footnote_block(b, BODY_SIZE, PAGE_H), f"should be footnote: {b['text'][:50]!r}"


def test_body_blocks_kept():
    for b in BODY_BLOCKS:
        assert not _is_footnote_block(b, BODY_SIZE, PAGE_H), f"should be body: {b['text'][:50]!r}"


def test_superscript_markers():
    for src, want in MARKER_CASES:
        got = strip_superscript_markers(src)
        assert got == want, f"{src!r}: got {got!r}, want {want!r}"


if __name__ == "__main__":
    test_footnote_blocks_removed()
    test_body_blocks_kept()
    test_superscript_markers()
    print(f"all {len(FOOTNOTE_BLOCKS) + len(BODY_BLOCKS) + len(MARKER_CASES)} footnote cases pass")
