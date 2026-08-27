"""Narration script IR: the typed contract between text and audio.

Every segment renders as its own audio chunk; pauses are literal silence
inserted by the assembler from segment metadata, never requested from the TTS.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

SegmentType = Literal["chapter_heading", "section_heading", "paragraph", "blockquote"]


class Segment(BaseModel):
    """Schema v2: the atomic cached unit is the SENTENCE, and segments are
    DRY (no pause data). Pauses are assembly-time policy, so pause and
    room-tone changes never orphan recorded takes."""

    id: str = Field(pattern=r"^seg_\d{4,}$")
    chapter_id: str
    type: SegmentType
    text: str = Field(min_length=1)
    para_id: Optional[str] = None  # sentences sharing a paragraph share this
    sentence_index: int = 0
    source_span: Optional[tuple[int, int]] = None  # char offsets in chapter text
    window_key: Optional[str] = None  # narration checkpoint key: window-precise retry


class NarrationScript(BaseModel):
    schema_version: int = 2
    book_id: str
    model: str
    segments: list[Segment] = Field(min_length=1)

    def chapters(self) -> dict[str, list[Segment]]:
        out: dict[str, list[Segment]] = {}
        for s in self.segments:
            out.setdefault(s.chapter_id, []).append(s)
        return out


def assembly_view(segments: list[dict], headings_cfg: dict) -> list[dict]:
    """Apply the heading reading policy at assembly time: headings set to
    'pause' are not spoken — their structural weight becomes silence added to
    the following segment. Shared by the assembler and silence_gaps check."""
    section_mode = (headings_cfg or {}).get("section", "announce")
    if section_mode != "pause":
        return [dict(s) for s in segments]
    out: list[dict] = []
    pending_gap = 0.0
    for s in segments:
        if s["type"] == "section_heading":
            pending_gap += 1.0  # replaced by policy lookup at gap time via _extra_gap flag
            continue
        s = dict(s)
        if pending_gap:
            s["_skipped_heading"] = True
            pending_gap = 0.0
        out.append(s)
    return out


def gap_between(prev: Optional[dict], seg: dict, policy: dict) -> float:
    """Assembly-time pause between two segments, derived LIVE from the pause
    policy config. The single source of truth for both the assembler and the
    silence_gaps_vs_policy check.

    Sentences within one paragraph get the sentence gap; everything else gets
    prev.type's `after` plus seg.type's `before`."""
    if prev is None:
        return 0.0
    if (prev.get("para_id") and prev.get("para_id") == seg.get("para_id")):
        return float(policy.get("sentence", {}).get("gap", 0.55))
    after = float(policy.get(prev["type"], {}).get("after", 0.0))
    before = float(policy.get(seg["type"], {}).get("before", 0.0))
    gap = after + before
    if seg.get("_skipped_heading"):
        # An unspoken section heading: its structural weight becomes silence.
        sh = policy.get("section_heading", {})
        gap += float(sh.get("before", 0.0)) + float(sh.get("after", 0.0))
    return gap
