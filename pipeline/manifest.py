"""Typed manifest for finished audiobooks: the contract a frontend consumes.

output/<book_id>/
    book.m4b            combined audiobook with chapter markers
    chapters/NN_*.m4a   one mastered file per chapter, track-tagged
    manifest.json       this schema
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChapterFile(BaseModel):
    id: str
    title: str
    track: int = Field(ge=1)
    file: str  # relative to the book's output directory
    duration_s: float = Field(ge=0)


class AudiobookManifest(BaseModel):
    schema_version: int = 1
    book_id: str
    title: Optional[str] = None
    author: Optional[str] = None
    source_pdf: str
    engine: str
    voice: str
    sample_rate: int
    bitrate: str
    duration_s: float
    lufs: float
    m4b: str  # relative to the book's output directory
    chapters: list[ChapterFile]
