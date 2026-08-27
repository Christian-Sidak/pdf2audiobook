"""Shared types for the eval harness and QC loop.

This is the entire coupling surface between evals and pipeline stages:
artifact filenames, the stage runner protocol, and the Violation/CheckResult
types that checks emit and the QC loop consumes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from pipeline.config import ARTIFACTS_DIR

ARTIFACT_FILES = {
    "extract": "01_extract.json",
    "structural": "02_structural.json",
    "body": "02_body.txt",
    "chapters": "03_chapters.json",
    "narration": "04_narration.json",
    "render_manifest": "05_render/manifest.json",
    "assemble": "06_assemble.json",
    "m4b": "06_book.m4b",
}

Status = Literal["pass", "fail", "skip", "error"]


@dataclass
class BookCtx:
    book_id: str
    pdf_path: Path
    artifacts_dir: Path
    config: dict = field(default_factory=dict)

    @classmethod
    def for_book(cls, book_id: str, pdf_path: str | Path, config: dict | None = None) -> "BookCtx":
        d = ARTIFACTS_DIR / book_id
        d.mkdir(parents=True, exist_ok=True)
        return cls(book_id=book_id, pdf_path=Path(pdf_path), artifacts_dir=d, config=config or {})


@dataclass
class Violation:
    message: str
    unit_id: str | None = None  # "p41", "ch03", "seg_0182"
    context: str | None = None  # excerpt around the offense
    fixable: bool = False  # True -> QC loop may feed it back to stage.retry()

    def to_dict(self) -> dict:
        return {"message": self.message, "unit_id": self.unit_id,
                "context": self.context, "fixable": self.fixable}


@dataclass
class CheckResult:
    dimension: str
    stage: int
    status: Status
    violations: list[Violation] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @classmethod
    def passed(cls, dimension: str, stage: int, **details) -> "CheckResult":
        return cls(dimension, stage, "pass", details=details)

    @classmethod
    def failed(cls, dimension: str, stage: int, violations: list[Violation], **details) -> "CheckResult":
        return cls(dimension, stage, "fail", violations=violations, details=details)

    @classmethod
    def skipped(cls, dimension: str, stage: int, reason: str) -> "CheckResult":
        return cls(dimension, stage, "skip", details={"reason": reason})

    def to_dict(self) -> dict:
        return {"dimension": self.dimension, "stage": self.stage, "status": self.status,
                "violations": [v.to_dict() for v in self.violations], "details": self.details}


class StageRunner(Protocol):
    """Each pipeline stage module exposes run(); stages 4-5 also expose retry()."""

    def run(self, book: BookCtx) -> Path: ...

    def retry(self, book: BookCtx, feedback: list[Violation]) -> Path: ...


class ArtifactMissing(Exception):
    """Raised by ArtifactSet when a stage artifact does not exist yet.

    The runner converts this into a SKIP (stage not run), distinct from FAIL.
    """

    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path
        super().__init__(f"artifact '{name}' missing: {path}")
