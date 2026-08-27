"""Lazy, memoized access to per-book pipeline artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from evals.contracts import ARTIFACT_FILES, ArtifactMissing


class ArtifactSet:
    def __init__(self, book_dir: Path):
        self.book_dir = Path(book_dir)
        self._cache: dict[str, object] = {}

    def path(self, name: str) -> Path:
        return self.book_dir / ARTIFACT_FILES[name]

    def has(self, name: str) -> bool:
        return self.path(name).exists()

    def _load_json(self, name: str) -> dict:
        if name not in self._cache:
            p = self.path(name)
            if not p.exists():
                raise ArtifactMissing(name, p)
            with open(p, encoding="utf-8") as f:
                self._cache[name] = json.load(f)
        return self._cache[name]  # type: ignore[return-value]

    @property
    def extract(self) -> dict:
        return self._load_json("extract")

    @property
    def structural(self) -> dict:
        return self._load_json("structural")

    @property
    def body_text(self) -> str:
        if "body" not in self._cache:
            p = self.path("body")
            if not p.exists():
                raise ArtifactMissing("body", p)
            self._cache["body"] = p.read_text(encoding="utf-8")
        return self._cache["body"]  # type: ignore[return-value]

    @property
    def chapters(self) -> dict:
        return self._load_json("chapters")

    @property
    def narration(self) -> dict:
        return self._load_json("narration")

    @property
    def render_manifest(self) -> dict:
        return self._load_json("render_manifest")

    @property
    def assemble(self) -> dict:
        return self._load_json("assemble")

    @property
    def m4b_path(self) -> Path:
        from pipeline.config import OUTPUT_DIR

        p = OUTPUT_DIR / self.book_dir.name / "book.m4b"
        if not p.exists():
            raise ArtifactMissing("m4b", p)
        return p
