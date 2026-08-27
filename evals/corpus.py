"""corpus.yaml loading: document specs, seeded assertions, defaults."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ruamel.yaml import YAML

from pipeline.config import ROOT

CORPUS_PATH = Path(__file__).resolve().parent / "corpus.yaml"

yaml = YAML()
yaml.preserve_quotes = True


@dataclass
class SeededString:
    id: str
    text: str
    match: str = "substring"  # substring | line | regex
    note: str = ""


@dataclass
class DocSpec:
    id: str
    pdf: Path
    kind: str = "book"  # book | article | scan
    language: str = "en"
    tier: str = "smoke"  # full | smoke
    golden: dict[str, Path] = field(default_factory=dict)
    expect: dict = field(default_factory=dict)
    must_contain: list[SeededString] = field(default_factory=list)
    must_not_contain: list[SeededString] = field(default_factory=list)
    keep_hyphens: list[str] = field(default_factory=list)
    deletion_allowlist: list[str] = field(default_factory=list)
    skip_checks: list[str] = field(default_factory=list)

    def golden_path(self, name: str) -> Path | None:
        return self.golden.get(name)


@dataclass
class Corpus:
    defaults: dict
    documents: list[DocSpec]
    raw: dict  # ruamel round-trip object, used by seed.py to write back

    def doc(self, doc_id: str) -> DocSpec:
        for d in self.documents:
            if d.id == doc_id:
                return d
        raise KeyError(f"unknown corpus document: {doc_id}")


def _seeded(items: list | None) -> list[SeededString]:
    out = []
    for it in items or []:
        out.append(SeededString(id=it.get("id", ""), text=str(it["text"]),
                                match=it.get("match", "substring"), note=it.get("note", "")))
    return out


def load_corpus(path: Path = CORPUS_PATH) -> Corpus:
    with open(path, encoding="utf-8") as f:
        raw = yaml.load(f)

    docs = []
    for d in raw.get("documents", []):
        assertions = d.get("assertions") or {}
        golden = {k: ROOT / v for k, v in (d.get("golden") or {}).items()}
        docs.append(DocSpec(
            id=d["id"],
            pdf=ROOT / d["pdf"],
            kind=d.get("kind", "book"),
            language=d.get("language", "en"),
            tier=d.get("tier", "smoke"),
            golden=golden,
            expect=dict(d.get("expect") or {}),
            must_contain=_seeded(assertions.get("must_contain")),
            must_not_contain=_seeded(assertions.get("must_not_contain")),
            keep_hyphens=list(assertions.get("keep_hyphens") or []),
            deletion_allowlist=list(assertions.get("deletion_allowlist") or []),
            skip_checks=list(d.get("skip_checks") or []),
        ))
    return Corpus(defaults=dict(raw.get("defaults") or {}), documents=docs, raw=raw)


def save_corpus(corpus: Corpus, path: Path = CORPUS_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(corpus.raw, f)
