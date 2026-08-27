"""Check registry. Each check module registers dimensions via @check."""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable

from evals.contracts import CheckResult

CheckFn = Callable[..., CheckResult]  # fn(doc: DocSpec, art: ArtifactSet, cfg: Defaults)


@dataclass
class RegisteredCheck:
    stage: int
    dimension: str
    fn: CheckFn
    deterministic: bool = True
    requires: tuple[str, ...] = field(default_factory=tuple)  # e.g. ("golden.chapters",)


REGISTRY: list[RegisteredCheck] = []


def check(stage: int, dimension: str, deterministic: bool = True, requires: tuple[str, ...] = ()):
    def deco(fn: CheckFn) -> CheckFn:
        REGISTRY.append(RegisteredCheck(stage, dimension, fn, deterministic, tuple(requires)))
        return fn
    return deco


_MODULES = ["s1_extract", "s2_structural", "s3_chapterize", "s4_narration", "s5_render", "s6_assemble"]


def load_all() -> list[RegisteredCheck]:
    for mod in _MODULES:
        try:
            importlib.import_module(f"evals.checks.{mod}")
        except ModuleNotFoundError as e:
            if e.name != f"evals.checks.{mod}":  # real dependency error inside a module
                raise
    return REGISTRY
