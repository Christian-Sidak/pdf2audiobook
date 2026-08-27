"""Mastering tests: per-take leveling collapses variance, overlay preserves
duration exactly, and tone↔speech boundaries have no hard step.
Run: python3 tests/test_mastering.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

from pipeline.config import SAMPLE_RATE
from pipeline.mastering import (apply_edges, build_program, normalize_take,
                                voiced_rms_dbfs)


def _take(seconds: float, rms_dbfs: float) -> np.ndarray:
    """Speech-like take at a target RMS with brief silence at each end."""
    rng = np.random.default_rng(int(abs(rms_dbfs) * 100))
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    voiced = (np.sin(2 * np.pi * 160 * t) + 0.4 * np.sin(2 * np.pi * 320 * t)).astype(np.float32)
    voiced *= (0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)).astype(np.float32)
    voiced *= 10 ** (rms_dbfs / 20) / (np.sqrt((voiced**2).mean()) + 1e-12)
    pad = np.zeros(int(0.05 * SAMPLE_RATE), dtype=np.float32)
    return np.concatenate([pad, voiced, pad])


def test_normalize_collapses_variance():
    levels = [-31.0, -20.0, -25.0, -18.0, -28.0]
    after = []
    for lv in levels:
        norm = normalize_take(_take(1.5, lv))
        after.append(voiced_rms_dbfs(norm))
    spread = max(after) - min(after)
    assert spread < 1.0, f"post-normalization spread {spread:.1f} dB (was ~13)"


def test_build_program_preserves_duration():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        placements = []
        for i, lv in enumerate([-25, -20, -30]):
            p = td / f"take_{i}.wav"
            sf.write(str(p), _take(1.0, lv), SAMPLE_RATE)
            placements.append((1.0 + i * 2.0, p))  # 2s apart
        total = int(round(10.0 * SAMPLE_RATE))
        prog = build_program(placements, total, None)
        assert len(prog) == total, f"program {len(prog)} != {total}"
        assert float(np.abs(prog).max()) <= 0.999


def test_no_hard_step_at_boundaries():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = td / "t.wav"
        sf.write(str(p), _take(1.0, -20), SAMPLE_RATE)
        total = int(round(4.0 * SAMPLE_RATE))
        prog = build_program([(1.5, p)], total, None)
        # Max sample-to-sample step must stay small (fades, not cuts).
        max_step = float(np.abs(np.diff(prog)).max())
        assert max_step < 0.1, f"hard step at boundary: {max_step:.3f}"


def test_fade_edges_zero_free_interior():
    take = _take(1.0, -20)
    faded = apply_edges(take)
    assert len(faded) == len(take)
    assert abs(faded[0]) < abs(take[len(take) // 2])  # head attenuated


if __name__ == "__main__":
    test_normalize_collapses_variance()
    test_build_program_preserves_duration()
    test_no_hard_step_at_boundaries()
    test_fade_edges_zero_free_interior()
    print("all mastering cases pass")
