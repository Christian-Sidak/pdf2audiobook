"""Content-addressed take cache for audition/scratch work.

Same invariant as the render pipeline's segment cache (s5_render.segment_hash):
a take's filename is the hash of (engine, params, text), where params includes
a sha of the voice reference file. Takes are NEVER deleted or overwritten —
changed text or a re-edited voice hashes to a new file, and every take ever
generated (including user-approved ones) stays addressable. Mix-level changes
(music, pauses, levels) must read from this cache, never regenerate.

An index sidecar (index.json) maps hash -> {text, voice, wpm, created} so
humans can browse what each file is.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.config import SAMPLE_RATE
from pipeline.s5_render import segment_hash


def take_path(cache_dir: str | Path, engine: str, params: dict, text: str) -> Path:
    return Path(cache_dir) / (segment_hash(engine, params, text) + ".wav")


def get_or_generate(cache_dir: str | Path, engine: str, params: dict, text: str,
                    generate) -> tuple[np.ndarray, Path, bool]:
    """Return (audio, path, was_cached). `generate` is only called on a miss;
    it must return a 1-D float array at SAMPLE_RATE (do QC/selection inside it
    so only accepted takes are ever cached)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = take_path(cache_dir, engine, params, text)
    if p.exists():
        a, _ = sf.read(str(p), dtype="float32")
        return a, p, True
    a = np.asarray(generate(), dtype=np.float32)
    sf.write(str(p), a, SAMPLE_RATE)
    idx_path = cache_dir / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    idx[p.stem] = {"text": text[:160], "voice": str(params.get("reference_wav", "?")),
                   "target_wpm": params.get("target_wpm"),
                   "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    idx_path.write_text(json.dumps(idx, indent=1, ensure_ascii=False))
    return a, p, False


def reject(cache_dir: str | Path, engine: str, params: dict, text: str) -> Path | None:
    """Invalidate ONE take that a listener judged bad: its key regenerates on
    the next get_or_generate. The audio is moved to rejected/ (never deleted)
    so even bad takes stay auditable and nothing is ever unrecoverable."""
    cache_dir = Path(cache_dir)
    p = take_path(cache_dir, engine, params, text)
    if not p.exists():
        return None
    rej = cache_dir / "rejected"
    rej.mkdir(exist_ok=True)
    dest = rej / (p.stem + time.strftime("-%Y%m%d%H%M%S") + ".wav")
    p.rename(dest)
    idx_path = cache_dir / "index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        if p.stem in idx:
            idx[p.stem]["rejected"] = dest.name
            idx_path.write_text(json.dumps(idx, indent=1, ensure_ascii=False))
    return dest
