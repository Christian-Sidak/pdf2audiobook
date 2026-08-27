"""Per-take mastering, applied at ASSEMBLY time (takes stay dry and cached).

Two artifacts this fixes, both measured on the first cloned book:
  - clip-to-clip volume jumps (per-take RMS varied 11.4 dB) -> normalize each
    take to a common voiced-RMS target;
  - hard tone<->speech pops -> overlay normalized+faded takes onto ONE
    continuous room-tone bed at fixed offsets (no concatenation), so speech
    emerges from and settles into the same air, and total duration is
    unchanged (every chapter marker stays exact by construction).
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

from pipeline.config import CFG, SAMPLE_RATE

_M = CFG["mastering"]
_PT = _M.get("per_take", {})

FADE_IN_MS = float(_M.get("fade_in_ms", 24))
FADE_TAIL_MS = float(_M.get("fade_tail_ms", 12))
PER_TAKE_ENABLED = bool(_PT.get("enabled", True))
TARGET_RMS_DBFS = float(_PT.get("target_rms_dbfs", -20.0))
MAX_GAIN_DB = float(_PT.get("max_gain_db", 6.0))
MIN_GAIN_DB = float(_PT.get("min_gain_db", -12.0))
PEAK_CEILING_DBFS = float(_PT.get("peak_ceiling_dbfs", -1.0))
VOICED_FLOOR_DBFS = float(_PT.get("voiced_floor_dbfs", -45.0))
VOICED_REL_DB = float(_PT.get("voiced_rel_db", 25.0))
FRAME_MS = float(_PT.get("frame_ms", 20))
FAILED_TAKE_FLOOR_DBFS = float(_PT.get("failed_take_floor_dbfs", -40.0))


def load_take(path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import torch
        import torchaudio.functional as AF
        audio = AF.resample(torch.from_numpy(audio), sr, SAMPLE_RATE).numpy()
    return audio


def _dbfs(rms: float) -> float:
    return 20 * np.log10(max(rms, 1e-12))


def voiced_rms_dbfs(audio: np.ndarray) -> float:
    """RMS over SPEECH-ACTIVE frames only: excludes leading/trailing quiet and
    the stage-5 edge ramps, so the level isn't underestimated (which would
    over-amplify)."""
    frame = max(1, int(SAMPLE_RATE * FRAME_MS / 1000))
    n = len(audio) // frame
    if n < 1:
        return _dbfs(float(np.sqrt((audio**2).mean()))) if len(audio) else -120.0
    frames = audio[: n * frame].reshape(n, frame)
    rms = np.sqrt((frames**2).mean(axis=1))
    peak_frame = float(rms.max()) or 1e-12
    keep = rms[(rms > 10 ** (VOICED_FLOOR_DBFS / 20))
               & (rms > peak_frame * 10 ** (-VOICED_REL_DB / 20))]
    if len(keep) == 0:
        keep = rms
    return _dbfs(float(np.sqrt((keep**2).mean())))


def normalize_take(audio: np.ndarray) -> np.ndarray:
    """Scalar gain to the common voiced-RMS target, clamped, with a post-gain
    peak guard. One gain per take preserves intra-take dynamics.

    A take whose voiced level is implausibly low is a failed/near-silent
    render, not quiet speech — it is NOT boosted (that would amplify hiss);
    upstream WER/audio checks flag it instead. This is the real protection
    the max-gain clamp only approximated, so the clamp can be generous."""
    if not PER_TAKE_ENABLED or len(audio) == 0:
        return audio
    voiced = voiced_rms_dbfs(audio)
    if voiced < FAILED_TAKE_FLOOR_DBFS:
        return audio  # don't amplify a dead take into noise
    gain_db = TARGET_RMS_DBFS - voiced
    gain_db = float(np.clip(gain_db, MIN_GAIN_DB, MAX_GAIN_DB))
    gain = 10 ** (gain_db / 20)
    peak = float(np.abs(audio).max()) * gain
    ceiling = 10 ** (PEAK_CEILING_DBFS / 20)
    if peak > ceiling:
        gain *= ceiling / peak
    return audio * gain


def apply_edges(audio: np.ndarray) -> np.ndarray:
    """Equal-power (sin²) fade-in so speech emerges from the tone bed, and a
    fade-out tail that also hides any clipped final phoneme."""
    audio = audio.copy()
    n_in = min(int(SAMPLE_RATE * FADE_IN_MS / 1000), len(audio) // 2)
    n_out = min(int(SAMPLE_RATE * FADE_TAIL_MS / 1000), len(audio) // 2)
    if n_in > 0:
        audio[:n_in] *= np.sin(np.linspace(0, np.pi / 2, n_in, dtype=np.float32)) ** 2
    if n_out > 0:
        audio[-n_out:] *= np.sin(np.linspace(np.pi / 2, 0, n_out, dtype=np.float32)) ** 2
    return audio


def build_program(placements: list[tuple[float, object]], total_samples: int,
                  reference_wav: str | None) -> np.ndarray:
    """Overlay normalized+faded takes onto one continuous room-tone bed.

    placements: (start_s, take_path) in reading order.
    Takes are ADDED at fixed offsets — never concatenated — so the program is
    exactly total_samples long and all timing is preserved.
    """
    from pipeline.roomtone import tone_bed

    program = tone_bed(total_samples, reference_wav)
    for start_s, path in placements:
        take = apply_edges(normalize_take(load_take(path)))
        i = int(round(start_s * SAMPLE_RATE))
        end = min(i + len(take), total_samples)
        if end > i:
            program[i:end] += take[: end - i]
    peak = float(np.abs(program).max())
    if peak > 0.999:  # numeric safety before the ffmpeg limiter
        program *= 0.999 / peak
    return program
