"""Room tone: pause audio synthesized at ASSEMBLY time, never inside takes.

The harvester uses silero-VAD to find true non-speech spans in the tone
source (a <voice>.roomtone.wav sidecar or the reference itself), keeps only
audio at least `guard_ms` away from any speech boundary (breath onsets and
pre-phonation live at the edges), and loops the surviving frames with
equal-power crossfades. Modes, guard, and level are config; changing them
re-runs assembly only.

Config (config.yaml):
    room_tone:
      mode: harvest | file | synth | none
      source: auto | /path/to/tone.wav
      guard_ms: 500
      level_db: null   # null = keep source level; else force dBFS RMS
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from pipeline.config import CFG, SAMPLE_RATE

ROOM_CFG = CFG.get("room_tone", {})
ROOM_TONE_DB = float(CFG["mastering"].get("room_tone_db", -55.0))

_rng = np.random.default_rng(1431)  # fixed seed: deterministic renders


def room_tone(n_samples: int, level_db: float = ROOM_TONE_DB) -> np.ndarray:
    """Synthetic pink-ish tone: the fallback when no harvestable air exists."""
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    white = _rng.standard_normal(n_samples + 1).astype(np.float32)
    tone = np.empty(n_samples, dtype=np.float32)
    acc = 0.0
    a = 0.12
    for i in range(n_samples):
        acc += a * (white[i] - acc)
        tone[i] = acc
    rms = float(np.sqrt(np.mean(tone**2))) or 1.0
    tone *= (10 ** (level_db / 20)) / rms
    return _fade_edges(tone)


def _fade_edges(audio: np.ndarray) -> np.ndarray:
    fade = min(int(0.015 * SAMPLE_RATE), len(audio) // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        audio[:fade] *= ramp
        audio[-fade:] *= ramp[::-1]
    return audio


@lru_cache(maxsize=8)
def _vad_nonspeech_pool(source_path: str, guard_ms: int, frame_ms: float = 60.0) -> tuple:
    """Frames of true ambience: silero-VAD finds speech; we keep frames whose
    entire extent sits >= guard_ms away from every speech boundary."""
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    audio, sr = sf.read(source_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    model = load_silero_vad()
    vad_sr = 16000
    if sr != vad_sr:
        import torchaudio.functional as AF
        wav16 = AF.resample(torch.from_numpy(audio), sr, vad_sr)
    else:
        wav16 = torch.from_numpy(audio)
    speech = get_speech_timestamps(wav16, model, sampling_rate=vad_sr)

    guard = guard_ms / 1000
    spans = [(s["start"] / vad_sr - guard, s["end"] / vad_sr + guard) for s in speech]

    frame = int(sr * frame_ms / 1000)
    keep = []
    n_frames = len(audio) // frame
    for i in range(n_frames):
        t0, t1 = i * frame / sr, (i + 1) * frame / sr
        if any(t0 < e and t1 > s for s, e in spans):
            continue
        f = audio[i * frame:(i + 1) * frame]
        if float(np.abs(f).max()) < 1e-5:
            continue  # digital black is not air
        keep.append(f)
    return (np.asarray(keep, dtype=np.float32), sr)


def _loop_pool(pool: np.ndarray, src_sr: int, n_samples: int) -> np.ndarray:
    """Loop pool frames with equal-power crossfades at the delivery rate."""
    if src_sr != SAMPLE_RATE:
        import torch
        import torchaudio.functional as AF
        pool = AF.resample(torch.from_numpy(pool), src_sr, SAMPLE_RATE).numpy()
    frame = pool.shape[1]
    xfade = frame // 4
    ramp_in = np.sin(np.linspace(0, np.pi / 2, xfade, dtype=np.float32)) ** 2
    out = np.zeros(n_samples + frame, dtype=np.float32)
    pos = 0
    while pos < n_samples:
        f = pool[_rng.integers(len(pool))].copy()
        if _rng.random() < 0.5:
            f = f[::-1]
        f[:xfade] *= ramp_in
        f[-xfade:] *= ramp_in[::-1]
        out[pos: pos + frame] += f
        pos += frame - xfade
    return out[:n_samples]


def tone_source_for(reference_wav: str | Path | None) -> Path | None:
    """Resolve the tone source: config override, else <voice>.roomtone.wav
    sidecar, else the reference itself."""
    src = ROOM_CFG.get("source", "auto")
    if src and src != "auto":
        p = Path(src)
        return p if p.exists() else None
    if not reference_wav:
        return None
    ref = Path(reference_wav)
    sidecar = ref.with_suffix(".roomtone.wav")
    return sidecar if sidecar.exists() else (ref if ref.exists() else None)


def tone_bed(n_samples: int, reference_wav: str | Path | None = None) -> np.ndarray:
    """One continuous room-tone bed spanning the whole program: the surface
    speech is overlaid onto at assembly (pipeline.mastering). Same source and
    modes as pause_audio, produced as a single loop rather than per-gap chunks
    so there are no seams between pauses and speech."""
    return pause_audio(n_samples, reference_wav)


def pause_audio(n_samples: int, reference_wav: str | Path | None = None) -> np.ndarray:
    """The pause bed: mode-driven, VAD-harvested by default."""
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    mode = ROOM_CFG.get("mode", "harvest")
    if mode == "none":
        return np.zeros(n_samples, dtype=np.float32)
    if mode == "synth":
        return room_tone(n_samples)

    source = tone_source_for(reference_wav)
    if mode in ("harvest", "file") and source is not None:
        guard = int(ROOM_CFG.get("guard_ms", 500))
        try:
            pool, sr = _vad_nonspeech_pool(str(source), guard)
        except Exception:
            pool = np.zeros((0, 1), dtype=np.float32)
            sr = SAMPLE_RATE
        if len(pool) >= 3:
            tone = _loop_pool(pool, sr, n_samples)
            level_db = ROOM_CFG.get("level_db")
            if level_db is not None:
                rms = float(np.sqrt((tone**2).mean())) or 1.0
                tone *= (10 ** (float(level_db) / 20)) / rms
            return _fade_edges(tone)
    return room_tone(n_samples)


# Back-compat shims for callers not yet migrated (s5 engine, scratch scripts).
def harvest_room_tone(source: np.ndarray, n_samples: int, frame_ms: float = 60.0) -> np.ndarray:
    """Deprecated: array-input harvest without VAD. Kept for old scripts;
    prefer pause_audio(n, reference_wav)."""
    import tempfile

    import soundfile as sf

    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, source, SAMPLE_RATE)
        try:
            pool, sr = _vad_nonspeech_pool(f.name, int(ROOM_CFG.get("guard_ms", 500)))
        finally:
            Path(f.name).unlink(missing_ok=True)
    if len(pool) >= 3:
        return _fade_edges(_loop_pool(pool, sr, n_samples))
    return room_tone(n_samples)
