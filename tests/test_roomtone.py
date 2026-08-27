"""VAD harvester test: harvested pause audio must contain no speech.

Fixture: a quiet tone bed with loud speech-like bursts inserted. The
harvested output must never include audio from a burst or its guard zone.
Run: python3 tests/test_roomtone.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

from pipeline.config import SAMPLE_RATE
from pipeline.roomtone import _vad_nonspeech_pool, pause_audio, room_tone


def make_fixture(path: str) -> None:
    """20s: -50 dBFS noise bed with three 1s speech-like bursts."""
    rng = np.random.default_rng(7)
    n = 20 * SAMPLE_RATE
    bed = rng.standard_normal(n).astype(np.float32) * (10 ** (-50 / 20))
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    # Speech-like burst: amplitude-modulated harmonics around 150 Hz (voiced).
    burst = (np.sin(2 * np.pi * 150 * t) + 0.5 * np.sin(2 * np.pi * 300 * t)
             + 0.3 * np.sin(2 * np.pi * 450 * t))
    burst *= (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)) * 0.3
    for start_s in (4, 10, 16):
        i = start_s * SAMPLE_RATE
        bed[i:i + len(burst)] += burst.astype(np.float32)
    sf.write(path, bed, SAMPLE_RATE)


def test_harvest_excludes_speech():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        make_fixture(f.name)
        pool, sr = _vad_nonspeech_pool(f.name, guard_ms=500)
        Path(f.name).unlink()
    assert len(pool) >= 3, "should find ambience frames in the quiet bed"
    # No harvested frame may contain burst-level energy.
    peak = float(np.abs(pool).max())
    assert peak < 0.02, f"speech leaked into harvest (peak {peak:.3f})"


def test_pause_audio_never_zeros():
    audio = pause_audio(SAMPLE_RATE)  # no reference: synth fallback
    assert len(audio) == SAMPLE_RATE
    assert float(np.abs(audio).max()) > 0, "pause audio must never be digital black"


def test_synth_tone_level():
    tone = room_tone(SAMPLE_RATE)
    rms_db = 20 * np.log10(float(np.sqrt((tone**2).mean())) + 1e-12)
    assert -60 < rms_db < -45, f"synthetic tone at {rms_db:.1f} dBFS"


if __name__ == "__main__":
    test_harvest_excludes_speech()
    test_pause_audio_never_zeros()
    test_synth_tone_level()
    print("all roomtone cases pass")
