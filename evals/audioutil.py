"""Audio measurement utilities: whisper round-trip (cached), ffprobe,
loudnorm measurement, silence detection."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ASR_CACHE = Path(__file__).resolve().parent / ".cache" / "asr"
WHISPER_MODEL = "mlx-community/whisper-small.en-mlx"  # darwin (mlx)
FW_MODEL = "small.en"                                 # linux/CUDA (faster-whisper)
_FW = None


def _fw_transcribe(wav: Path) -> dict:
    """faster-whisper backend for non-Apple hosts (same small.en weights)."""
    global _FW
    if _FW is None:
        import torch
        from faster_whisper import WhisperModel
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _FW = WhisperModel(FW_MODEL, device=dev,
                           compute_type="float16" if dev == "cuda" else "int8")
    segments, _ = _FW.transcribe(str(wav), word_timestamps=True)
    segments = list(segments)
    words = [w for s in segments for w in (s.words or [])]
    return {"text": "".join(s.text for s in segments),
            "speech_start": words[0].start if words else None,
            "speech_end": words[-1].end if words else None}


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:24]


def transcribe_timed(wav: Path) -> dict:
    """Whisper transcription with first/last word times, disk-cached by audio
    content hash: {"text", "speech_start", "speech_end"}. One pass serves both
    the take-review (text) and the pre-speech artifact check (timing). Cache
    entries written before timing existed are upgraded in place."""
    key = _file_hash(wav)
    cache = ASR_CACHE / f"{key}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        if "speech_start" in d:
            return d

    import platform
    if platform.system() == "Darwin":
        import mlx_whisper
        r = mlx_whisper.transcribe(str(wav), path_or_hf_repo=WHISPER_MODEL,
                                   word_timestamps=True)
        words = [w for s in r["segments"] for w in s.get("words", [])]
        d = {"text": r["text"],
             "speech_start": words[0]["start"] if words else None,
             "speech_end": words[-1]["end"] if words else None}
    else:
        d = _fw_transcribe(wav)
    ASR_CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(d))
    return d


def transcribe(wav: Path) -> str:
    """Whisper transcription text, disk-cached by audio content hash."""
    return transcribe_timed(wav)["text"]


def ffprobe_json(path: Path, *extra: str) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
         "-show_streams", *extra, str(path)],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}")
    return json.loads(result.stdout)


def container_duration(path: Path) -> float:
    return float(ffprobe_json(path)["format"]["duration"])


def container_chapters(path: Path) -> list[dict]:
    return ffprobe_json(path, "-show_chapters").get("chapters", [])


def measure_loudness(path: Path) -> dict:
    """loudnorm measurement pass: input_i (LUFS), input_tp (dBTP), etc."""
    result = subprocess.run(
        ["ffmpeg", "-i", str(path), "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True)
    stderr = result.stderr
    start, end = stderr.rfind("{"), stderr.rfind("}")
    if start < 0:
        raise RuntimeError(f"loudnorm measurement failed on {path}")
    return json.loads(stderr[start:end + 1])


def detect_silences(path: Path, noise_db: int = -45, min_s: float = 0.25) -> list[tuple[float, float]]:
    """(start, end) of silence runs via ffmpeg silencedetect."""
    result = subprocess.run(
        ["ffmpeg", "-i", str(path), "-af", f"silencedetect=noise={noise_db}dB:d={min_s}",
         "-f", "null", "-"], capture_output=True, text=True)
    starts = [float(m) for m in re.findall(r"silence_start: ([\d.]+)", result.stderr)]
    ends = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", result.stderr)]
    return list(zip(starts, ends))


def wav_peak_and_sr(path: Path) -> tuple[float, int, float]:
    """(peak_abs [0..1], sample_rate, duration_s) via soundfile."""
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32")
    peak = float(np.abs(data).max()) if len(data) else 0.0
    return peak, sr, len(data) / sr if sr else 0.0
