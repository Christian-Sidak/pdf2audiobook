"""Stage 5: render narration segments to per-segment WAVs.

Engines: kokoro (fast draft renders, preset voice) and qwen3tts (final
renders, cloned voice). Every segment WAV is cached by a hash of
(engine, voice, params, text): reruns and targeted regenerations are free.

Artifact: 05_render/manifest.json + 05_render/segments/*.wav
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from evals.contracts import ARTIFACT_FILES, BookCtx, Violation
from pipeline.config import DEFAULT_ENGINE, SAMPLE_RATE, TTS_BATCH, TTS_ENGINES


class KokoroEngine:
    name = "kokoro"

    def __init__(self, voice: str, speed: float = 1.0):
        import torch
        from kokoro import KPipeline

        self.voice = voice
        self.speed = speed
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else None)
        try:
            self.pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)
        except Exception:
            self.pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

    def params(self) -> dict:
        return {"voice": self.voice, "speed": self.speed}

    def synthesize(self, text: str) -> np.ndarray:
        chunks = []
        for _, _, audio in self.pipeline(text, voice=self.voice, speed=self.speed):
            # Ramp chunk edges: raw concatenation leaves DC steps that become
            # impulse clicks (and AAC ringing overshoots any limiter on them).
            chunks.append(_edge_ramp(np.asarray(audio, dtype=np.float32)))
        if not chunks:
            raise RuntimeError(f"kokoro produced no audio for: {text[:60]!r}")
        return np.concatenate(chunks)


class Qwen3TTSEngine:
    """Cloned-voice final renders via Qwen3-TTS Base (zero-shot cloning).
    The voice-clone prompt is computed once from the reference WAV and reused
    for every segment."""

    name = "qwen3tts"

    def __init__(self, reference_wav: str, params: dict):
        self.MODEL_ID = params.get("model_id") or TTS_ENGINES["qwen3tts"]["model_id"]
        from qwen_tts import Qwen3TTSModel

        if not reference_wav:
            raise RuntimeError("qwen3tts requires TTS_ENGINES['qwen3tts']['reference_wav']")
        from pipeline.config import ROOT
        ref_path = Path(reference_wav)
        if not ref_path.is_absolute():
            ref_path = ROOT / ref_path
        self.reference_wav = str(ref_path)
        self._params = params

        import torch
        # bf16 everywhere: fp32 is ~2x slower on MPS, fp16 overflows this
        # model's logits into NaN; on CUDA bf16 is the native fast path.
        kwargs = {"dtype": torch.bfloat16}
        if torch.cuda.is_available():
            kwargs["device_map"] = "cuda:0"
            try:
                import flash_attn  # noqa: F401  — optional speedup when installed
                kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                pass
        elif torch.backends.mps.is_available():
            kwargs["device_map"] = "mps"
        try:
            self.model = Qwen3TTSModel.from_pretrained(self.MODEL_ID, **kwargs)
        except Exception:
            self.model = Qwen3TTSModel.from_pretrained(self.MODEL_ID)

        ref_text = params.get("ref_text")
        if not ref_text:
            # Non-English references (Czech, Arabic, ...) must carry their own
            # text: the auto-transcribe path below is an English-only model.
            sidecar = ref_path.with_suffix(".ref_text.txt")
            if sidecar.exists():
                ref_text = sidecar.read_text().strip()
        if not ref_text:
            from evals.audioutil import transcribe
            ref_text = transcribe(ref_path).strip()
        self._ref_text = ref_text

        # Room-tone source: the reference RECORDING's real air, not the
        # codec's floor (and never breaths: the harvester filters those).
        # A <voice>.roomtone.wav sidecar wins when present: it can be a longer
        # ambience-rich cut without the sentence-clean constraints of the
        # cloning reference.
        import soundfile as sf
        sidecar = ref_path.with_suffix(".roomtone.wav")
        self._room_source, _ = sf.read(str(sidecar if sidecar.exists() else ref_path),
                                       dtype="float32")
        self.prompt = self.model.create_voice_clone_prompt(
            ref_audio=self.reference_wav, ref_text=ref_text)

        langs = [l for l in self.model.get_supported_languages()]
        self.language = next((l for l in langs if "en" in l.lower()), langs[0] if langs else "English")

    def params(self) -> dict:
        return {"reference_wav": self.reference_wav, **{k: v for k, v in self._params.items()
                                                        if k != "ref_text"}}

    def _postprocess(self, wav, sr: int) -> np.ndarray:
        audio = np.asarray(wav, dtype=np.float32)
        if sr != SAMPLE_RATE:
            import torch
            import torchaudio.functional as AF
            audio = AF.resample(torch.from_numpy(audio), sr, SAMPLE_RATE).numpy()
        tempo = float(self._params.get("tempo", 1.0))
        if tempo != 1.0:
            audio = _sox_tempo(audio, tempo)
        return audio

    def synthesize(self, text: str) -> np.ndarray:
        """DRY take: pure speech, no pause audio. Segments are sentence-level
        (IR v2); every gap is inserted at assembly from config, so pause and
        room-tone changes never orphan recorded takes. The collapse guard
        self-truncates runaway generations at ~4x expected length."""
        wavs, sr = self.model.generate_voice_clone(
            text=text, language=self.language, voice_clone_prompt=self.prompt,
            max_new_tokens=_collapse_cap_tokens(text, self._params))
        return self._postprocess(wavs[0], sr)

    def synthesize_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Batched generation amortizes prefill: the main MPS speed lever."""
        wavs, sr = self.model.generate_voice_clone(
            text=texts, language=[self.language] * len(texts),
            voice_clone_prompt=self.prompt * len(texts) if isinstance(self.prompt, list)
            else self.prompt)
        return [self._postprocess(w, sr) for w in wavs]


from pipeline.config import CFG

_GUARD = CFG["tts"].get("collapse_guard", {})


def _expected_seconds(text: str, params: dict) -> float:
    """Expected take length from word count at a natural reading pace."""
    words = max(1, len(text.split()))
    return words / float(_GUARD.get("expected_wpm", 150)) * 60


def _collapse_cap_tokens(text: str, params: dict) -> int:
    cap_s = max(15.0, float(_GUARD.get("cap_factor", 4.0)) * _expected_seconds(text, params))
    return int(cap_s * float(_GUARD.get("tokens_per_sec", 13.5)))


def hot_ending(audio: np.ndarray, tail_ms: float = 60.0) -> bool:
    """A take that ends at speech-level energy had its final phoneme release
    amputated (premature EOS). Natural speech decays to the floor; ASR still
    hears the word, so only energy analysis catches this."""
    n_tail = int(SAMPLE_RATE * tail_ms / 1000)
    if len(audio) < n_tail * 4:
        return False
    tail_rms = float(np.sqrt((audio[-n_tail:] ** 2).mean()))
    body_rms = float(np.sqrt((audio ** 2).mean())) or 1e-9
    return tail_rms > float(_GUARD.get("end_decay_ratio", 0.35)) * body_rms


def _make_engine(name: str, params: dict):
    if name == "kokoro":
        return KokoroEngine(voice=params["voice"], speed=params.get("speed", 1.0))
    if name == "qwen3tts":
        return Qwen3TTSEngine(reference_wav=params.get("reference_wav"), params=params)
    raise ValueError(f"unknown TTS engine: {name}")


def _pace_to_target(audio: np.ndarray, text: str, params: dict) -> np.ndarray:
    """Adaptive pacing: the model's base pace varies with content, so a fixed
    multiplier over- or under-shoots. Measure each segment's natural wpm and
    stretch only as far as that segment needs, clamped so stretch artifacts
    stay inaudible."""
    target = params.get("target_wpm")
    if not target:
        return audio
    words = len(text.split())
    duration = len(audio) / SAMPLE_RATE
    if words < 3 or duration <= 0:
        return audio
    natural = words / duration * 60
    # Clamp tight: beyond ~12% stretch WSOLA smears audibly; the rest of the
    # slowdown comes from sentence pauses, not stretching.
    tempo = max(0.88, min(1.08, target / natural))
    if abs(tempo - 1.0) < 0.03:
        return audio
    return _sox_tempo(audio, tempo)


def _sox_tempo(audio: np.ndarray, tempo: float) -> np.ndarray:
    """Pitch-preserving pace change via sox WSOLA in speech mode. The clone
    inherits a generic reading pace, not the reference speaker's; tempo < 1
    slows it back down (e.g. 0.75 ≈ generic 140 wpm -> a measured ~105)."""
    import subprocess
    import tempfile

    import soundfile as sf

    with tempfile.TemporaryDirectory() as td:
        a, b = f"{td}/in.wav", f"{td}/out.wav"
        sf.write(a, audio, SAMPLE_RATE)
        subprocess.run(["sox", a, b, "tempo", "-s", str(tempo)], check=True, capture_output=True)
        out, _ = sf.read(b, dtype="float32")
    return out


def segment_hash(engine: str, params: dict, text: str) -> str:
    key = json.dumps({"engine": engine, "params": params, "text": text}, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def _render_dir(book: BookCtx) -> Path:
    d = book.artifacts_dir / "05_render"
    (d / "segments").mkdir(parents=True, exist_ok=True)
    return d


def _edge_ramp(audio: np.ndarray, ms: float = 3.0) -> np.ndarray:
    n = min(int(SAMPLE_RATE * ms / 1000), len(audio) // 2)
    if n > 0:
        ramp = np.sin(np.linspace(0, np.pi / 2, n)) ** 2
        audio = audio.copy()
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
    return audio


def _write_wav(path: Path, audio: np.ndarray) -> float:
    import soundfile as sf

    sf.write(str(path), _edge_ramp(audio), SAMPLE_RATE)
    return len(audio) / SAMPLE_RATE


def _base_params(engine_name: str, jitter: float = 0.0, overrides: dict | None = None) -> dict:
    """Engine params from defaults + per-build overrides (--voice on the CLI:
    a preset name for kokoro, a reference WAV path for qwen3tts)."""
    cfg = TTS_ENGINES[engine_name]
    voice = (overrides or {}).get("voice")
    if engine_name == "kokoro":
        params = {"voice": voice or cfg["voice"], "speed": cfg.get("speed", 1.0) + jitter,
                  "post": "fade3ms"}  # cache-version bump: chunk-edge ramps
    else:
        model_id = (overrides or {}).get("tts_model") or cfg["model_id"]
        ref = voice or cfg.get("reference_wav")
        params = {"reference_wav": ref, "model_id": model_id, "post": "fade3ms"}
        # Reference CONTENT in the cache key: re-editing a voice file
        # (denoise, re-trim) must invalidate takes made from the old audio.
        if ref:
            from pipeline.config import ROOT
            ref_path = Path(ref) if Path(ref).is_absolute() else ROOT / ref
            if ref_path.exists():
                params["voice_sha"] = hashlib.sha256(ref_path.read_bytes()).hexdigest()[:8]
        # NOTE: no pause/tone parameters in the cache key — the dry-takes
        # invariant that makes pause iteration free.
        target_wpm = (overrides or {}).get("target_wpm") or cfg.get("target_wpm")
        if target_wpm:
            params["target_wpm"] = int(target_wpm)
        else:
            params["tempo"] = float((overrides or {}).get("tempo") or cfg.get("tempo", 1.0))
    if jitter:
        params["jitter"] = jitter
    return params


def _render(book: BookCtx, only_segments: set[str] | None = None,
            param_jitter: float = 0.0) -> Path:
    engine_name = book.config.get("tts_engine", DEFAULT_ENGINE)
    engine = None  # lazy: cached segments never load the model
    params = _base_params(engine_name, param_jitter, book.config)

    narration = json.loads((book.artifacts_dir / ARTIFACT_FILES["narration"]).read_text(encoding="utf-8"))
    only_ch = book.config.get("only_chapters")
    if only_ch:
        # honor --chapters at render time too (multi-voice casting renders a
        # different voice per chapter group; without this every pass renders
        # the whole book)
        narration = {**narration,
                     "segments": [s for s in narration["segments"]
                                  if s.get("chapter_id") in only_ch]}
    render_dir = _render_dir(book)
    manifest_path = book.artifacts_dir / ARTIFACT_FILES["render_manifest"]

    # Batched pre-render for engines that support it: fill the cache in
    # batches, then the per-segment loop below finds everything cached.
    batch_size = int(book.config.get("tts_batch", TTS_BATCH))
    pending = []
    for seg in narration["segments"]:
        if only_segments is not None and seg["id"] not in only_segments:
            continue
        h = segment_hash(engine_name, params, seg["text"])
        wav_path = render_dir / "segments" / f"{h}.wav"
        if not wav_path.exists() and not any(p[0] == wav_path for p in pending):
            pending.append((wav_path, seg["text"]))
    if pending and batch_size > 1:
        probe = _make_engine(engine_name, params)
        if hasattr(probe, "synthesize_batch"):
            engine = probe
            done = 0
            for i in range(0, len(pending), batch_size):
                batch = pending[i:i + batch_size]
                for (wav_path, _), audio in zip(batch, engine.synthesize_batch([t for _, t in batch])):
                    _write_wav(wav_path, audio)
                done += len(batch)
                print(f"  batched {done}/{len(pending)}", flush=True)
        else:
            engine = probe  # reuse the loaded model in the loop below
    old_rows = {}
    if manifest_path.exists():
        old_rows = {r["segment_id"]: r for r in json.loads(manifest_path.read_text())["segments"]}

    rows = []
    total = len(narration["segments"])
    for i, seg in enumerate(narration["segments"]):
        sid = seg["id"]
        if only_segments is not None and sid not in only_segments and sid in old_rows:
            rows.append(old_rows[sid])
            continue
        h = segment_hash(engine_name, params, seg["text"])
        wav_path = render_dir / "segments" / f"{h}.wav"
        if not wav_path.exists():
            if engine is None:
                engine = _make_engine(engine_name, params)
            audio = engine.synthesize(seg["text"])
            # Collapse escape: a take far beyond expected length is babble;
            # one fresh sampling path almost always leaves the basin.
            expected = _expected_seconds(seg["text"], params)
            escape = float(_GUARD.get("escape_threshold", 3.5))
            if engine_name == "qwen3tts" and len(audio) / SAMPLE_RATE > escape * expected:
                print(f"    collapse detected ({len(audio)/SAMPLE_RATE:.0f}s vs ~{expected:.0f}s "
                      f"expected); resampling {seg['id']}", flush=True)
                retake = engine.synthesize(seg["text"])
                if len(retake) < len(audio):
                    audio = retake
            if engine_name == "qwen3tts" and hot_ending(audio):
                print(f"    hot ending (clipped final phoneme); resampling {seg['id']}", flush=True)
                retake = engine.synthesize(seg["text"])
                if not hot_ending(retake):
                    audio = retake
            audio = _pace_to_target(audio, seg["text"], params)
            duration = _write_wav(wav_path, audio)
        else:
            import soundfile as sf
            duration = sf.info(str(wav_path)).duration
        rows.append({
            "segment_id": sid, "chapter_id": seg["chapter_id"], "type": seg["type"],
            "wav": str(wav_path.relative_to(book.artifacts_dir)), "hash": h,
            "engine": engine_name, "params": params,
            "duration_s": round(duration, 3), "chars": len(seg["text"]),
        })
        if (i + 1) % 50 == 0:
            print(f"  rendered {i + 1}/{total}", flush=True)

    manifest = {"engine": engine_name, "params": params, "sample_rate": SAMPLE_RATE,
                "segments": rows}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    return manifest_path


def run(book: BookCtx) -> Path:
    return _render(book)


ADJUDICATE_SCHEMA = {
    "type": "object", "required": ["action", "text"],
    "properties": {"action": {"type": "string", "enum": ["rewrite", "accept"]},
                   "text": {"type": "string"}, "reason": {"type": "string"}},
}

ADJUDICATE_PROMPT = """A text-to-speech narrator repeatedly fails on this passage (looping, garbling, or mispronouncing). Failure notes: {notes}

PASSAGE:
{text}

Decide: if the passage can be made easier to speak, action "rewrite" with a version that PRESERVES THE MEANING EXACTLY but is TTS-friendly: split overlong sentences, replace hard-to-pronounce foreign spellings with natural readable forms (e.g. 'Beaupère' -> 'Beaupere'), expand awkward abbreviations. If it is already as speakable as it can be, action "accept" with the original text.

Return JSON: {{"action": "rewrite"|"accept", "text": "...", "reason": "..."}}"""


def _adjudicate(seg: dict, notes: str, model: str) -> str | None:
    """The hard track for stubborn takes: the LLM adjusts the TEXT (meaning
    preserved, speakability restored) instead of endlessly re-rolling audio."""
    from pipeline.ollama_client import OllamaError, chat_json

    try:
        result = chat_json(model, [{"role": "user", "content": ADJUDICATE_PROMPT.format(
            notes=notes[:300], text=seg["text"])}], ADJUDICATE_SCHEMA, temperature=0.2)
    except OllamaError:
        return None
    if result.get("action") == "rewrite" and result.get("text", "").strip():
        return result["text"].strip()
    return None


def retry(book: BookCtx, feedback: list[Violation]) -> Path:
    """Re-render only the segments named in violations; later attempts jitter
    generation params, and stubborn segments get their text adjudicated by
    the LLM before the final attempts."""
    bad = {v.unit_id for v in feedback if v.unit_id and v.unit_id.startswith("seg_")}
    if not bad:
        return run(book)
    attempt = int(book.config.get("_render_attempt", 0)) + 1
    book.config["_render_attempt"] = attempt
    jitter = 0.02 * (attempt - 1)

    if attempt >= int(_GUARD.get("adjudicate_after", 2)):
        narration_path = book.artifacts_dir / ARTIFACT_FILES["narration"]
        narration = json.loads(narration_path.read_text(encoding="utf-8"))
        notes_by_seg = {v.unit_id: v.message for v in feedback if v.unit_id}
        model = book.config.get("rewrite_model") or CFG["narration"]["rewrite_model"]
        changed = 0
        for seg in narration["segments"]:
            if seg["id"] in bad:
                new_text = _adjudicate(seg, notes_by_seg.get(seg["id"], ""), model)
                if new_text and new_text != seg["text"]:
                    seg["text"] = new_text
                    seg["adjudicated"] = True
                    changed += 1
        if changed:
            narration_path.write_text(json.dumps(narration, indent=1), encoding="utf-8")
            print(f"    adjudicated {changed} stubborn segments (text adjusted for speakability)",
                  flush=True)
    # Drop stale cache entries for the bad segments so they regenerate.
    narration = json.loads((book.artifacts_dir / ARTIFACT_FILES["narration"]).read_text(encoding="utf-8"))
    engine_name = book.config.get("tts_engine", DEFAULT_ENGINE)
    for seg in narration["segments"]:
        if seg["id"] in bad and jitter == 0.0:
            h = segment_hash(engine_name, _base_params(engine_name, overrides=book.config), seg["text"])
            (_render_dir(book) / "segments" / f"{h}.wav").unlink(missing_ok=True)
    return _render(book, only_segments=bad, param_jitter=jitter)
