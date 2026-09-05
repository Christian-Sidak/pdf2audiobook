"""Stage 5 (render) checks: LLM take review of whisper transcripts, duration
outliers, audio validity, cache integrity."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from evals.artifacts import ArtifactSet
from evals.checks import check
from evals.contracts import CheckResult, Violation
from evals.corpus import DocSpec


def _rows(art: ArtifactSet) -> list[dict]:
    return art.render_manifest["segments"]


def _segment_texts(art: ArtifactSet) -> dict[str, str]:
    return {s["id"]: s["text"] for s in art.narration["segments"]}


REVIEW_CACHE = Path(__file__).resolve().parent.parent / ".cache" / "take_review"
REVIEW_PROMPT_VERSION = "v1"

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"enum": ["pass", "retake"]},
        "issues": {"type": "array", "items": {"enum": [
            "stutter", "dropped_final_words", "start_artifact", "skipped_words",
            "inserted_content", "looping", "garbled"]}},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "issues", "reason"],
}

REVIEW_PROMPT = """You are reviewing one take from a text-to-speech audiobook render. \
SOURCE is what the narrator was supposed to speak. TRANSCRIPT is a Whisper ASR \
transcription of the audio that was actually generated.

Flag the take (verdict "retake") if the transcript shows evidence of TTS failure:
- stutter: a word, syllable, or phrase repeated back-to-back that is not repeated in the source
- dropped_final_words: the transcript ends before the source does (final words or the final \
sentence missing; TTS often truncates the last phoneme or word)
- start_artifact: extraneous syllables, a stray word, or a repeated first word at the very \
beginning that is not in the source
- skipped_words: words or whole phrases from the middle of the source missing from the transcript
- inserted_content: words or sentences in the transcript that are not in the source at all
- looping: the same source phrase spoken more than once
- garbled: a stretch where the transcript is babble that matches nothing in the source

Do NOT flag transcription formatting differences. These are normal ASR behavior, not audio \
errors: numbers as digits vs words ('204' vs 'two hundred four'), punctuation, capitalization, \
contractions ('do not' vs 'don't'), homophones, and misspelled proper nouns or foreign words \
(Whisper guesses spellings). Judge only whether the spoken audio matched the source.

SOURCE:
{source}

TRANSCRIPT:
{transcript}

Return JSON: {{"verdict": "pass"|"retake", "issues": [...], "reason": "one short sentence"}}"""


def _review_take(source: str, transcript: str, model: str) -> dict:
    """Qwen verdict on one take, disk-cached by (prompt, model, source, transcript)."""
    import hashlib

    from pipeline.ollama_client import chat_json

    key = hashlib.sha256(
        f"{REVIEW_PROMPT_VERSION}|{model}|{source}|{transcript}".encode()).hexdigest()[:24]
    cache = REVIEW_CACHE / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    result = chat_json(model, [{"role": "user", "content": REVIEW_PROMPT.format(
        source=source, transcript=transcript)}], REVIEW_SCHEMA,
        temperature=0.0, num_ctx=4096)
    if result.get("verdict") not in ("pass", "retake"):
        result = {"verdict": "pass", "issues": [], "reason": "malformed review, not counted"}
    REVIEW_CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result))
    return result


@check(stage=5, dimension="take_review", deterministic=False)
def take_review(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    """LLM review of every take: whisper transcript vs source text, judged by
    the configured judge model for stutters, dropped final words, start
    artifacts, skips, insertions, looping, and babble. Transcripts are cached
    by WAV hash and verdicts by (source, transcript) hash, so reruns and
    retake loops only pay for regenerated takes."""
    from evals.audioutil import transcribe
    from pipeline.config import JUDGE_MODEL
    from pipeline.ollama_client import OllamaError

    texts = _segment_texts(art)
    rows = _rows(art)
    violations = []
    counts: dict[str, int] = {}
    errors = 0
    for r in rows:
        source = texts.get(r["segment_id"], "")
        if len(source.strip()) < 3:
            continue
        transcript = transcribe(art.book_dir / r["wav"]).strip()
        try:
            review = _review_take(source, transcript, JUDGE_MODEL)
        except OllamaError:
            errors += 1
            continue
        if review["verdict"] == "retake":
            for issue in review.get("issues", []) or ["unspecified"]:
                counts[issue] = counts.get(issue, 0) + 1
            violations.append(Violation(
                message=f"{','.join(review.get('issues', []))}: {review.get('reason', '')[:120]} "
                        f"| source {source[:60]!r} heard {transcript[:60]!r}",
                unit_id=r["segment_id"], fixable=True))
    details = {"checked": len(rows), "issue_counts": counts, "review_errors": errors}
    if errors and errors > len(rows) // 10:
        return CheckResult.failed("take_review", 5, [Violation(
            message=f"{errors} takes could not be reviewed (Ollama unreachable?)")], **details)
    if violations:
        # Full list: the retry loop re-renders exactly what is returned here.
        # Capping at 25 made a 14%-failure book unclearable in four attempts
        # (Carnegie 2026-09-05). Display and the review queue truncate on
        # their own.
        return CheckResult.failed("take_review", 5, violations, total=len(violations), **details)
    return CheckResult.passed("take_review", 5, **details)


@check(stage=5, dimension="sec_per_char_outlier")
def sec_per_char_outlier(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    band = cfg.get("sec_per_char", {"min": 0.045, "max": 0.14})
    rows = [r for r in _rows(art) if r["chars"] >= 40]
    if not rows:
        return CheckResult.skipped("sec_per_char_outlier", 5, "no long segments")
    ratios = {r["segment_id"]: r["duration_s"] / r["chars"] for r in rows}
    med = statistics.median(ratios.values())
    mad = statistics.median(abs(v - med) for v in ratios.values()) or 0.005

    violations = []
    # This check hunts truncation (audio far shorter than text) and looping
    # (far longer), not ordinary pacing variation: slow deliberate prose at
    # 2x median is fine speech.
    for r in rows:
        v = ratios[r["segment_id"]]
        if not (band["min"] <= v <= band["max"]) or v > 2.2 * med or v < 0.45 * med:
            violations.append(Violation(
                message=f"{v:.3f}s/char (median {med:.3f}): truncated or looping audio",
                unit_id=r["segment_id"], fixable=True))
    if violations:
        return CheckResult.failed("sec_per_char_outlier", 5, violations,
                                  total=len(violations), median=round(med, 4))
    return CheckResult.passed("sec_per_char_outlier", 5, median=round(med, 4))


@check(stage=5, dimension="segment_audio_valid")
def segment_audio_valid(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    from evals.audioutil import wav_peak_and_sr

    expected_sr = art.render_manifest.get("sample_rate", 24000)
    violations = []
    for r in _rows(art):
        path = art.book_dir / r["wav"]
        if not path.exists():
            violations.append(Violation(message="missing WAV", unit_id=r["segment_id"], fixable=True))
            continue
        peak, sr, duration = wav_peak_and_sr(path)
        if sr != expected_sr or duration <= 0.05 or peak == 0.0 or peak > 0.999:
            violations.append(Violation(
                message=f"bad audio: sr={sr}, dur={duration:.2f}s, peak={peak:.3f}",
                unit_id=r["segment_id"], fixable=True))
            continue
        import soundfile as sf

        from pipeline.s5_render import hot_ending
        audio, _ = sf.read(str(path), dtype="float32")
        if hot_ending(audio):
            violations.append(Violation(
                message="hot ending: take ends at speech-level energy (clipped final phoneme)",
                unit_id=r["segment_id"], fixable=True))
    if violations:
        return CheckResult.failed("segment_audio_valid", 5, violations[:20], total=len(violations))
    return CheckResult.passed("segment_audio_valid", 5, segments=len(_rows(art)))


@check(stage=5, dimension="start_artifact_shape")
def start_artifact_shape(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    """Pre-speech artifact detector: a short energy blob at the very start
    followed by a silent gap before sustained speech (the clone reproducing a
    breath/onset tic from the reference). Shape-based because neither whisper
    (first-word timestamps clamp to 0) nor raw head energy (loud first words
    are normal) can localize these. Tuned on 1,045 takes: 6.9% hit rate,
    majority confirmed artifacts by ear; false positives only cost a retake."""
    import numpy as np
    import soundfile as sf

    frame_s = 0.02
    violations = []
    for r in _rows(art):
        path = art.book_dir / r["wav"]
        if not path.exists():
            continue
        a, sr = sf.read(str(path), dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        body = float(np.sqrt((a ** 2).mean())) or 1e-9
        n = int(frame_s * sr)
        nf = len(a) // n
        if nf < 30:
            continue
        act = np.sqrt((a[:nf * n].reshape(nf, n) ** 2).mean(1)) > 0.1 * body
        regs, start = [], None
        for i, v in enumerate(act):
            if v and start is None:
                start = i
            if not v and start is not None:
                regs.append((start * frame_s, i * frame_s))
                start = None
                if len(regs) >= 2:
                    break
        if len(regs) < 2:
            continue
        (s1, e1), (s2, _) = regs[0], regs[1]
        if s1 < 0.15 and (e1 - s1) <= 0.40 and (s2 - e1) >= 0.20:
            violations.append(Violation(
                message=f"pre-speech artifact: {e1 - s1:.2f}s blob then "
                        f"{s2 - e1:.2f}s gap before speech",
                unit_id=r["segment_id"], fixable=True))
    if violations:
        return CheckResult.failed("start_artifact_shape", 5, violations,
                                  total=len(violations))
    return CheckResult.passed("start_artifact_shape", 5, checked=len(_rows(art)))


@check(stage=5, dimension="speaker_similarity", deterministic=False)
def speaker_similarity(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    """Voice-identity gate: zero-shot cloning occasionally drifts to a
    different voice entirely, which every transcript- and duration-based
    check is deaf to (same words, wrong speaker).

    Each take is embedded (resemblyzer, cached by audio hash so a take is
    embedded once for the life of the build) and compared to the CLONE
    REFERENCE the render was made from. Calibrated on the Carnegie/Spader
    render 2026-09-05 against a distilled reference: long takes median 0.89,
    1st percentile 0.79, min 0.77; 1-2.5s takes 1st percentile 0.57. A raw
    interview reference (TV channel, room) depresses every score, so when the
    render's median reference-similarity is poor the check falls back to the
    render's own on-voice centroid, the pre-2026-09-05 method. Short takes
    embed noisily and get a lower bar; sub-second takes are skipped (the
    embedding window is ~1.6s)."""
    import numpy as np

    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        return CheckResult.skipped("speaker_similarity", 5, "resemblyzer not installed")
    import soundfile as sf

    from evals.audioutil import _file_hash
    from pipeline.config import CFG, ROOT

    scfg = dict(CFG.get("tts", {}).get("speaker_similarity") or {})
    min_long = float(scfg.get("ref_min_long", 0.70))
    min_short = float(scfg.get("ref_min_short", 0.50))
    fallback_median = float(scfg.get("centroid_fallback_median", 0.75))
    cache_dir = REVIEW_CACHE.parent / "speaker_emb"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = [r for r in _rows(art) if (art.book_dir / r["wav"]).exists()]
    if len(rows) < 20:
        return CheckResult.skipped("speaker_similarity", 5, "too few takes")
    enc = None

    def embed(path):
        nonlocal enc
        key = _file_hash(path)
        c = cache_dir / f"{key}.npy"
        if c.exists():
            return np.load(c)
        if enc is None:
            enc = VoiceEncoder(verbose=False)
        e = enc.embed_utterance(preprocess_wav(str(path)))
        np.save(c, e)
        return e

    embs, durs = {}, {}
    for r in rows:
        p = art.book_dir / r["wav"]
        try:
            embs[r["segment_id"]] = embed(p)
            durs[r["segment_id"]] = sf.info(str(p)).duration
        except Exception:
            continue
    if len(embs) < 20:
        return CheckResult.skipped("speaker_similarity", 5, "embedding failures")

    # Anchor: the clone reference from the manifest params.
    manifest = json.loads((art.book_dir / "05_render" / "manifest.json").read_text())
    ref = (manifest.get("params") or {}).get("reference_wav")
    ref_path = None
    if ref:
        ref_path = Path(ref) if Path(ref).is_absolute() else ROOT / ref
        if not ref_path.exists():
            ref_path = None
    anchor = "reference"
    if ref_path is not None:
        ref_emb = embed(ref_path)
        sims = {sid: float(np.dot(ref_emb, e)) for sid, e in embs.items()}
        long_sims = [v for s, v in sims.items() if durs.get(s, 0) >= 2.5]
        if long_sims and float(np.median(long_sims)) < fallback_median:
            anchor = "centroid"  # raw/noisy reference depresses everything
    else:
        anchor = "centroid"
    if anchor == "centroid":
        mat = np.stack(list(embs.values()))
        c = mat.mean(0)
        c /= np.linalg.norm(c)
        sims = {sid: float(np.dot(c, e)) for sid, e in embs.items()}
        keep = [embs[s] for s, v in sims.items() if v >= np.median(list(sims.values()))]
        c = np.mean(keep, axis=0)
        c /= np.linalg.norm(c)
        sims = {sid: float(np.dot(c, e)) for sid, e in embs.items()}
        min_long, min_short = 0.75, 0.60  # ear-validated centroid bars (2026-08)

    violations = []
    for sid, v in sims.items():
        d = durs.get(sid, 0)
        if d < 1.0:
            continue
        thresh = min_long if d >= 2.5 else min_short
        if v < thresh:
            violations.append(Violation(
                message=f"voice drift: similarity {v:.2f} to {anchor} ({d:.1f}s take, bar {thresh})",
                unit_id=sid, fixable=True))
    details = {"takes": len(sims), "anchor": anchor,
               "median": round(float(np.median(list(sims.values()))), 3)}
    if violations:
        return CheckResult.failed("speaker_similarity", 5, violations,
                                  total=len(violations), **details)
    return CheckResult.passed("speaker_similarity", 5, **details)


@check(stage=5, dimension="cache_integrity")
def cache_integrity(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    from pipeline.s5_render import segment_hash

    texts = _segment_texts(art)
    manifest = art.render_manifest
    violations = []
    for r in manifest["segments"]:
        expected = segment_hash(r["engine"], r["params"], texts.get(r["segment_id"], ""))
        if r["hash"] != expected:
            violations.append(Violation(
                message=f"stale cache: manifest hash {r['hash'][:10]} != {expected[:10]} "
                        f"(text or params changed without re-render)",
                unit_id=r["segment_id"], fixable=True))
    if violations:
        return CheckResult.failed("cache_integrity", 5, violations[:20], total=len(violations))
    return CheckResult.passed("cache_integrity", 5)
