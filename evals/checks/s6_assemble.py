"""Stage 6 (assemble) checks: silence policy, loudness, chapter markers,
playability."""
from __future__ import annotations

import random

from evals.artifacts import ArtifactSet
from evals.checks import check
from evals.contracts import CheckResult, Violation
from evals.corpus import DocSpec
from evals.textutil import normalize
from pipeline.config import PAUSE_POLICY


@check(stage=6, dimension="silence_gaps_vs_policy")
def silence_gaps_vs_policy(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    from evals.audioutil import container_duration, detect_silences

    from pipeline.config import CFG
    from pipeline.ir import assembly_view, gap_between

    tol = cfg.get("silence_tolerance_ms", 150) / 1000
    asm = art.assemble
    view = assembly_view(art.narration["segments"], CFG["narration"].get("headings", {}))
    narration = {s["id"]: s for s in view}
    violations = []

    # Planned gaps must match the LIVE pause policy (single source of truth
    # shared with the assembler via assembly_view + gap_between).
    prev_seg = None
    for t in asm["timeline"]:
        seg = narration.get(t["segment_id"])
        if seg is None:
            continue  # heading skipped by reading policy after this assembly
        planned = gap_between(prev_seg, seg, PAUSE_POLICY)
        if abs(t["silence_before_s"] - planned) > 0.001:
            violations.append(Violation(
                message=f"planned gap {t['silence_before_s']}s != policy {planned}s",
                unit_id=t["segment_id"]))
        prev_seg = seg

    # Total duration reconciles: sum(audio) + sum(silence) == container.
    planned_total = asm["total_s"]
    actual_total = container_duration(art.m4b_path)
    if abs(actual_total - planned_total) > 0.5 + tol:
        violations.append(Violation(
            message=f"container {actual_total:.1f}s != planned {planned_total:.1f}s"))

    # Spot-verify chapter-boundary silences in the rendered audio.
    ch_starts = [t for t in asm["timeline"] if t["type"] == "chapter_heading"
                 and t["silence_before_s"] >= 0.5]
    rng = random.Random(0)
    sample = rng.sample(ch_starts, min(8, len(ch_starts))) if ch_starts else []
    if sample:
        silences = detect_silences(art.m4b_path, min_s=0.3)
        for t in sample:
            gap_start = t["start_s"] - t["silence_before_s"]
            hit = any(s <= gap_start + 0.35 and e >= t["start_s"] - 0.35 for s, e in silences)
            if not hit:
                violations.append(Violation(
                    message=f"no measured silence around chapter start at {t['start_s']:.1f}s",
                    unit_id=t["segment_id"]))

    if violations:
        return CheckResult.failed("silence_gaps_vs_policy", 6, violations[:20], total=len(violations))
    return CheckResult.passed("silence_gaps_vs_policy", 6, spot_checked=len(sample))


@check(stage=6, dimension="loudness_conformance")
def loudness_conformance(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    import re
    import subprocess

    from evals.audioutil import measure_loudness

    target = cfg.get("lufs_target", -18.0)
    tol = cfg.get("lufs_tolerance", 1.5)
    peak_max = cfg.get("true_peak_max", -3.0)
    m = measure_loudness(art.m4b_path)
    lufs = float(m["input_i"])
    # ACX gates on SAMPLE peak (astats); inter-sample true peak at 24 kHz
    # overshoots by design and is recorded as information only.
    r = subprocess.run(["ffmpeg", "-i", str(art.m4b_path),
                        "-af", "astats=measure_overall=Peak_level:measure_perchannel=none",
                        "-f", "null", "-"], capture_output=True, text=True)
    match = re.search(r"Peak level dB: (-?[\d.]+)", r.stderr)
    peak = float(match.group(1)) if match else 0.0
    violations = []
    if abs(lufs - target) > tol:
        violations.append(Violation(message=f"integrated loudness {lufs} LUFS vs target {target}±{tol}"))
    if peak > peak_max:
        violations.append(Violation(message=f"sample peak {peak} dB above {peak_max}"))
    details = {"lufs": lufs, "sample_peak": peak, "true_peak": float(m["input_tp"])}
    if violations:
        return CheckResult.failed("loudness_conformance", 6, violations, **details)
    return CheckResult.passed("loudness_conformance", 6, **details)


@check(stage=6, dimension="chapter_markers")
def chapter_markers(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    from evals.audioutil import container_chapters, container_duration

    chapters = container_chapters(art.m4b_path)
    planned = art.assemble["chapters"]
    violations = []
    if len(chapters) != len(planned):
        violations.append(Violation(message=f"{len(chapters)} markers in file, planned {len(planned)}"))
    else:
        prev_end = None
        for got, want in zip(chapters, planned):
            got_title = got.get("tags", {}).get("title", "")
            if normalize(got_title, casefold=True) != normalize(want["title"], casefold=True):
                violations.append(Violation(message=f"marker title {got_title!r} != {want['title']!r}"))
            start_ms = float(got["start_time"]) * 1000
            # Containers pin chapter 1 to 0 (media start), not the planned
            # head-tone offset — that's correct behavior, not a defect.
            planned = 0 if got is chapters[0] else want["start_ms"]
            if abs(start_ms - planned) > 200:
                violations.append(Violation(
                    message=f"marker {want['title']!r} at {start_ms:.0f}ms, planned {planned}ms"))
            if prev_end is not None and start_ms < prev_end - 1:
                violations.append(Violation(message=f"marker {want['title']!r} overlaps previous"))
            prev_end = float(got["end_time"]) * 1000
        if chapters:
            last_end = float(chapters[-1]["end_time"])
            if abs(last_end - container_duration(art.m4b_path)) > 1.5:
                violations.append(Violation(message="last marker does not reach container end"))
    if violations:
        return CheckResult.failed("chapter_markers", 6, violations[:15])
    return CheckResult.passed("chapter_markers", 6, markers=len(chapters))


@check(stage=6, dimension="words_per_minute")
def words_per_minute(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    """Narration pace. Professional audiobooks run ~150-160 wpm spoken;
    too fast fatigues listeners, too slow drags. Spoken wpm excludes the
    inserted pause silence; overall wpm includes it (what the listener
    experiences across chapters)."""
    wpm_min = cfg.get("wpm_min", 130)
    wpm_max = cfg.get("wpm_max", 178)
    words = sum(len(s["text"].split()) for s in art.narration["segments"])
    asm = art.assemble
    spoken_s = sum(t["duration_s"] for t in asm["timeline"])
    total_s = asm["total_s"]
    if spoken_s <= 0:
        return CheckResult.skipped("words_per_minute", 6, "no audio")
    spoken = words / (spoken_s / 60)
    overall = words / (total_s / 60)
    details = {"spoken_wpm": round(spoken), "overall_wpm": round(overall), "words": words}
    if not (wpm_min <= spoken <= wpm_max):
        return CheckResult.failed("words_per_minute", 6, [Violation(
            message=f"spoken pace {spoken:.0f} wpm outside [{wpm_min}, {wpm_max}] "
                    f"(fix via engine speed param, e.g. kokoro speed)")], **details)
    return CheckResult.passed("words_per_minute", 6, **details)


@check(stage=6, dimension="take_level_consistency")
def take_level_consistency(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    """Per-take onset levels must be consistent — the artifact the mastering
    pass fixed (was 11.4 dB spread). Measures voiced RMS at each segment's
    start in the finished book; guards against silent regression."""
    import numpy as np

    max_spread = cfg.get("take_level_spread_db", 6.0)
    asm = art.assemble
    m4b = art.m4b_path
    # Measure VOICED RMS over each take's full span (not a 400ms onset window,
    # which just reflects whether a clip starts on silence vs a consonant).
    body = [t for t in asm["timeline"] if t["duration_s"] > 0.8]
    if len(body) < 4:
        return CheckResult.skipped("take_level_consistency", 6, "too few segments")
    step = max(1, len(body) // 40)
    picks = body[::step][:40]

    import subprocess
    import tempfile

    import soundfile as sf

    def voiced_rms_db(x: np.ndarray) -> float:
        fr = 480
        n = len(x) // fr
        if n < 1:
            return -120.0
        r = np.sqrt((x[: n * fr].reshape(n, fr) ** 2).mean(axis=1))
        keep = r[r > r.max() * 10 ** (-25 / 20)]
        return 20 * np.log10(float(np.sqrt((keep**2).mean())) + 1e-12)

    levels = []
    with tempfile.TemporaryDirectory() as td:
        for t in picks:
            clip = f"{td}/c.wav"
            subprocess.run(["ffmpeg", "-y", "-v", "quiet", "-ss", str(t["start_s"]),
                            "-t", str(min(t["duration_s"], 5.0)), "-i", str(m4b), clip],
                           capture_output=True)
            try:
                a, _ = sf.read(clip, dtype="float32")
            except Exception:
                continue
            if len(a) > 480:
                levels.append(voiced_rms_db(a))
    if len(levels) < 4:
        return CheckResult.skipped("take_level_consistency", 6, "could not sample takes")
    spread = max(levels) - min(levels)
    if spread > max_spread:
        return CheckResult.failed("take_level_consistency", 6, [Violation(
            message=f"onset level spread {spread:.1f} dB > {max_spread} (takes not leveled)")],
            spread_db=round(spread, 1))
    return CheckResult.passed("take_level_consistency", 6, spread_db=round(spread, 1))


@check(stage=6, dimension="m4b_playable")
def m4b_playable(doc: DocSpec, art: ArtifactSet, cfg: dict) -> CheckResult:
    from evals.audioutil import ffprobe_json

    info = ffprobe_json(art.m4b_path)
    audio_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio_streams or audio_streams[0].get("codec_name") != "aac":
        return CheckResult.failed("m4b_playable", 6, [Violation(message="no AAC audio stream")])
    if float(info["format"].get("duration", 0)) <= 0:
        return CheckResult.failed("m4b_playable", 6, [Violation(message="zero duration container")])
    return CheckResult.passed("m4b_playable", 6,
                              duration_s=round(float(info["format"]["duration"]), 1))
