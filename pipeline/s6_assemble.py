"""Stage 6: assemble the M4B.

Inserts literal silence around each segment per the pause policy (gap between
segments = prev.pause_after + next.pause_before), concatenates with the ffmpeg
concat demuxer, masters with two-pass loudnorm, and writes chapter markers
from the chapter tree with proper FFMETADATA escaping.

Artifacts: 06_book.m4b + 06_assemble.json
"""
from __future__ import annotations

import json
import subprocess
import tempfile

import numpy as np
from pathlib import Path

from evals.contracts import ARTIFACT_FILES, BookCtx
from pipeline.config import (AAC_BITRATE, CFG, LOUDNESS_TARGET_LUFS, OUTPUT_DIR,
                             PAUSE_POLICY, SAMPLE_RATE, TRUE_PEAK_MAX_DBTP)
from pipeline.ir import assembly_view, gap_between

HEAD_TONE_S = float(CFG["mastering"].get("head_tone_s", 0.75))
TAIL_TONE_S = float(CFG["mastering"].get("tail_tone_s", 2.0))


def _ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(["ffmpeg", "-y", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-2000:]}")
    return result


def _escape_meta(value: str) -> str:
    out = []
    for c in value:
        if c in "=;#\\\n":
            out.append("\\" + ("n" if c == "\n" else c))
        else:
            out.append(c)
    return "".join(out)


# Encode with headroom below the check ceiling: AAC encoding overshoots peaks.
_ENCODE_TP = TRUE_PEAK_MAX_DBTP - 1.0


def _measure_loudnorm(path: Path) -> dict:
    result = subprocess.run(
        ["ffmpeg", "-i", str(path), "-af",
         f"loudnorm=I={LOUDNESS_TARGET_LUFS}:TP={_ENCODE_TP}:LRA=11:print_format=json",
         "-f", "null", "-"], capture_output=True, text=True)
    stderr = result.stderr
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    return json.loads(stderr[start:end + 1])


def run(book: BookCtx) -> Path:
    narration = json.loads((book.artifacts_dir / ARTIFACT_FILES["narration"]).read_text(encoding="utf-8"))
    manifest = json.loads((book.artifacts_dir / ARTIFACT_FILES["render_manifest"]).read_text(encoding="utf-8"))
    chapters_doc = json.loads((book.artifacts_dir / ARTIFACT_FILES["chapters"]).read_text(encoding="utf-8"))
    rows = {r["segment_id"]: r for r in manifest["segments"]}
    chapter_titles = {c["id"]: c["title"] for c in chapters_doc["chapters"]}

    segments = assembly_view(narration["segments"], CFG["narration"].get("headings", {}))
    timeline = []
    # ACX: room tone at the head of the program; gaps derived LIVE from the
    # pause policy config (pause changes = reassembly only, never re-render).
    cursor = HEAD_TONE_S
    chapter_starts: dict[str, float] = {}
    # (start_s, take_path) placements onto the continuous tone bed — book-wide
    # and per chapter. Overlay, not concat: timing below is unchanged.
    placements: list[tuple[float, Path]] = []
    chapter_placements: dict[str, list[tuple[float, Path]]] = {}

    prev_seg: dict | None = None
    for seg in segments:
        row = rows[seg["id"]]
        gap = gap_between(prev_seg, seg, PAUSE_POLICY)
        # A chapter owns its leading pause: the marker sits at the gap start.
        chapter_starts.setdefault(seg["chapter_id"], cursor)
        start = cursor + gap
        take_path = book.artifacts_dir / row["wav"]
        placements.append((start, take_path))
        chapter_placements.setdefault(seg["chapter_id"], []).append((start, take_path))
        timeline.append({
            "segment_id": seg["id"], "chapter_id": seg["chapter_id"], "type": seg["type"],
            "start_s": round(start, 3), "duration_s": row["duration_s"],
            "silence_before_s": round(gap, 3),
        })
        cursor = start + row["duration_s"]
        prev_seg = seg
    cursor += TAIL_TONE_S
    total_s = cursor

    out_dir = OUTPUT_DIR / book.book_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_m4b = out_dir / "book.m4b"

    with tempfile.TemporaryDirectory(prefix="pdf2audiobook_asm_") as td:
        td = Path(td)
        ref = manifest.get("params", {}).get("reference_wav")

        # Overlay normalized+faded takes onto one continuous tone bed. Fixed
        # offsets = zero duration change = timeline/markers exact.
        import soundfile as sf

        from pipeline.mastering import build_program

        program = build_program(placements, int(round(total_s * SAMPLE_RATE)), ref)
        combined = td / "combined.wav"
        sf.write(str(combined), program, SAMPLE_RATE)

        measured = _measure_loudnorm(combined)

        # Chapter metadata (FFMETADATA, escaped).
        meta_lines = [";FFMETADATA1"]
        ordered = sorted(chapter_starts.items(), key=lambda kv: kv[1])
        chapter_markers = []
        for i, (ch_id, start_s) in enumerate(ordered):
            end_s = ordered[i + 1][1] if i + 1 < len(ordered) else total_s
            title = chapter_titles.get(ch_id, ch_id)
            chapter_markers.append({"id": ch_id, "title": title,
                                    "start_ms": int(start_s * 1000), "end_ms": int(end_s * 1000)})
            meta_lines += ["[CHAPTER]", "TIMEBASE=1/1000",
                           f"START={int(start_s * 1000)}", f"END={int(end_s * 1000)}",
                           f"title={_escape_meta(title)}"]
        meta_file = td / "chapters.txt"
        meta_file.write_text("\n".join(meta_lines))

        ln = (f"loudnorm=I={LOUDNESS_TARGET_LUFS}:TP={_ENCODE_TP}:LRA=11:"
              f"measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
              f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
              f"offset={measured['target_offset']}:linear=true,"
              # loudnorm upsamples to 192 kHz internally; resample BEFORE the
              # limiter so the ceiling holds at the delivery rate.
              f"aresample={SAMPLE_RATE},"
              f"alimiter=limit=0.63:level=false")  # ~-4 dBFS ceiling; AAC adds <1 dB
        args = ["-i", str(combined), "-i", str(meta_file), "-map_metadata", "1",
                "-af", ln, "-c:a", "aac", "-b:a", AAC_BITRATE]
        title = book.config.get("title")
        author = book.config.get("author")
        if title:
            args += ["-metadata", f"title={title}"]
        if author:
            args += ["-metadata", f"artist={author}", "-metadata", f"album_artist={author}"]
        args += ["-f", "mp4", str(out_m4b)]
        print("  ffmpeg " + " ".join(args), flush=True)
        _ffmpeg(args)

        # One file per chapter, mastered with the SAME global loudnorm values
        # so loudness stays consistent across files.
        chapters_dir = out_dir / "chapters"
        chapters_dir.mkdir(exist_ok=True)
        for old in chapters_dir.glob("*.m4a"):
            old.unlink()
        chapter_files = []
        ordered_ids = [ch_id for ch_id, _ in ordered]
        for idx, ch_id in enumerate(ordered_ids, 1):
            title = chapter_titles.get(ch_id, ch_id)
            safe = "".join(c if c.isalnum() or c in " -" else "" for c in title).strip()
            safe = "_".join(safe.split())[:60] or ch_id
            ch_out = chapters_dir / f"{idx:02d}_{safe}.m4a"
            # Chapter-local placements: shift so the first take sits at
            # HEAD_TONE_S; chapter length = last take end + TAIL_TONE_S.
            ch_places = chapter_placements.get(ch_id, [])
            if not ch_places:
                continue
            shift = ch_places[0][0] - HEAD_TONE_S
            local = [(s - shift, p) for s, p in ch_places]
            last_dur = next(t["duration_s"] for t in timeline
                            if t["chapter_id"] == ch_id
                            and abs(t["start_s"] - ch_places[-1][0]) < 0.01)
            ch_total = local[-1][0] + last_dur + TAIL_TONE_S
            ch_program = build_program(local, int(round(ch_total * SAMPLE_RATE)), ref)
            ch_wav = td / f"chapter_{ch_id}.wav"
            sf.write(str(ch_wav), ch_program, SAMPLE_RATE)
            ch_args = ["-i", str(ch_wav), "-af", ln, "-c:a", "aac", "-b:a", AAC_BITRATE,
                       "-metadata", f"title={title}",
                       "-metadata", f"track={idx}/{len(ordered_ids)}"]
            if book.config.get("title"):
                ch_args += ["-metadata", f"album={book.config['title']}"]
            if book.config.get("author"):
                ch_args += ["-metadata", f"artist={book.config['author']}"]
            ch_args += ["-f", "mp4", str(ch_out)]
            _ffmpeg(ch_args)
            chapter_files.append({"id": ch_id, "title": title, "track": idx,
                                  "file": str(ch_out.relative_to(out_dir))})

    final = _measure_loudnorm(out_m4b)
    artifact = {
        "total_s": round(total_s, 3),
        "timeline": timeline,
        "chapters": chapter_markers,
        "chapter_files": chapter_files,
        "loudnorm": {"measured_input": measured, "final": final},
        "bitrate": AAC_BITRATE,
    }
    (book.artifacts_dir / ARTIFACT_FILES["assemble"]).write_text(json.dumps(artifact, indent=1))

    # Typed manifest: the contract a frontend consumes.
    from evals.audioutil import container_duration
    from pipeline.manifest import AudiobookManifest, ChapterFile

    render = json.loads((book.artifacts_dir / ARTIFACT_FILES["render_manifest"]).read_text())
    manifest = AudiobookManifest(
        book_id=book.book_id,
        title=book.config.get("title"),
        author=book.config.get("author"),
        source_pdf=str(book.pdf_path),
        engine=render["engine"],
        voice=str(render["params"].get("voice") or render["params"].get("reference_wav") or ""),
        sample_rate=SAMPLE_RATE,
        bitrate=AAC_BITRATE,
        duration_s=round(total_s, 1),
        lufs=float(final["input_i"]),
        m4b="book.m4b",
        chapters=[ChapterFile(**cf, duration_s=round(container_duration(out_dir / cf["file"]), 1))
                  for cf in chapter_files],
    )
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return out_m4b
