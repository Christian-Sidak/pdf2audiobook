#!/usr/bin/env python3
"""PDF to audiobook: eval-gated local pipeline.

    python main.py build book.pdf                # full QC'd pipeline -> M4B
    python main.py preview book.pdf              # stages 1-3 + chapter table
    python main.py evals --stages 1-3            # eval matrix over the corpus
    python main.py add-to-music artifacts/<id>/06_book.m4b
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Convert PDFs to audiobooks with a QC'd, eval-gated local pipeline."""


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--stages", default="1-6", help="stages to run (default 1-6)")
@click.option("--engine", type=click.Choice(["kokoro", "qwen3tts"]), default=None,
              help="TTS engine override (kokoro = fast draft, qwen3tts = cloned final)")
@click.option("--voice", default=None,
              help="kokoro: preset voice name (e.g. am_michael); qwen3tts: reference WAV to clone")
@click.option("--chapters", default=None,
              help="narrate only these chapters, e.g. '1' or 'ch01,ch03' (default: all body chapters)")
@click.option("--title", default=None, help="book title for metadata (default: from filename)")
@click.option("--author", default=None, help="author for metadata")
@click.option("--tts-model", default=None,
              help="qwen3tts checkpoint: 'fast' (0.6B), 'full' (1.7B), or an explicit HF id")
@click.option("--fast", is_flag=True,
              help="fast non-cloned narration (kokoro preset voice, ~20x faster than cloning)")
@click.option("--target-wpm", type=int, default=None,
              help="pacing target in words/minute (default from config; e.g. 120 for gravitas)")
@click.option("--keep-going", is_flag=True, help="continue past failed stages")
@click.option("--to-books", is_flag=True, help="send the finished M4B to Apple Books")
def build(pdf_path: str, stages: str, engine: str | None, voice: str | None,
          chapters: str | None, title: str | None, author: str | None,
          tts_model: str | None, fast: bool, target_wpm: int | None,
          keep_going: bool, to_books: bool):
    """Run the full pipeline with QC gates: extract, clean, chapterize,
    narrate (local Qwen), render (TTS), assemble M4B."""
    from evals.qc_loop import qc_run
    from evals.run import parse_stages
    from pipeline.config import OUTPUT_DIR, book_id_for

    book_id = book_id_for(pdf_path)
    config = {}
    if fast:
        engine = "kokoro"
        if voice and not voice.endswith(".wav"):
            pass  # kokoro preset name is valid with --fast
        else:
            voice = None  # reference WAVs are clone-only
    if engine:
        config["tts_engine"] = engine
    if voice:
        config["voice"] = voice
    if chapters:
        config["only_chapters"] = [c if c.startswith("ch") else f"ch{int(c):02d}"
                                   for c in chapters.split(",")]
    if target_wpm:
        config["target_wpm"] = target_wpm
    config["title"] = title or Path(pdf_path).stem.replace("_", " ").replace("-", " ").title()
    if author:
        config["author"] = author
    if tts_model:
        from pipeline.config import TTS_ENGINES
        aliases = {"fast": TTS_ENGINES["qwen3tts"].get("fast_model_id"),
                   "full": TTS_ENGINES["qwen3tts"]["model_id"]}
        config["tts_model"] = aliases.get(tts_model, tts_model)

    result = qc_run(book_id, parse_stages(stages), keep_going=keep_going, pdf=pdf_path,
                    config=config)
    if result.halted_at:
        console.print(f"[red]halted at stage {result.halted_at}; "
                      f"run 'python -m evals.review list'[/red]")
        sys.exit(1)

    m4b = OUTPUT_DIR / book_id / "book.m4b"
    if m4b.exists():
        console.print(f"[green bold]Audiobook ready:[/green bold] {OUTPUT_DIR / book_id}")
        if to_books:
            from export.books import publish as send
            send(book_id)


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
def preview(pdf_path: str):
    """Extract, clean, and chapterize only; show the chapter table."""
    import json

    from evals.contracts import ARTIFACT_FILES, BookCtx
    from pipeline.config import book_id_for
    from pipeline.run_stages import run_stages

    book = BookCtx.for_book(book_id_for(pdf_path), pdf_path)
    run_stages(book, {1, 2, 3})

    chapters = json.loads((book.artifacts_dir / ARTIFACT_FILES["chapters"]).read_text())
    table = Table(title=f"Chapters: {Path(pdf_path).name}")
    table.add_column("#")
    table.add_column("title")
    table.add_column("matter")
    table.add_column("pages")
    table.add_column("words", justify="right")
    for c in chapters["chapters"]:
        table.add_row(c["id"], c["title"][:60], c["matter"],
                      f"{c['start_page'] + 1}-{c['end_page'] + 1}",
                      f"{len(c['text'].split()):,}")
    console.print(table)


@cli.command(context_settings={"ignore_unknown_options": True})
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def evals(args):
    """Run the eval matrix (passes arguments through to evals.run)."""
    from evals.run import main as evals_main

    sys.exit(evals_main(list(args)))


@cli.command()
@click.argument("name")
@click.argument("clip", type=click.Path(exists=True))
@click.option("--tone-clip", type=click.Path(exists=True), default=None,
              help="same-session recording containing clean non-speech air (fallback: harvested from CLIP)")
@click.option("--start", default=None, help="trim start, seconds (e.g. 4.3)")
@click.option("--end", default=None, help="trim end, seconds")
def voice_add(name: str, clip: str, tone_clip: str | None, start: str | None, end: str | None):
    """Bank a narrator voice with validation: sentence-clean reference plus a
    SAME-SESSION room-tone sample. Warns loudly when the session has no
    harvestable clean air (the tone must match the takes' floor; it cannot be
    synthesized or borrowed afterward)."""
    import subprocess

    import numpy as np
    import soundfile as sf

    from pipeline.config import ROOT, SAMPLE_RATE
    from pipeline.roomtone import _vad_nonspeech_pool

    ref = ROOT / "voices" / f"{name}.wav"
    trim = ([] if start is None else ["-ss", start]) + ([] if end is None else ["-to", end])
    subprocess.run(["ffmpeg", "-y", "-v", "quiet", *trim, "-i", clip,
                    "-ac", "1", "-ar", str(SAMPLE_RATE), str(ref)], check=True)
    dur = sf.info(str(ref)).duration
    if not 3 <= dur <= 60:
        console.print(f"[yellow]reference is {dur:.1f}s; 10-30s is ideal[/yellow]")

    tone_src = tone_clip or str(ref)
    sidecar = ref.with_suffix(".roomtone.wav")
    if tone_clip:
        subprocess.run(["ffmpeg", "-y", "-v", "quiet", "-i", tone_clip,
                        "-ac", "1", "-ar", str(SAMPLE_RATE), str(sidecar)], check=True)
        tone_src = str(sidecar)
    pool, _sr = _vad_nonspeech_pool(tone_src, 500)
    seconds_of_air = len(pool) * 0.06
    if seconds_of_air < 1.5:
        console.print(f"[red bold]WARNING: only {seconds_of_air:.1f}s of clean non-speech air "
                      f"in this session[/red bold] — pauses will not match the voice. "
                      f"Provide --tone-clip from the same recording with a silent stretch.")
    else:
        if not tone_clip:
            import shutil
            shutil.copy(ref, sidecar)
        levels = 20 * np.log10(np.sqrt((pool**2).mean()) + 1e-12)
        console.print(f"[green]banked {name}[/green]: {dur:.1f}s reference, "
                      f"{seconds_of_air:.1f}s same-session air at {levels:.0f} dBFS")


@cli.command()
@click.argument("book")
def reassemble(book: str):
    """Re-run assembly only (stage 6): iterate pause lengths and room tone
    from config.yaml without re-rendering a single take."""
    import json

    from evals.contracts import BookCtx
    from pipeline import s6_assemble
    from pipeline.config import ARTIFACTS_DIR

    from pipeline.config import OUTPUT_DIR

    art_dir = ARTIFACTS_DIR / book
    if not (art_dir / "05_render/manifest.json").exists():
        console.print(f"[red]no render manifest for {book}; run build first[/red]")
        sys.exit(1)
    source_pdf = json.loads((art_dir / "01_extract.json").read_text()).get("pdf", "")
    config = {}
    prior = OUTPUT_DIR / book / "manifest.json"
    if prior.exists():  # keep title/author across reassemblies
        m = json.loads(prior.read_text())
        config = {k: v for k, v in (("title", m.get("title")), ("author", m.get("author"))) if v}
    ctx = BookCtx.for_book(book, source_pdf, config=config)
    out = s6_assemble.run(ctx)
    console.print(f"[green bold]Reassembled:[/green bold] {out.parent}")


@cli.command("voice-grab")
@click.argument("name")
@click.argument("source")
@click.option("--start", default=None, help="clip start (MM:SS or seconds)")
@click.option("--end", default=None, help="clip end (MM:SS or seconds)")
@click.option("--seconds", "ref_seconds", default=25.0, help="target reference length")
def voice_grab(name: str, source: str, start, end, ref_seconds):
    """Bank a voice from a YouTube URL or local file: auto-extracts a clean
    sentence-bounded reference AND a same-session room-tone sample.

    A longer reference (~25-30s) gives a stronger voiceprint than a short
    clip. Give --start/--end to target a specific stretch of a long video.
    """
    from pipeline.voice_intake import grab

    sys.exit(0 if grab(name, source, start, end, ref_seconds) else 1)


@cli.command("voice-distill")
@click.argument("name")
@click.option("--text", default=None,
              help="narration text to synthesize (default: a public-domain passage)")
@click.option("--candidates", "n", default=6, help="takes to generate and QC")
def voice_distill(name: str, text, n: int):
    """Replace a banked real-audio reference with a cleaner GENERATED one.

    Synthesizes candidate takes with the fresh clone, QCs each (pitch/onset
    audit, whisper transcript, speaker similarity), snips hallucinated
    preambles, and banks the winner. The real audio is kept as <name>.orig.wav.
    A distilled reference has no room noise or interviewer bleed and provably
    ends on a complete sentence.
    """
    from pipeline.voice_intake import distill_reference

    sys.exit(0 if distill_reference(name, text, n) else 1)


@cli.command("repair-splits")
@click.argument("book")
@click.option("--dry-run", is_flag=True, help="report joins without writing the narration")
def repair_splits_cmd(book: str, dry_run: bool):
    """Rejoin wrongly split sentences (middle initials, abbreviations) in an
    existing narration script, judged by the local LLM. Then re-run stages
    5-6: only the joined segments re-render."""
    import json as _json

    from evals.contracts import ARTIFACT_FILES, BookCtx
    from pipeline.config import ARTIFACTS_DIR, REWRITE_MODEL
    from pipeline.s4_narration import adjudicate_splits, repair_splits

    art_dir = ARTIFACTS_DIR / book
    if not (art_dir / ARTIFACT_FILES["narration"]).exists():
        console.print(f"[red]no narration for {book}; run stages 1-4 first[/red]")
        sys.exit(1)
    source_pdf = _json.loads((art_dir / "01_extract.json").read_text()).get("pdf", "")
    ctx = BookCtx.for_book(book, source_pdf)
    if dry_run:
        path = ctx.artifacts_dir / ARTIFACT_FILES["narration"]
        segs = _json.loads(path.read_text(encoding="utf-8"))["segments"]
        out = adjudicate_splits(segs, REWRITE_MODEL)
        joined = {s["text"] for s in out} - {s["text"] for s in segs}
        for t in sorted(joined)[:40]:
            console.print(f"  [green]JOIN[/green] {t[:140]}")
        console.print(f"[cyan]dry run:[/cyan] {len(segs)} -> {len(out)} segments")
        return
    n = repair_splits(ctx)
    console.print(f"[green]joined {n} boundaries[/green]; re-run --stages 5-6 to render them")


@cli.command()
@click.argument("book")
@click.argument("image", type=click.Path(exists=True))
def cover(book: str, image: str):
    """Embed cover art into a finished audiobook's M4B and chapter files."""
    from export.cover import embed_cover

    sys.exit(0 if embed_cover(book, image) else 1)


@cli.command()
@click.argument("book")
@click.option("--to", "target", default="books",
              help="export target: books (Apple Books), folder, audiobookshelf")
@click.option("--dest", default=None, help="destination path for folder-style targets")
def publish(book: str, target: str, dest: str | None):
    """Publish a finished audiobook (book id from output/, or an .m4b path)."""
    import export as export_pkg

    mod = export_pkg.get(target)
    sys.exit(0 if mod.publish(book, dest=dest) else 1)


@cli.command()
@click.argument("book", required=False)
@click.option("--watch", is_flag=True, help="refresh every 15s until interrupted")
def status(book: str | None, watch: bool):
    """Colorful live progress for running/finished builds (all books, or one)."""
    import json
    import subprocess
    import time as _time
    from datetime import datetime, timedelta

    from rich.panel import Panel

    art_root = Path("artifacts")

    def _bar(frac: float, width: int = 34) -> str:
        full = int(frac * width)
        color = "green" if frac >= 0.999 else ("yellow" if frac < 0.35 else "cyan")
        return f"[{color}]{'█' * full}[/][grey35]{'░' * (width - full)}[/]"

    def _one(bdir: Path):
        narr = bdir / "04_narration.json"
        segdir = bdir / "05_render" / "segments"
        stages = [p.name[:2] for p in sorted(bdir.glob("0*"))]
        title = bdir.name.replace("_", " ").title()
        if not narr.exists():
            done = ", ".join(sorted(set(stages))) or "nothing yet"
            return Panel(f"[dim]stages on disk:[/] {done}\n[italic]waiting on narration "
                         f"(stage 4) before rendering can start", title=f"[bold]{title}[/]",
                         border_style="grey50")
        total = len(json.load(open(narr))["segments"])
        wavs = list(segdir.glob("*.wav")) if segdir.exists() else []
        n = len(wavs)
        frac = min(1.0, n / max(1, total))
        now = _time.time()
        recent = [w for w in wavs if now - w.stat().st_mtime < 1800]
        rate = len(recent) * 2  # per hour, from a 30-min window
        alive = subprocess.run(["pgrep", "-f", "main.py build"], capture_output=True).returncode == 0
        m4b = list(Path("output").glob(f"{bdir.name}/*.m4b")) if Path("output").exists() else []
        if m4b:
            state, style = f"[bold green]✓ published[/] → {m4b[0]}", "green"
            detail = ""
        elif frac >= 0.999:
            state, style = "[bold green]✓ all takes rendered[/] — awaiting assembly & mastering", "green"
            detail = ""
        elif alive and rate:
            eta = (total - n) / rate
            fin = (datetime.now() + timedelta(hours=eta)).strftime("%a %-I:%M %p")
            state = f"[bold cyan]rendering[/] with the eval-gated TTS loop"
            style = "cyan"
            detail = f"[dim]pace[/] {rate}/hr   [dim]remaining[/] {total - n:,}   [dim]ETA[/] [bold]{fin}[/] (~{eta:.1f}h)"
        elif alive:
            state, style = "[bold yellow]build process alive[/] — between takes (stage 4 retries or model load)", "yellow"
            detail = ""
        else:
            state, style = "[bold red]✗ no build running[/] — resume with the same build command (takes are cached)", "red"
            detail = ""
        body = (f"{_bar(frac)}  [bold]{n:,}[/]/[bold]{total:,}[/] takes ({frac:5.1%})\n"
                f"{state}" + (f"\n{detail}" if detail else ""))
        return Panel(body, title=f"[bold]{title}[/]", border_style=style)

    def _render_all():
        books = [d for d in sorted(art_root.iterdir()) if d.is_dir()]
        if book:
            books = [d for d in books if book.lower().replace(" ", "_") in d.name]
        for b in books:
            console.print(_one(b))

    if watch:
        try:
            while True:
                console.clear()
                console.print(f"[dim]{datetime.now():%H:%M:%S} — refreshes every 15s, "
                              f"ctrl-c to exit[/]")
                _render_all()
                _time.sleep(15)
        except KeyboardInterrupt:
            pass
    else:
        _render_all()


if __name__ == "__main__":
    cli()
