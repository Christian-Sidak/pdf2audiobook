"""Publish a finished audiobook into Apple Books.

Books (not Music) owns .m4b on modern macOS: it files the book as an
audiobook with resume-where-you-left-off, chapter navigation from our
markers, speed control, and sleep timer.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from rich.console import Console

from pipeline.config import OUTPUT_DIR

console = Console()


def resolve_m4b(book: str) -> Path | None:
    """Accept a book id from output/, or a direct path to an .m4b."""
    p = Path(book)
    if p.suffix.lower() == ".m4b" and p.exists():
        return p
    candidate = OUTPUT_DIR / book / "book.m4b"
    return candidate if candidate.exists() else None


NAME = "books"


def publish(book: str, **opts) -> bool:
    m4b = resolve_m4b(book)
    if m4b is None:
        console.print(f"[red]No finished audiobook found for:[/red] {book}")
        return False
    result = subprocess.run(["open", "-a", "Books", str(m4b)], capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]Books import failed:[/red] {result.stderr.strip()}")
        return False
    console.print(f"[green]Sent to Apple Books:[/green] {m4b.name} "
                  f"[dim](Books copies it into its own library)[/dim]")
    return True
