"""Export to a folder (NAS share, synced drive, external disk).

Copies the whole output bundle: book.m4b, chapters/, manifest.json — the
layout any frontend or server consumes directly.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from rich.console import Console

from export.books import resolve_m4b

console = Console()

NAME = "folder"


def publish(book: str, dest: str | None = None, **opts) -> bool:
    if not dest:
        console.print("[red]folder export needs --dest PATH[/red]")
        return False
    m4b = resolve_m4b(book)
    if m4b is None:
        console.print(f"[red]No finished audiobook found for:[/red] {book}")
        return False
    src_dir = m4b.parent
    target = Path(dest).expanduser() / src_dir.name
    target.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.is_dir():
            shutil.copytree(item, target / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target / item.name)
    console.print(f"[green]Exported:[/green] {target}")
    return True
