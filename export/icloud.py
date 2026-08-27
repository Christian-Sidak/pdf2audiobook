"""Export to iCloud Drive so the phone's Files app can hand the book to
Apple Books (Files -> tap to download -> Share -> Books).

Copies just the M4B, named from the book's display title, into
iCloud Drive/Audiobooks/. Non-store audiobooks don't sync via Books'
own iCloud, so this folder is the wireless bridge to iOS.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from rich.console import Console

from export.books import resolve_m4b

console = Console()

NAME = "icloud"

ICLOUD = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs"


def publish(book: str, dest: str | None = None, **opts) -> bool:
    if not ICLOUD.exists():
        console.print("[red]iCloud Drive not found on this Mac[/red]")
        return False
    m4b = resolve_m4b(book)
    if m4b is None:
        console.print(f"[red]No finished audiobook found for:[/red] {book}")
        return False
    title = None
    manifest = m4b.parent / "manifest.json"
    if manifest.exists():
        try:
            title = json.loads(manifest.read_text()).get("title")
        except Exception:
            pass
    name = (title or m4b.parent.name.replace("_", " ").title()) + ".m4b"
    folder = ICLOUD / (dest or "Audiobooks")
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(m4b, folder / name)
    console.print(f"[green]Exported to iCloud Drive:[/green] {folder.name}/{name} "
                  f"[dim](uploads in background; then on iPhone: Files → share → Books)[/dim]")
    return True
