"""Export into an Audiobookshelf library folder.

Audiobookshelf watches its library directory, so a copy is a publish; if
ABS_URL and ABS_TOKEN are configured (env or config.yaml `export.abs`), a
library scan is triggered immediately instead of waiting for the watcher.
"""
from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console

from export.folder import publish as folder_publish
from pipeline.config import CFG

console = Console()

NAME = "audiobookshelf"


def publish(book: str, dest: str | None = None, **opts) -> bool:
    abs_cfg = (CFG.get("export") or {}).get("abs") or {}
    dest = dest or abs_cfg.get("library_dir")
    if not dest:
        console.print("[red]audiobookshelf export needs --dest or export.abs.library_dir "
                      "in config.yaml[/red]")
        return False
    if not folder_publish(book, dest=dest):
        return False

    url = os.environ.get("ABS_URL") or abs_cfg.get("url")
    token = os.environ.get("ABS_TOKEN") or abs_cfg.get("token")
    library_id = os.environ.get("ABS_LIBRARY_ID") or abs_cfg.get("library_id")
    if url and token and library_id:
        import urllib.request
        req = urllib.request.Request(f"{url.rstrip('/')}/api/libraries/{library_id}/scan",
                                     method="POST",
                                     headers={"Authorization": f"Bearer {token}"})
        try:
            urllib.request.urlopen(req, timeout=10)
            console.print("[green]Audiobookshelf scan triggered[/green]")
        except OSError as e:
            console.print(f"[yellow]scan trigger failed ({e}); the folder watcher will pick it up[/yellow]")
    return True
