"""Export plugins: each module exposes NAME and publish(book, **opts) -> bool.

Adding a target = adding a module here and listing it in TARGETS.
`book` is a book id under output/ or a direct path to an .m4b.
"""
from __future__ import annotations

TARGETS = ["books", "folder", "audiobookshelf", "icloud"]


def get(target: str):
    import importlib

    if target not in TARGETS:
        raise KeyError(f"unknown export target {target!r}; available: {', '.join(TARGETS)}")
    return importlib.import_module(f"export.{target}")
