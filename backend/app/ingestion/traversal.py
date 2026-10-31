"""Deterministic, resumable traversal shared by local ingestion adapters."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import select

from backend.app.db.models import ScanCursor


def scan_cursor(db, source_id: int, stage: str) -> ScanCursor:
    cursor = db.scalar(select(ScanCursor).where(ScanCursor.source_id == source_id, ScanCursor.stage == stage))
    if cursor is None:
        cursor = ScanCursor(source_id=source_id, stage=stage, path="")
        db.add(cursor)
        db.flush()
    return cursor


def iter_source_files(root: Path, extensions: set[str], *, after: str = "") -> Iterator[Path]:
    """Keep at most one directory's entries per traversal depth, never a full inventory.

    Symlinks are excluded so a configured root cannot silently include another
    library. I/O errors propagate; callers must not infer deletion from failure.
    Cursor order is the case-sensitive absolute POSIX path order.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Source directory unavailable: {root}")

    def walk(directory: Path) -> Iterator[Path]:
        entries = sorted(directory.iterdir(), key=lambda item: item.name + ("/" if item.is_dir() else ""))
        for path in entries:
            if path.is_symlink() or path.name.startswith("."):
                continue
            if path.is_dir():
                prefix = str(path) + "/"
                if not after or prefix > after or after.startswith(prefix):
                    yield from walk(path)
            elif str(path) > after and path.is_file() and path.suffix.lower() in extensions:
                yield path

    yield from walk(root)
