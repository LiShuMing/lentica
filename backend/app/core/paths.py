from __future__ import annotations

from pathlib import Path


def to_windows_path(path: str | Path) -> str:
    text = str(path)
    if text.startswith("/mnt/c/"):
        return "C:\\" + text[len("/mnt/c/") :].replace("/", "\\")
    return text


def safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name
