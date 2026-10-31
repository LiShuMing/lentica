"""Explicit local model artifacts and reproducible encoding-space identities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from . import snapshots


def verify(directory: Path, required: Sequence[str]) -> dict[str, str]:
    """Verify selected model files against a locally recorded acquisition manifest.

    No directory walk, download, credentials or model import occurs. Digests
    establish the exact local artifact identity; the mirror's manifest is not an
    upstream signature. Runtime constructors still disable remote model code.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "lentica-artifacts.json").read_text(encoding="utf-8"))
    result = {}
    for name in required:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Model files must stay inside the explicitly selected artifact directory")
        expected = manifest["files"][name]
        actual = directory / path
        digest = snapshots.sha256_file(actual)
        if actual.stat().st_size != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError(f"Model artifact identity mismatch: {name}")
        result[name] = digest
    return result


def space_id(kind: str, files: Mapping[str, str], encoding: Mapping[str, object]) -> str:
    """Bind weights/tokenizers/processors, prefixes, pooling and runtime versions."""
    digest = hashlib.sha256(
        snapshots.canonical({"files": dict(files), "encoding": dict(encoding)}).encode()
    ).hexdigest()
    return f"{kind}:{digest}"
