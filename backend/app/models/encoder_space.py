"""An embedding space includes the encoder artifact and preprocessing contract."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def space_id(kind: str, model: str, revision: str | None, *, dimensions: int, encoding: str) -> str:
    artifact = []
    path = Path(model).expanduser()
    if path.is_dir():
        model = str(path.resolve())
        # Stat only the selected artifact directory; never read model weights.
        for entry in sorted(path.iterdir()):
            if entry.is_file() and entry.suffix in {".json", ".safetensors", ".bin", ".model"}:
                stat = entry.stat()
                artifact.append((entry.name, stat.st_size, stat.st_mtime_ns))
    try:
        runtime = version("transformers")
    except PackageNotFoundError:
        runtime = "unavailable"
    payload = json.dumps([kind, model, revision, artifact, dimensions, encoding, runtime], ensure_ascii=False)
    return kind + ":" + hashlib.sha256(payload.encode()).hexdigest()
