"""Actual bounded final reads shared by explicit public-PDF quality controls."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path


class PublicPdfStore:
    """Read the selected public resource only after final unit/resource ranking."""

    def __init__(self, source: Path, digest: str) -> None:
        """Bind the quality controls' resource42 to an immutable expected source."""
        self.source, self.digest = Path(source), digest
        self.calls = 0
        self.rows = 0

    def fetch(self, ids: Sequence[int]) -> dict[int, object]:
        """Actually read at most16MiB and verify content before counting success."""
        if tuple(ids) != (42,):
            raise ValueError("Unexpected final public-resource IDs")
        with self.source.open("rb") as stream:
            content = stream.read(16 * 1024**2 + 1)
        if len(content) > 16 * 1024**2 or hashlib.sha256(content).hexdigest() != self.digest:
            raise ValueError("Public resource changed or exceeded budget during final read")
        self.calls += 1
        self.rows += len(ids)
        return {42: {"filename": self.source.name, "bytes": len(content), "sha256": self.digest}}
