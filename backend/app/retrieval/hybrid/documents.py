"""Frozen corpus records and an in-memory final-record store."""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Mapping, Sequence

from . import contracts


@dataclasses.dataclass(frozen=True)
class CorpusDocument:
    """Frozen source fields with a stable ID and a shallow metadata snapshot.

    The top-level metadata mapping is copied and read-only; nested values are
    retained by reference. Callers must not treat it as recursively immutable.
    """

    asset_id: int
    text: str
    title: str = ""
    metadata: Mapping[str, object] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the ID and snapshot metadata as a read-only mapping."""
        contracts.asset_id(self.asset_id)
        object.__setattr__(self, "metadata", types.MappingProxyType(dict(self.metadata)))


class MemoryDocumentStore:
    """Resolve final ranked IDs from an explicit in-memory corpus."""

    def __init__(self, documents: Sequence[CorpusDocument]) -> None:
        """Index documents by unique asset ID without copying their contents."""
        self.documents = {item.asset_id: item for item in documents}
        if len(self.documents) != len(documents):
            raise ValueError("Corpus IDs must be unique")

    def fetch(self, ids: Sequence[int]) -> Mapping[int, CorpusDocument]:
        """Fetch final documents in one batch.

        Args:
            ids: Ranked asset IDs to resolve.

        Returns:
            Records keyed by the requested IDs.

        Raises:
            KeyError: A requested document is absent.
        """
        return {identifier: self.documents[identifier] for identifier in ids}
