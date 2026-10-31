"""Compatibility exports and the precomputed-ranking adapter.

Concrete retrieval implementations live in focused modules. These explicit
re-exports preserve callers of the original hybrid.adapters API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from . import contracts, fusion
from .documents import CorpusDocument as CorpusDocument
from .documents import MemoryDocumentStore as MemoryDocumentStore
from .lexical import Bm25Retriever as Bm25Retriever
from .lexical import tokenize as tokenize
from .vector import ExactVectorRetriever as ExactVectorRetriever


class RankedListRetriever:
    """Adapter for precomputed ranks, including cached evaluation results.

    Applies filters only to the supplied pool; documents excluded when that pool
    was built cannot be recovered here. Rerank evaluation is therefore equivalent
    to retrieval only when the cache covers each requested candidate window and
    was built with compatible filters. Metadata is retained by reference.
    """

    def __init__(self, ids: Sequence[int], *, metadata: Mapping[int, Mapping[str, object]] | None = None) -> None:
        """Snapshot an ordered pool while retaining optional filter metadata."""
        self.ids = tuple(ids)
        self.metadata = metadata or {}

    def retrieve(self, query: contracts.SearchQuery, limit: int) -> tuple[int, ...]:
        """Filter and deduplicate cached IDs before the candidate limit."""
        return fusion.unique_ids(
            [
                identifier
                for identifier in self.ids
                if query.filters.matches(identifier, self.metadata.get(identifier, {}))
            ],
            limit,
        )
