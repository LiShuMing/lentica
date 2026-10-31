"""Reusable ranked retrieval contracts; importing this package has no runtime effects."""

from .contracts import (
    Branch,
    DocumentStore,
    FusionStrategy,
    Retriever,
    SearchConfig,
    SearchFilter,
    SearchQuery,
    SearchResult,
)
from .engine import HybridSearch, HydrationError, RetrievalError
from .fusion import RrfFusion

__all__ = [
    "Branch",
    "DocumentStore",
    "FusionStrategy",
    "Retriever",
    "SearchConfig",
    "SearchFilter",
    "SearchQuery",
    "SearchResult",
    "HybridSearch",
    "HydrationError",
    "RetrievalError",
    "RrfFusion",
]
