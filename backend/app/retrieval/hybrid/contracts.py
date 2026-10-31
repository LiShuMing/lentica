"""Application-independent contracts for ranked retrieval and record fetching."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

MAX_WINDOW = 10_000
MAX_CANDIDATES = 50_000


def asset_id(value: int) -> int:
    """Validate and return a nonnegative integer ID, excluding booleans."""
    if type(value) is not int or value < 0:
        raise ValueError("Asset IDs must be nonnegative integers")
    return value


@dataclass(frozen=True)
class SearchFilter:
    """Snapshot exact-match metadata and optional allowed IDs for every branch.

    Equalities are conjunctive and use Python scalar equality; a missing key
    never matches an explicit None. allowed_ids=None leaves IDs unrestricted,
    whereas an empty set permits none. Adapters apply both constraints before
    their candidate limit so excluded documents cannot consume the window.
    """

    equals: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)
    allowed_ids: frozenset[int] | None = None

    def __post_init__(self) -> None:
        copied = dict(self.equals)
        for key, value in copied.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Filter keys must be nonempty strings")
            if value is not None and type(value) not in (str, int, float, bool):
                raise ValueError("Exact filters accept scalar values only")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Filter values must be finite")
        object.__setattr__(self, "equals", MappingProxyType(copied))
        if self.allowed_ids is not None:
            object.__setattr__(self, "allowed_ids", frozenset(asset_id(item) for item in self.allowed_ids))

    def matches(self, identifier: int, metadata: Mapping[str, object]) -> bool:
        """Require every equality key to exist and match, including null values."""
        return (self.allowed_ids is None or identifier in self.allowed_ids) and all(
            key in metadata and metadata[key] == value for key, value in self.equals.items()
        )


@dataclass(frozen=True)
class SearchQuery:
    """Nonempty query text with the shared pre-window filter contract."""

    text: str
    filters: SearchFilter = field(default_factory=SearchFilter)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Query text must be nonempty")


@dataclass(frozen=True)
class Branch:
    """One named retriever's bounded candidate window and nonnegative weight.

    A zero window or weight disables execution and removes the requirement for
    a registered retriever. The window counts unique IDs, not duplicate entries.
    """

    name: str
    window: int = 50
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Branch names must be nonempty")
        if type(self.window) is not int or not 0 <= self.window <= MAX_WINDOW:
            raise ValueError(f"Branch window must be between 0 and {MAX_WINDOW}")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("Branch weights must be finite and nonnegative")


@dataclass(frozen=True)
class SearchConfig:
    """Validated branch budgets, fusion settings, final limit and failure policy.

    top_k limits the fused result; branch windows bound the input candidates.
    rrf_constant smooths rank differences: a smaller value favors top positions.
    failure_policy='partial' tolerates retriever failures only and marks the
    result degraded, even if every enabled retriever fails. Configuration errors,
    invalid fusion output and missing final records still fail the search.
    """

    branches: tuple[Branch, ...]
    top_k: int = 10
    rrf_constant: float = 60.0
    failure_policy: str = "raise"

    def __post_init__(self) -> None:
        object.__setattr__(self, "branches", tuple(self.branches))
        if type(self.top_k) is not int or not 1 <= self.top_k <= 1000:
            raise ValueError("top_k must be between 1 and 1000")
        if not math.isfinite(self.rrf_constant) or self.rrf_constant < 0:
            raise ValueError("RRF constant must be finite and nonnegative")
        if len({branch.name for branch in self.branches}) != len(self.branches):
            raise ValueError("Branch names must be unique")
        if not self.branches or sum(branch.window for branch in self.branches) > MAX_CANDIDATES:
            raise ValueError(f"Require branches with a total window budget <= {MAX_CANDIDATES}")
        if not math.isfinite(sum(branch.weight for branch in self.branches)):
            raise ValueError("Total branch weight must be finite")
        if self.failure_policy not in {"raise", "partial"}:
            raise ValueError("failure_policy must be raise or partial")


@dataclass(frozen=True)
class Contribution:
    """One branch's one-based rank and weighted reciprocal contribution."""

    branch: str
    rank: int
    score: float


@dataclass(frozen=True)
class FusedHit:
    """Stable document ID, final score and auditable branch contributions."""

    asset_id: int
    score: float
    contributions: tuple[Contribution, ...]


@dataclass(frozen=True)
class BranchDiagnostic:
    """Branch execution status, observed cost and optional failure class name."""

    name: str
    status: str
    candidates: int = 0
    elapsed_ms: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class SearchResult:
    """Ranked hits and optional final records, with explicit degradation state."""

    hits: tuple[FusedHit, ...]
    records: Mapping[int, object]
    branches: tuple[BranchDiagnostic, ...]
    degraded: bool


class Retriever(Protocol):
    """Produce ranked IDs after applying all shared filters before the limit."""

    def retrieve(self, query: SearchQuery, limit: int) -> Sequence[int]:
        """Return ordered IDs, filtering before limit; native scores stay private."""
        ...


class DocumentStore(Protocol):
    """Resolve only the final fused IDs in a single batch."""

    def fetch(self, ids: Sequence[int]) -> Mapping[int, object]:
        """Return every requested record or raise; omitted records fail search."""
        ...


class FusionStrategy(Protocol):
    """Combine ID ranks without comparing route-specific native scores."""

    def fuse(self, rankings: Mapping[str, Sequence[int]], config: SearchConfig) -> Sequence[FusedHit]:
        """Return unique finite-scored candidate hits within the final budget."""
        ...
