"""Opt-in candidate, reranking and resource-presentation stages; no model imports."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Protocol

from . import contracts, engine, units


class Reranker(Protocol):
    """Order supplied candidate features without introducing additional IDs."""

    def rerank(self, query: contracts.SearchQuery, candidates: Sequence[units.RetrievalUnit]) -> Sequence[int]:
        """Return a complete permutation; ties should retain input order."""
        ...


@dataclass(frozen=True)
class StageConfig:
    """Separate branch windows, fused candidates, reranked prefix and asset top-k.

    search.top_k denotes final resources here. A temporary copy requests more
    unit candidates from HybridSearch, leaving the caller's search config intact.
    """

    search: contracts.SearchConfig
    candidate_window: int = 100
    rerank_window: int = 50
    max_units_per_asset: int = 2

    def __post_init__(self) -> None:
        if (
            type(self.candidate_window) is not int
            or type(self.rerank_window) is not int
            or not self.search.top_k <= self.rerank_window <= self.candidate_window <= 1000
            or type(self.max_units_per_asset) is not int
            or not 1 <= self.max_units_per_asset <= self.candidate_window
        ):
            raise ValueError("Require top_k <= rerank_window <= candidate_window <= 1000 and a positive unit cap")


@dataclass(frozen=True)
class UnitHit:
    """Candidate evidence and its original fusion score, retained after reranking."""

    unit: units.RetrievalUnit
    fusion_hit: contracts.FusedHit


@dataclass(frozen=True)
class AssetHit:
    """One resource with its best unit first and bounded supporting evidence.

    Rank is determined by unit order, potentially after reranking. fusion_score
    is the best unit's original RRF evidence, not a calibrated reranker score.
    """

    asset_id: int
    fusion_score: float
    evidence: tuple[UnitHit, ...]


class EvidenceStore(Protocol):
    """Hydrate final parents and only their selected supporting evidence.

    Unlike candidate feature reads, this call runs after aggregation and final
    top-k. SQL adapters can resolve exact selected chunks without reading every
    candidate chunk of each winning file. Existing ID-only stores remain valid.
    """

    def fetch_evidence(self, hits: Sequence[AssetHit]) -> Mapping[int, object]:
        """Return every selected parent record or raise; no missing citations."""
        ...


@dataclass(frozen=True)
class StageResult:
    """Final resources, candidate trace, branch status and separately scoped costs."""

    hits: tuple[AssetHit, ...]
    records: Mapping[int, object]
    candidates: tuple[contracts.FusedHit, ...]
    ordered_unit_ids: tuple[int, ...]
    branches: tuple[contracts.BranchDiagnostic, ...]
    degraded: bool
    feature_units: int
    reranked_units: int
    final_fetch_calls: int
    elapsed_ms: float


class StagedSearch:
    """Reuse existing retrieval, then rank units and fetch only final resources.

    First slice accepts text queries over multimodal units. Candidate feature
    access is explicitly separate from final record hydration. This is a local
    synchronous pipeline, with no implied model, persistent index or cancellation.
    """

    def __init__(
        self,
        retrievers: Mapping[str, contracts.Retriever],
        catalog: units.UnitStore,
        *,
        reranker: Reranker | None = None,
        store: contracts.DocumentStore | None = None,
        fusion: contracts.FusionStrategy | None = None,
        evidence_store: EvidenceStore | None = None,
    ) -> None:
        """Bind ranked-ID providers, an explicit unit snapshot and optional stages."""
        if store is not None and evidence_store is not None:
            raise ValueError("Choose one final ID-only or evidence-aware store")
        self.searcher = engine.HybridSearch(retrievers, fusion=fusion)
        self.catalog = catalog
        self.reranker = reranker
        self.store = store
        self.evidence_store = evidence_store

    def search(self, query: contracts.SearchQuery, config: StageConfig) -> StageResult:
        """Retrieve unit candidates, cap resource repetition and rerank a prefix.

        Raises:
            RetrievalError: A unit is missing, fails the shared filters, or the
                reranker returns anything other than the supplied permutation.
            HydrationError: Final asset fetching fails or omits a requested ID.

        Provider errors follow HybridSearch strict/partial policy. Exceptions
        raised by the reranker itself propagate unchanged, without partial output.
        """
        started = time.perf_counter()
        candidate_result = self.searcher.search(query, replace(config.search, top_k=config.candidate_window))
        features = self._features(query, candidate_result.hits)
        ordered = self._limit_assets(candidate_result.hits, features, config.max_units_per_asset)
        prefix = ordered[: config.rerank_window]
        if self.reranker is not None and prefix:
            reranked = tuple(self.reranker.rerank(query, [features[identifier] for identifier in prefix]))
            if (
                any(type(identifier) is not int for identifier in reranked)
                or len(reranked) != len(prefix)
                or set(reranked) != set(prefix)
            ):
                raise engine.RetrievalError("Reranker must return a complete candidate permutation")
            ordered = reranked + ordered[len(prefix) :]
        hits = self._aggregate(ordered, features, candidate_result.hits, config.search.top_k)
        records = self._fetch_assets(hits)
        return StageResult(
            hits,
            MappingProxyType(records),
            candidate_result.hits,
            ordered,
            candidate_result.branches,
            candidate_result.degraded,
            len(features),
            len(prefix) if self.reranker is not None else 0,
            int(bool(hits) and (self.store is not None or self.evidence_store is not None)),
            (time.perf_counter() - started) * 1000,
        )

    def _features(
        self, query: contracts.SearchQuery, hits: Sequence[contracts.FusedHit]
    ) -> Mapping[int, units.RetrievalUnit]:
        try:
            features = self.catalog.fetch_units([hit.asset_id for hit in hits])
        except KeyError as exc:
            raise engine.RetrievalError("Candidate ID absent from the unit catalog") from exc
        requested = {hit.asset_id for hit in hits}
        if set(features) != requested or any(identifier != unit.unit_id for identifier, unit in features.items()):
            raise engine.RetrievalError("Feature store must return exactly the requested unit identities")
        if any(not query.filters.matches(identifier, unit.metadata) for identifier, unit in features.items()):
            raise engine.RetrievalError("Candidate violates the shared pre-window filter")
        return features

    @staticmethod
    def _limit_assets(
        hits: Sequence[contracts.FusedHit], features: Mapping[int, units.RetrievalUnit], cap: int
    ) -> tuple[int, ...]:
        counts: dict[int, int] = {}
        result = []
        for hit in hits:
            parent = features[hit.asset_id].asset_id
            if counts.get(parent, 0) < cap:
                result.append(hit.asset_id)
                counts[parent] = counts.get(parent, 0) + 1
        return tuple(result)

    @staticmethod
    def _aggregate(
        ordered: Sequence[int],
        features: Mapping[int, units.RetrievalUnit],
        hits: Sequence[contracts.FusedHit],
        top_k: int,
    ) -> tuple[AssetHit, ...]:
        scores = {hit.asset_id: hit for hit in hits}
        groups: dict[int, list[UnitHit]] = {}
        for identifier in ordered:
            unit = features[identifier]
            if unit.asset_id not in groups and len(groups) >= top_k:
                continue
            groups.setdefault(unit.asset_id, []).append(UnitHit(unit, scores[identifier]))
        return tuple(
            AssetHit(parent, evidence[0].fusion_hit.score, tuple(evidence)) for parent, evidence in groups.items()
        )

    def _fetch_assets(self, hits: Sequence[AssetHit]) -> dict[int, object]:
        ids = [hit.asset_id for hit in hits]
        if (self.store is None and self.evidence_store is None) or not ids:
            return {}
        try:
            fetched = (
                self.evidence_store.fetch_evidence(hits) if self.evidence_store is not None else self.store.fetch(ids)
            )
            return {identifier: fetched[identifier] for identifier in ids}
        except Exception as exc:
            # Store plugins may raise arbitrary exceptions; final records are
            # mandatory even if an earlier retrieval branch degraded explicitly.
            raise engine.HydrationError("Final asset fetch failed or omitted a resource") from exc
