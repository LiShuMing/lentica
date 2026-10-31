"""Product SQL/model adapters composed through the shared hybrid coordinator."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from sqlalchemy.orm import Session

from backend.app.models import local_clip
from backend.app.retrieval import asset_store, photo_search, query_plan, service
from backend.app.retrieval.hybrid import adapters, contracts, stages, units


@dataclass(frozen=True)
class AssetHit:
    """One globally ranked file with typed presentation and selected evidence."""

    file_id: int
    asset_type: str
    rank: int
    score: float
    unit_ids: tuple[int, ...]
    photo: photo_search.PhotoHit | None
    documents: tuple[service.RetrievedSource, ...]


@dataclass(frozen=True)
class AssetSearchResult:
    """Authoritative global order plus compatibility projections and diagnostics."""

    results: tuple[AssetHit, ...]
    plan: query_plan.RetrievalPlan
    debug: dict[str, object]

    @property
    def photos(self) -> list[photo_search.PhotoHit]:
        """Project selected images in global order, without a second search."""
        return [hit.photo for hit in self.results if hit.photo is not None]

    @property
    def documents(self) -> list[service.RetrievedSource]:
        """Project selected supporting chunks in file/evidence order."""
        return [document for hit in self.results for document in hit.documents]

    @property
    def answer(self) -> str:
        """Describe only the final global result set."""
        if not self.results:
            return "本地索引中没有找到相关结果。"
        images = len(self.photos)
        return f"找到 {len(self.results)} 条本地结果：{images} 张图片、{len(self.results) - images} 份文档。"


@dataclass
class _Prepared:
    catalog: dict[int, units.RetrievalUnit]
    rankings: dict[str, tuple[int, ...]]
    weights: dict[str, float]
    photos: photo_search.PhotoCandidates | None
    documents: dict[int, service.RetrievedSource]
    debug: dict[str, object]


def _prepare(
    db: Session,
    plan: query_plan.RetrievalPlan,
    window: int,
    embedders: Callable[[], tuple[local_clip.Embedder, local_clip.Embedder]] | None,
) -> _Prepared:
    """Read only candidate features; preserve plan constraints before each window."""
    catalog, rankings, weights, documents, debug = {}, {}, {}, {}, {}
    photos = None
    if set(plan.asset_types) & {"photo", "screenshot"}:
        if embedders is None:
            raise ValueError("Photo search requires an explicit embedder factory")
        visual, text = embedders()
        photos = photo_search.PhotoSearch(db, visual, text).prepare(plan.original_query, plan=plan, window=window)
        for name, hits in photos.rankings.items():
            rankings["photo." + name] = tuple(hit.asset_id * 2 for hit in hits)
            weights["photo." + name] = photos.weights[name]
            for hit in hits:
                kind = photo_search.photo_asset_type(photos.candidates[hit.asset_id]["image"])
                catalog[hit.asset_id * 2] = units.RetrievalUnit(
                    hit.asset_id * 2, hit.asset_id, "image", metadata={"asset_type": kind, "eligible": True}
                )
        debug.update(photos.debug)
    if plan.use_document_retrieval:
        # Existing cached TF-IDF/exact policy is an unfused native branch. It
        # remains a compatibility helper for callers outside this main service.
        candidates = service.search_knowledge(db, plan.original_query, limit=window)
        for row in candidates:
            identifier = row.chunk_id * 2 + 1
            documents[row.chunk_id] = row
            catalog[identifier] = units.RetrievalUnit(
                identifier,
                row.file_id,
                "text",
                metadata={"asset_type": "document", "eligible": True},
            )
        rankings["document.tfidf"] = tuple(row.chunk_id * 2 + 1 for row in candidates)
        weights["document.tfidf"] = 1.0
    return _Prepared(catalog, rankings, weights, photos, documents, debug)


def _present(result: stages.StageResult, prepared: _Prepared, plan: query_plan.RetrievalPlan) -> tuple[AssetHit, ...]:
    presented = []
    for rank, hit in enumerate(result.hits, 1):
        record = result.records[hit.asset_id]
        photo, citations = None, []
        for evidence in hit.evidence:
            unit = evidence.unit
            if unit.kind == "image":
                maximum = sum(prepared.photos.weights.values()) / 61
                score = min(1.0, evidence.fusion_hit.score / maximum) if maximum else 0.0
                photo = photo_search.present_photo(
                    record.file,
                    record.image,
                    prepared.photos,
                    plan,
                    score=round(score, 6),
                    sources=[part.branch.removeprefix("photo.") for part in evidence.fusion_hit.contributions],
                )
            else:
                chunk_id = (unit.unit_id - 1) // 2
                chunk = record.chunks[chunk_id]
                citations.append(
                    service.RetrievedSource(
                        hit.asset_id,
                        chunk.id,
                        record.file.canonical_path,
                        record.file.windows_path,
                        chunk.page_or_segment,
                        chunk.text,
                        prepared.documents[chunk.id].score,
                    )
                )
        presented.append(
            AssetHit(
                hit.asset_id,
                str(hit.evidence[0].unit.metadata["asset_type"]),
                rank,
                hit.fusion_score,
                tuple(e.unit.unit_id for e in hit.evidence),
                photo,
                tuple(citations),
            )
        )
    return tuple(presented)


def _run_pipeline(
    db: Session, prepared: _Prepared, query: str, limit: int, window: int
) -> tuple[stages.StageResult, asset_store.SqlAssetStore, int]:
    """Compose ranked-ID routes, bounded aggregation and one final SQL store."""
    metadata = {identifier: unit.metadata for identifier, unit in prepared.catalog.items()}
    retrievers = {name: adapters.RankedListRetriever(ids, metadata=metadata) for name, ids in prepared.rankings.items()}
    store = asset_store.SqlAssetStore(db)
    pipeline = stages.StagedSearch(retrievers, units.UnitCatalog(list(prepared.catalog.values())), evidence_store=store)
    candidate_window = min(1000, max(limit, sum(len(ids) for ids in prepared.rankings.values())))
    config = stages.StageConfig(
        contracts.SearchConfig(
            tuple(contracts.Branch(name, window, prepared.weights[name]) for name in retrievers), top_k=limit
        ),
        candidate_window=candidate_window,
        rerank_window=candidate_window,
        max_units_per_asset=min(2, candidate_window),
    )
    result = pipeline.search(
        contracts.SearchQuery(
            query, contracts.SearchFilter(equals={"eligible": True}, allowed_ids=frozenset(metadata))
        ),
        config,
    )
    return result, store, candidate_window


def search_assets(
    db: Session,
    query: str,
    *,
    limit: int = 12,
    asset_types: list[str] | None = None,
    embedders: Callable[[], tuple[local_clip.Embedder, local_clip.Embedder]] | None = None,
) -> AssetSearchResult:
    """Run the product's ID-first hybrid pipeline with one total file limit.

    Model construction is lazy and only selected for an image-capable plan. The
    injected factory follows existing API configuration; no model downloads,
    private reindex, native index activation or service lifecycle change is added.
    Candidate preparation/SQL record hydration costs are distinct from the core
    cached-rank branch diagnostics. Failures propagate without silent degradation.

    Args:
        db: Current local database session; no library scan or reindex occurs.
        query: Nonempty user text, also used by the existing query-plan rules.
        limit: Total file budget, shared across all requested asset types.
        asset_types: Explicit scope, or None/empty to use existing intent rules.
        embedders: Factory for the selected visual and text encoders; required
            only when the plan includes photos or screenshots.

    Returns:
        Globally ranked files, compatibility projections, plan and diagnostics.

    Raises:
        ValueError: The query, limit, asset scope or required factory is invalid.
        engine.RetrievalError: A candidate provider violates the shared contract.
        engine.HydrationError: Final selected records cannot be resolved.
    """
    if not isinstance(query, str) or not query.strip() or type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Search requires nonempty text and a 1..100 resource limit")
    if asset_types is not None and any(kind not in {"photo", "screenshot", "document"} for kind in asset_types):
        raise ValueError("Unsupported product asset type")
    plan = query_plan.build_retrieval_plan(db, query, expand=photo_search.expand_query, asset_types=asset_types)
    window = max(50, limit * 2)
    started = time.perf_counter()
    prepared = _prepare(db, plan, window, embedders)
    preparation_ms = (time.perf_counter() - started) * 1000
    result, store, candidate_window = _run_pipeline(db, prepared, query, limit, window)
    presented = _present(result, prepared, plan)
    debug = {
        **prepared.debug,
        "pipeline": "staged-hybrid",
        "pipeline_elapsed_ms": result.elapsed_ms,
        "candidate_preparation_ms": preparation_ms,
        "candidate_window": candidate_window,
        "branch_window": window,
        "branches": [asdict(row) for row in result.branches],
        "degraded": result.degraded,
        "rrf_candidate_count": len(result.candidates),
        "final_result_count": len(presented),
        "final_fetch_calls": store.calls,
        "final_file_ids": list(store.file_ids),
        "final_chunk_ids": list(store.chunk_ids),
        "document_result_count": sum(len(hit.documents) for hit in presented),
        "score_scope": "results.score=raw RRF; compatibility photo/doc scores are route-specific, not probabilities",
    }
    return AssetSearchResult(presented, plan, debug)
