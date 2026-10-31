"""The original eleven-configuration CPU benchmark, independent of CLI IO."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence

from backend.app.evaluation import dataset as evaluation_dataset
from backend.app.evaluation import metrics
from backend.app.evaluation.hybrid_eval import dataset as datasets
from backend.app.evaluation.hybrid_eval import encoders, reporting
from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import adapters
from backend.app.retrieval.hybrid import lexical as lexical_adapter
from backend.app.retrieval.hybrid import vector as vector_adapter


def configurations(corpus_size: int) -> dict[str, hybrid.SearchConfig]:
    """Return the original eleven fixed baseline/ablation configurations."""

    def config(
        text_weight: float = 1.0, vector_weight: float = 1.0, constant: float = 60, window: int = 50
    ) -> hybrid.SearchConfig:
        return hybrid.SearchConfig(
            (
                hybrid.Branch("text", min(window, corpus_size), text_weight),
                hybrid.Branch("vector", min(window, corpus_size), vector_weight),
            ),
            rrf_constant=constant,
        )

    return {
        "text-only": config(vector_weight=0, window=200),
        "vector-only": config(text_weight=0, window=200),
        "hybrid-equal": config(),
        "text-weight-0.5": config(text_weight=0.5),
        "text-weight-2": config(text_weight=2),
        **{f"constant-{value}": config(constant=value) for value in (1, 10, 1000)},
        **{f"window-{value}": config(window=value) for value in (10, 200)},
        "window-full": config(window=corpus_size),
    }


def _collect_rankings(
    dataset: datasets.Dataset,
    configs: dict[str, hybrid.SearchConfig],
    lexical: hybrid.Retriever,
    vector: hybrid.Retriever,
    store: reporting.CountingStore,
) -> tuple[
    dict[str, list[tuple[evaluation_dataset.EvaluationCase, list[int]]]],
    dict[str, reporting.Rankings],
    dict[str, list[float]],
    dict[str, dict[str, list[int]]],
]:
    """Retrieve each route once and reuse its ranks across the fixed matrix."""
    documents, cases = dataset.documents, dataset.cases
    outcomes = {name: [] for name in configs}
    rankings = {name: {} for name in configs}
    latencies = {"text_full_pool": [], "vector_full_pool": [], "cached_fusion_fetch": []}
    single_routes = {}
    for case in cases:
        query = hybrid.SearchQuery(case.query)
        pools = {}
        for name, retriever in (("text", lexical), ("vector", vector)):
            started = time.perf_counter()
            pools[name] = retriever.retrieve(query, len(documents))
            latencies[name + "_full_pool"].append(time.perf_counter() - started)
        single_routes[case.query_id] = {name: list(pool[:5]) for name, pool in pools.items()}
        engine = hybrid.HybridSearch(
            {name: adapters.RankedListRetriever(pool) for name, pool in pools.items()}, store=store
        )
        for name, config in configs.items():
            started = time.perf_counter()
            result = engine.search(query, config)
            latencies["cached_fusion_fetch"].append(time.perf_counter() - started)
            ranked_ids = [hit.asset_id for hit in result.hits]
            outcomes[name].append((case, ranked_ids))
            rankings[name][case.query_id] = ranked_ids
    return outcomes, rankings, latencies, single_routes


def _per_query(dataset: datasets.Dataset, rankings: dict[str, reporting.Rankings]) -> list[reporting.JsonObject]:
    """Format per-query judgments, metrics and source-ID ranks."""
    cases = dataset.cases
    all_comparisons = []
    for case in cases:
        all_comparisons.append(
            {
                "query_id": case.query_id,
                "query": case.query,
                "category": case.category,
                "relevant_source_ids": [dataset.source_ids[index] for index in sorted(case.relevant_asset_ids)],
                "metrics": {
                    name: metrics.metrics_for_case(case, ranks[case.query_id]) for name, ranks in rankings.items()
                },
                "top10_source_ids": {
                    name: [dataset.source_ids[index] for index in ranks[case.query_id]]
                    for name, ranks in rankings.items()
                },
            }
        )
    return all_comparisons


def _examples(
    dataset: datasets.Dataset,
    all_comparisons: list[reporting.JsonObject],
    single_routes: Mapping[str, Mapping[str, Sequence[int]]],
    store: reporting.CountingStore,
) -> reporting.JsonObject:
    """Select both improvements and regressions from the actual comparisons."""
    # Dataset validation guarantees unique query IDs. Index judgments once rather
    # than scanning every case again for each displayed route/document pair.
    relevance_by_query = {case.query_id: case.relevant_asset_ids for case in dataset.cases}
    examples = {}
    for category, preferred, other in (
        ("text_wins", "text-only", "vector-only"),
        ("vector_wins", "vector-only", "text-only"),
        ("hybrid_gain", "hybrid-equal", "text-only"),
        ("hybrid_regression", "text-only", "hybrid-equal"),
    ):
        ordered = sorted(
            all_comparisons,
            key=lambda row: (
                -(row["metrics"][preferred]["ndcg_at_10"] - row["metrics"][other]["ndcg_at_10"]),
                row["query_id"],
            ),
        )
        selected = [
            row for row in ordered if row["metrics"][preferred]["ndcg_at_10"] > row["metrics"][other]["ndcg_at_10"]
        ][:3]
        examples[category] = [
            {
                **row,
                "routes_top5": {
                    route: [
                        {
                            "source_id": dataset.source_ids[index],
                            "title": store.documents[index].title,
                            "relevant": index in relevance_by_query[row["query_id"]],
                        }
                        for index in pool
                    ]
                    for route, pool in single_routes[row["query_id"]].items()
                },
            }
            for row in selected
        ]
    return examples


def run(dataset: datasets.Dataset, *, seed: int = 42, dimensions: int = 128) -> reporting.JsonObject:
    """Run the original eleven-way CPU comparison without changing defaults."""
    documents, cases = dataset.documents, dataset.cases
    started = time.perf_counter()
    lexical = lexical_adapter.Bm25Retriever(documents)
    lexical_build = time.perf_counter() - started
    started = time.perf_counter()
    encoder = encoders.LatentSemanticEncoder(documents, dimensions=dimensions, seed=seed)
    ids = [doc.asset_id for doc in documents]
    vector = vector_adapter.ExactVectorRetriever(
        ids,
        encoder.vectors,
        encoder.encode_query,
        encoder_id=encoder.encoder_id,
        query_encoder_id=encoder.encoder_id,
        metadata={doc.asset_id: doc.metadata for doc in documents},
    )
    vector_build = time.perf_counter() - started
    configs = configurations(len(documents))
    store = reporting.CountingStore(documents)
    outcomes, rankings, latencies, single_routes = _collect_rankings(dataset, configs, lexical, vector, store)
    reports = {
        name: {
            "metrics": metrics.aggregate_metrics(rows),
            "config": {
                "top_k": configs[name].top_k,
                "constant": configs[name].rrf_constant,
                "branches": [branch.__dict__ for branch in configs[name].branches],
            },
        }
        for name, rows in outcomes.items()
    }
    all_comparisons = _per_query(dataset, rankings)
    examples = _examples(dataset, all_comparisons, single_routes, store)
    ranking_hash = hashlib.sha256(json.dumps(rankings, sort_keys=True).encode()).hexdigest()
    return {
        "dataset": {
            **dataset.provenance,
            "documents": len(documents),
            "judged_queries": len(cases),
            "relevance_pairs": sum(len(case.relevant_asset_ids) for case in cases),
        },
        "environment": reporting.environment(),
        "encoder": {
            "kind": "corpus-fitted TF-IDF + truncated SVD",
            "seed": seed,
            "dimensions": encoder.dimensions,
            "encoder_id": encoder.encoder_id,
            "explained_variance": encoder.explained_variance,
            "training_uses": "corpus only; no query labels or query fitting",
        },
        "build_seconds": {"bm25": round(lexical_build, 4), "lsa_and_vectors": round(vector_build, 4)},
        "latency_ms": {
            name: {"p50": reporting.percentile(values, 0.5), "p95": reporting.percentile(values, 0.95)}
            for name, values in latencies.items()
        },
        "latency_scope": "Full-pool exact retrieval measured once per query; ablations reuse ordered IDs. "
        "Cached fusion/fetch timings are not end-to-end index performance.",
        "configurations": reports,
        "ranking_sha256": ranking_hash,
        "final_fetch": {"calls": store.calls, "records": store.rows, "max_records_per_call": store.max_rows},
        "examples": examples,
        "per_query": all_comparisons,
    }
