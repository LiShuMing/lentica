"""Fixed same-domain persistent-index controls; no test-based tuning."""

from __future__ import annotations

import dataclasses
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from backend.app.evaluation import dataset as evaluation_dataset
from backend.app.evaluation import metrics
from backend.app.evaluation.hybrid_eval import dataset, reporting
from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import adapters, ann, documents, fulltext, lexical, vector

if TYPE_CHECKING:
    from numpy import typing as npt


def summarize(
    cases: Sequence[evaluation_dataset.EvaluationCase], rankings: Mapping[str, Sequence[int]]
) -> reporting.JsonObject:
    """Macro-average binary relevance, never counting empty judgments as positives."""
    positive = [case for case in cases if case.relevant_asset_ids]
    rows = [metrics.metrics_for_case(case, list(rankings[case.query_id])[:10]) for case in positive]
    return {
        "positive_queries": len(positive),
        "no_eligible_relevant_queries": len(cases) - len(positive),
        "metrics": {
            key: sum(row[key] for row in rows) / len(rows) if rows else 0.0
            for key in ("recall_at_10", "ndcg_at_10", "mrr")
        },
    }


def compare_ann(
    source: dataset.Dataset, values: npt.ArrayLike, query_vectors: npt.ArrayLike, encoder_id: str, directory: Path
) -> reporting.JsonObject:
    """Build once, then compare true HNSW/exact under the same three ID domains.

    Query embeddings are actually computed by the calling workflow and shared;
    measured retrieval costs exclude query encoding. Native thread count is one.
    """
    import faiss

    faiss.omp_set_num_threads(1)
    ids = [doc.asset_id for doc in source.documents]
    encoded = {case.query: query_vectors[index] for index, case in enumerate(source.cases)}
    encode = encoded.__getitem__
    started = time.perf_counter()
    approximate = ann.HnswRetriever.build(
        directory,
        ids,
        values,
        encode,
        encoder_id=encoder_id,
        source_version=dataset.dataset_identity(source),
    )
    build_seconds = time.perf_counter() - started
    started = time.perf_counter()
    reloaded = ann.HnswRetriever(directory, encode, query_encoder_id=encoder_id)
    reload_seconds = time.perf_counter() - started
    exact = vector.ExactVectorRetriever(ids, values, encode, encoder_id=encoder_id, query_encoder_id=encoder_id)
    domains = {}
    for modulus in (1, 4, 100):
        allowed = frozenset(identifier for identifier in ids if identifier % modulus == 0)
        cases = [
            dataclasses.replace(case, relevant_asset_ids=case.relevant_asset_ids & allowed) for case in source.cases
        ]
        rankings = {name: {} for name in ("exact", "ann", "postfilter10", "postfilter50")}
        timings = {"exact": [], "ann": []}
        overlaps = []
        for case in cases:
            query = hybrid.SearchQuery(case.query, hybrid.SearchFilter(allowed_ids=allowed))
            for name, retriever in (("exact", exact), ("ann", approximate)):
                started = time.perf_counter()
                rankings[name][case.query_id] = retriever.retrieve(query, 10)
                timings[name].append(time.perf_counter() - started)
                if not set(rankings[name][case.query_id]) <= allowed:
                    raise ValueError("Index returned an excluded ID")
            if reloaded.retrieve(query, 10) != rankings["ann"][case.query_id]:
                raise ValueError("Reload changed ANN ranks")
            reference = rankings["exact"][case.query_id]
            overlaps.append(
                len(set(reference) & set(rankings["ann"][case.query_id])) / len(reference) if reference else 1
            )
            for window in (10, 50):
                unfiltered = exact.retrieve(hybrid.SearchQuery(case.query), window)
                rankings[f"postfilter{window}"][case.query_id] = tuple(item for item in unfiltered if item in allowed)[
                    :10
                ]
        domains[str(modulus)] = {
            "eligible_documents": len(allowed),
            "index_recall_at_10": sum(overlaps) / len(overlaps),
            "filter_violations": 0,
            "all_reload_ranks_equal": True,
            "quality": {name: summarize(cases, rank) for name, rank in rankings.items()},
            "retrieval_ms_excluding_encoding": {
                name: {"p50": reporting.percentile(times, 0.5), "p95": reporting.percentile(times, 0.95)}
                for name, times in timings.items()
            },
            "rankings": rankings,
        }
    return {
        "build_seconds": build_seconds,
        "reload_seconds": reload_seconds,
        "artifact_bytes": sum(
            (directory / name).stat().st_size for name in ("index.faiss", "ids.npy", "vectors.npy", "manifest.json")
        ),
        "parameters": {"m": 32, "ef_construction": 200, "ef_search": 128, "exact_filter_threshold": 256},
        "domains": domains,
    }


def shard_groups(corpus: Sequence[documents.CorpusDocument]) -> list[list[documents.CorpusDocument]]:
    """Split disjoint source IDs deterministically without using labels or queries."""
    shuffled = list(corpus)
    random.Random(42).shuffle(shuffled)
    return [shuffled[index::4] for index in range(4)]


def _reference_shards(groups, cases) -> reporting.JsonObject:
    local = [lexical.Bm25Retriever(group) for group in groups]
    shared = lexical.Bm25Statistics.merge([index.local_statistics for index in local])
    global_shards = [lexical.Bm25Retriever(group, statistics=shared) for group in groups]
    reference = lexical.Bm25Retriever([doc for group in groups for doc in group])
    equal = 0
    for case in cases:
        query = hybrid.SearchQuery(case.query)
        scores = {key: value for shard in global_shards for key, value in shard.score_documents(query).items()}
        merged = tuple(sorted(scores, key=lambda key: (-scores[key], key))[:50])
        if merged != reference.retrieve(query, 50):
            raise ValueError("Shared shard statistics differ from full reference")
        equal += 1
    return {
        "shared_stats_equal_queries": equal,
        "statistics": {
            "document_count": shared.document_count,
            "total_length": shared.total_length,
            "vocabulary": len(shared.document_frequencies),
        },
    }


def compare_shards(
    source: dataset.Dataset, dense_pools: Mapping[str, Sequence[int]], directory: Path
) -> reporting.JsonObject:
    """Compare native local/global statistics and their fixed rank fusion.

    Global native scoring uses one whole-corpus index, since this Python binding
    has no global-statistics injection API. The separate reference check verifies
    shared N/df/length with exact unquantized reference BM25, not Tantivy scoring.
    """
    groups = shard_groups(source.documents)
    started = time.perf_counter()
    indexes = [
        fulltext.TantivyRetriever.build(
            directory / f"shard-{index}", group, source_version=dataset.dataset_identity(source)
        )
        for index, group in enumerate(groups)
    ]
    whole = fulltext.TantivyRetriever.build(
        directory / "whole", source.documents, source_version=dataset.dataset_identity(source)
    )
    build_seconds = time.perf_counter() - started
    rankings = {name: {} for name in ("local", "global", "local-hybrid", "global-hybrid")}
    examples = []
    config = hybrid.SearchConfig((hybrid.Branch("text", 50), hybrid.Branch("dense", 50)))
    for case in source.cases:
        query = hybrid.SearchQuery(case.query)
        native = [row for index in indexes for row in index.retrieve_scored(query, 50)]
        local = tuple(key for key, _ in sorted(native, key=lambda row: (-row[1], row[0]))[:50])
        global_ids = whole.retrieve(query, 50)
        for name, text_ids in (("local", local), ("global", global_ids)):
            rankings[name][case.query_id] = text_ids[:10]
            result = hybrid.HybridSearch(
                {
                    "text": adapters.RankedListRetriever(text_ids),
                    "dense": adapters.RankedListRetriever(dense_pools[case.query_id]),
                }
            ).search(query, config)
            rankings[name + "-hybrid"][case.query_id] = tuple(hit.asset_id for hit in result.hits)
        if local[:5] != global_ids[:5] and len(examples) < 3:
            examples.append(
                {"query_id": case.query_id, "query": case.query, "local_top5": local[:5], "global_top5": global_ids[:5]}
            )
    return {
        "native_build_seconds": build_seconds,
        "shard_sizes": [len(group) for group in groups],
        "reference_global_stats": _reference_shards(groups, source.cases),
        "quality": {name: summarize(source.cases, rank) for name, rank in rankings.items()},
        "changed_top10_queries": {
            name: sum(
                rankings[name][case.query_id] != rankings[name.replace("local", "global")][case.query_id]
                for case in source.cases
            )
            for name in ("local", "local-hybrid")
        },
        "examples": examples,
        "rankings": rankings,
    }
