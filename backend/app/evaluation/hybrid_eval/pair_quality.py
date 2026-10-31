"""Fixed real text-pair reranking controls on historical public judgments."""

from __future__ import annotations

import dataclasses
import time

from backend.app.evaluation.hybrid_eval import dataset, index_quality, native_quality, pair_checkpoints, reporting
from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import lexical, pair_reranking, sparse, vector


def run(
    source: dataset.Dataset,
    dense: vector.ExactVectorRetriever,
    model: pair_reranking.PairReranker,
    *,
    progress: bool = False,
    score_cache: pair_checkpoints.PairScoreCache | None = None,
) -> reporting.JsonObject:
    """Rerank fixed BM25/three-route top20, recording all gains and regressions.

    The model scores the union once per query; timings identify this shared
    computation rather than pretending both routes independently ran the model.
    Labels participate only in metrics after ranks have been produced.
    """
    routes = {
        "bm25": lexical.Bm25Retriever(source.documents),
        "char": sparse.SparseTfidfRetriever(source.documents, analyzer="char_wb", ngram_range=(3, 5)),
        "bge": dense,
    }
    documents = {doc.asset_id: doc for doc in source.documents}
    configs = {
        name: dataclasses.replace(native_quality.fixed_configs()[name], top_k=20) for name in ("bm25", "three-routes")
    }
    rankings = {name + suffix: {} for name in configs for suffix in ("", "-pair")}
    candidate_recalls = {name: [] for name in configs}
    observations = []
    timings = []
    actual_pairs = 0
    reused_pairs = 0
    for index, case in enumerate(source.cases, start=1):
        query = hybrid.SearchQuery(case.query)
        candidates = {
            name: tuple(hit.asset_id for hit in hybrid.HybridSearch(routes).search(query, config).hits)
            for name, config in configs.items()
        }
        union = tuple(dict.fromkeys(identifier for ids in candidates.values() for identifier in ids))
        texts = [documents[identifier].title + " " + documents[identifier].text for identifier in union]
        started = time.perf_counter()
        key = reporting.digest([case.query_id, case.query, union])
        logits = score_cache.get(key, len(union)) if score_cache is not None else None
        if logits is None:
            logits = model.score(case.query, texts)
            timings.append(time.perf_counter() - started)
            actual_pairs += len(union)
            if score_cache is not None:
                score_cache.put(key, logits)
        else:
            reused_pairs += len(union)
        scores = dict(zip(union, logits, strict=True))
        for name, ids in candidates.items():
            rankings[name][case.query_id] = ids[:10]
            order = sorted(range(len(ids)), key=lambda position: (-scores[ids[position]], position))
            rankings[name + "-pair"][case.query_id] = tuple(ids[position] for position in order[:10])
            candidate_recalls[name].append(len(set(ids) & case.relevant_asset_ids) / len(case.relevant_asset_ids))
        observations.append(
            {"query_id": case.query_id, "union_ids": union, "logits": [scores[identifier] for identifier in union]}
        )
        if progress and (index % 25 == 0 or index == len(source.cases)):
            print(f"Pair reranked {index}/{len(source.cases)} queries", flush=True)
    return {
        "schema": "lentica-native-pair-fixed-v1",
        "model_version": model.version,
        "source": source.provenance,
        "quality": {name: index_quality.summarize(source.cases, rank) for name, rank in rankings.items()},
        "candidate_recall_at_20": {name: sum(values) / len(values) for name, values in candidate_recalls.items()},
        "paired_vs_same_candidates": {
            name: reporting.paired_comparison(source.cases, rankings[name], rankings[name + "-pair"])
            for name in configs
        },
        "actual_model_work": {
            "unique_query_document_pairs": actual_pairs,
            "reused_query_document_pairs": reused_pairs,
            "seconds": sum(timings),
            "query_union_ms": {"p50": reporting.percentile(timings, 0.5), "p95": reporting.percentile(timings, 0.95)},
        },
        "rankings": rankings,
        "pair_logits": observations,
        "ranking_sha256": reporting.digest(rankings),
        "scope": "Fixed historical public-test comparison; union pair work shared by two controls; no threshold tuning or default switch.",
    }
