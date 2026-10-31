"""Candidate coverage and loss attribution; relevance labels stay in evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from backend.app.evaluation import dataset as evaluation_dataset
from backend.app.evaluation import metrics
from backend.app.evaluation.hybrid_eval import reporting


def recall(relevant: set[int] | frozenset[int], ranked: Sequence[int], k: int) -> float:
    """Compute set coverage at k; no-positive cases have recall zero."""
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def candidate_report(
    cases: Sequence[evaluation_dataset.EvaluationCase],
    pools: Mapping[str, Mapping[str, Sequence[int]]],
    final: Mapping[str, list[int]],
    *,
    windows: Mapping[str, int],
    candidates: Mapping[str, list[int]],
) -> reporting.JsonObject:
    """Separate route misses, bounded-fusion losses and final-ranking losses.

    Args:
        cases: Nonempty judged queries with aligned pool/ranking keys.
        pools: Query -> route -> ordered IDs, covering diagnostic cutoffs.
        final: Final IDs, truncated to ten, keyed by query ID.
        windows: Actual per-route windows for the selected pipeline.
        candidates: Fused candidate IDs before final top-k/reranking.

    Returns:
        Macro route/union recall, exclusive relevant counts and per-query loss
        counts. Oracle nDCG is a label-derived upper bound, never a retriever.
    """
    if (
        not cases
        or not windows
        or any(type(window) is not int or window < 0 for window in windows.values())
        or any(not case.relevant_asset_ids for case in cases)
        or len({case.query_id for case in cases}) != len(cases)
    ):
        raise ValueError("Diagnostics require judged queries and nonnegative route windows")
    routes = sorted(windows)
    route_rows = {name: {str(k): [] for k in (50, 100, 200)} for name in routes}
    union_rows = {str(k): [] for k in (50, 100, 200)}
    exclusive = dict.fromkeys(routes, 0)
    per_query = []
    for case in cases:
        relevant = case.relevant_asset_ids
        pool = pools[case.query_id]
        for k in (50, 100, 200):
            combined: set[int] = set()
            for name in routes:
                route_rows[name][str(k)].append(recall(relevant, pool[name], k))
                combined.update(pool[name][:k])
            union_rows[str(k)].append(len(relevant & combined) / len(relevant))
        bounded = {name: set(pool[name][: windows[name]]) for name in routes}
        union = set().union(*bounded.values())
        for name in routes:
            others = set().union(*(values for other, values in bounded.items() if other != name))
            exclusive[name] += len((relevant & bounded[name]) - others)
        fused = set(candidates[case.query_id])
        found = set(final[case.query_id][:10])
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(10, len(relevant)) + 1))
        oracle = sum(1 / math.log2(rank + 1) for rank in range(1, min(10, len(relevant & fused)) + 1)) / ideal
        per_query.append(
            {
                "query_id": case.query_id,
                "category": case.category,
                "relevant": len(relevant),
                "not_in_route_union": len(relevant - union),
                "lost_at_candidate_fusion": len((relevant & union) - fused),
                "lost_after_candidates": len((relevant & fused) - found),
                "found_top10": len(relevant & found),
                "bounded_union_recall": len(relevant & union) / len(relevant),
                "fused_candidate_recall": len(relevant & fused) / len(relevant),
                "oracle_ndcg_at_10": oracle,
                **metrics.metrics_for_case(case, final[case.query_id][:10]),
            }
        )
    by_category = {}
    for category in sorted({case.category for case in cases}):
        rows = [row for row in per_query if row["category"] == category]
        by_category[category] = {
            "queries": len(rows),
            **{
                key: sum(row[key] for row in rows) / len(rows)
                for key in ("bounded_union_recall", "fused_candidate_recall", "recall_at_10", "ndcg_at_10", "mrr")
            },
        }
    return {
        "queries": len(cases),
        "route_recall": {
            name: {f"at_{k}": sum(values) / len(values) for k, values in rows.items()}
            for name, rows in route_rows.items()
        },
        "union_recall": {f"at_{k}_per_route": sum(values) / len(values) for k, values in union_rows.items()},
        "exclusive_relevant_pairs_at_configured_windows": exclusive,
        "mean_bounded_union_recall": sum(row["bounded_union_recall"] for row in per_query) / len(cases),
        "mean_fused_candidate_recall": sum(row["fused_candidate_recall"] for row in per_query) / len(cases),
        "mean_oracle_ndcg_at_10": sum(row["oracle_ndcg_at_10"] for row in per_query) / len(cases),
        "by_category": by_category,
        "per_query": per_query,
        "scope": "Macro judged-query coverage; exclusive counts are query-document pairs; oracle uses labels only here.",
    }
