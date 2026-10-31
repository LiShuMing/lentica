"""Label-dependent historical weight diagnostics, never a query routing policy."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from backend.app.evaluation.hybrid_eval import reporting

WEIGHTS = {"text-weight-0.5": 0.5, "hybrid-equal": 1.0, "text-weight-2": 2.0}
ROUTES = ("text-only", "vector-only", *WEIGHTS)


def _validate(report: reporting.JsonObject) -> Sequence[reporting.JsonObject]:
    """Check fixed controls and recompute quality from recorded ranks and qrels."""
    rows = report["per_query"]
    if not rows or len(rows) > 10000 or len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("Weight diagnostics require bounded unique judged queries")
    for name, weight in WEIGHTS.items():
        config = report["configurations"][name]["config"]
        branches = config["branches"]
        if (
            config["top_k"] != 10
            or config["constant"] != 60
            or branches
            != [
                {"name": "text", "window": 50, "weight": weight},
                {"name": "vector", "window": 50, "weight": 1.0},
            ]
        ):
            raise ValueError("Weight diagnostics require the fixed c60/window50 controls")
    for row in rows:
        positives = set(row["relevant_source_ids"])
        if not positives:
            raise ValueError("Weight slices require positive qrels; no-answer is a separate task")
        ideal = sum(1 / math.log2(index + 2) for index in range(min(10, len(positives))))
        for name in ROUTES:
            ranking = row["top10_source_ids"][name]
            if len(ranking) > 10 or len(set(ranking)) != len(ranking):
                raise ValueError("Historical diagnostic ranking must be a unique Top10")
            recall = len(set(ranking) & positives) / len(positives)
            ndcg = (
                sum(1 / math.log2(index + 2) for index, identifier in enumerate(ranking) if identifier in positives)
                / ideal
            )
            metrics = row["metrics"][name]
            if any(
                not math.isfinite(metrics[key]) or abs(metrics[key] - actual) > 1e-12
                for key, actual in (("recall_at_10", recall), ("ndcg_at_10", ndcg))
            ):
                raise ValueError("Historical metrics disagree with recorded rankings/qrels")
    return rows


def _slice(rows: Sequence[reporting.JsonObject]) -> reporting.JsonObject:
    """Report group means and whether any one weight wins every query in it."""
    by_weight = {
        str(weight): {
            key: sum(row["metrics"][name][key] for row in rows) / len(rows) if rows else None
            for key in ("recall_at_10", "ndcg_at_10")
        }
        for name, weight in WEIGHTS.items()
    }
    intersection = set(WEIGHTS)
    win_memberships = {str(weight): 0 for weight in WEIGHTS.values()}
    for row in rows:
        best = max(row["metrics"][name]["ndcg_at_10"] for name in WEIGHTS)
        winners = {name for name in WEIGHTS if abs(row["metrics"][name]["ndcg_at_10"] - best) <= 1e-12}
        intersection &= winners
        for name in winners:
            win_memberships[str(WEIGHTS[name])] += 1
    return {
        "queries": len(rows),
        "query_ids": [row["query_id"] for row in rows],
        "weights": by_weight,
        "ndcg_best_memberships_including_ties": win_memberships,
        "weights_optimal_on_every_query": sorted(WEIGHTS[name] for name in intersection) if rows else [],
        "top_examples": [
            {key: row[key] for key in ("query_id", "query", "metrics", "top10_source_ids")}
            for row in sorted(
                rows,
                key=lambda row: (
                    -abs(row["metrics"]["vector-only"]["ndcg_at_10"] - row["metrics"]["text-only"]["ndcg_at_10"]),
                    row["query_id"],
                ),
            )[:3]
        ],
    }


def diagnose(report: reporting.JsonObject) -> reporting.JsonObject:
    """Compute fixed post-hoc slices; no inference, fitting or runtime changes.

    The margin0.1 and three weights are predeclared in spec0007. Group membership
    needs relevance labels and is unusable as an online query classifier. All
    means use each group's own query denominator, including failed retrievals.
    """
    rows = _validate(report)
    groups: Mapping[str, list[reporting.JsonObject]] = {"text_advantage": [], "vector_advantage": [], "other": []}
    for row in rows:
        delta = row["metrics"]["vector-only"]["ndcg_at_10"] - row["metrics"]["text-only"]["ndcg_at_10"]
        groups["vector_advantage" if delta >= 0.1 else "text_advantage" if delta <= -0.1 else "other"].append(row)
    return {
        "schema": "lentica-historical-weight-slices-v1",
        "source_ranking_sha256": report["ranking_sha256"],
        "encoder_id": report["encoder"]["encoder_id"],
        "margin": 0.1,
        "all": _slice(rows),
        "slices": {name: _slice(group) for name, group in groups.items()},
        "scope": "Recomputed statistics over existing exposed-test reports; zero new inference. Qrels define post-hoc slices, not online routing or validated weight selection. No default change.",
    }
