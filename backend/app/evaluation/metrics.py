from __future__ import annotations

import math
from collections import defaultdict

from backend.app.evaluation.dataset import EvaluationCase


def metrics_for_case(case: EvaluationCase, ranked_ids: list[int]) -> dict[str, float]:
    ranked_ids = list(dict.fromkeys(ranked_ids))
    relevant = case.relevant_asset_ids
    result: dict[str, float] = {}
    for k in (1, 5, 10):
        found = relevant & set(ranked_ids[:k])
        result[f"recall_at_{k}"] = len(found) / len(relevant) if relevant else 0.0
        result[f"hit_at_{k}"] = float(bool(found))
    reciprocal_rank = next(
        (1.0 / rank for rank, asset_id in enumerate(ranked_ids, start=1) if asset_id in relevant), 0.0
    )
    result["mrr"] = reciprocal_rank
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, asset_id in enumerate(ranked_ids[:10], start=1) if asset_id in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(10, len(relevant)) + 1))
    result["ndcg_at_10"] = dcg / ideal if ideal else 0.0
    return result


def aggregate_metrics(rows: list[tuple[EvaluationCase, list[int]]]) -> dict[str, object]:
    by_category: dict[str, list[dict[str, float]]] = defaultdict(list)
    all_rows: list[dict[str, float]] = []
    for case, ranked_ids in rows:
        values = metrics_for_case(case, ranked_ids)
        all_rows.append(values)
        by_category[case.category].append(values)
    return {
        "queries": len(all_rows),
        "overall": _average(all_rows),
        "by_category": {category: _average(values) for category, values in sorted(by_category.items())},
    }


def _average(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: round(sum(row[key] for row in rows) / len(rows), 4) for key in rows[0]}
