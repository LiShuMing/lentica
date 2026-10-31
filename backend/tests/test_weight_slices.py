"""Reject inconsistent historical evidence; report post-hoc winners with ties."""

from __future__ import annotations

import copy

import pytest

from backend.app.evaluation import dataset, metrics
from backend.app.evaluation.hybrid_eval import weight_slices


def _report():
    configs = {
        name: {
            "config": {
                "top_k": 10,
                "constant": 60,
                "branches": [
                    {"name": "text", "window": 50, "weight": weight},
                    {"name": "vector", "window": 50, "weight": 1.0},
                ],
            }
        }
        for name, weight in weight_slices.WEIGHTS.items()
    }
    case = dataset.EvaluationCase("q", "diagnostic", {1}, "fixture")
    ranks = {"text-only": [2], "vector-only": [1], "text-weight-0.5": [1], "hybrid-equal": [2], "text-weight-2": [2]}
    return {
        "ranking_sha256": "old-fixed-report",
        "encoder": {"encoder_id": "fixture"},
        "configurations": configs,
        "per_query": [
            {
                "query_id": "q",
                "query": case.query,
                "relevant_source_ids": ["1"],
                "metrics": {name: metrics.metrics_for_case(case, rank) for name, rank in ranks.items()},
                "top10_source_ids": {name: [str(identifier) for identifier in rank] for name, rank in ranks.items()},
            }
        ],
    }


def test_slices_report_label_dependence_and_empty_group_without_division_errors():
    report = _report()
    actual = weight_slices.diagnose(report)
    assert actual["slices"]["vector_advantage"]["queries"] == 1
    assert actual["slices"]["vector_advantage"]["weights_optimal_on_every_query"] == [0.5]
    assert actual["slices"]["text_advantage"]["weights"]["1.0"]["ndcg_at_10"] is None
    for tamper in ("metrics", "config", "duplicate"):
        invalid = copy.deepcopy(report)
        if tamper == "metrics":
            invalid["per_query"][0]["metrics"]["vector-only"]["ndcg_at_10"] = 0
        elif tamper == "config":
            invalid["configurations"]["hybrid-equal"]["config"]["constant"] = 10
        else:
            invalid["per_query"].append(invalid["per_query"][0])
        with pytest.raises(ValueError):
            weight_slices.diagnose(invalid)
