from __future__ import annotations

import unittest

from backend.app.evaluation.dataset import EvaluationCase
from backend.app.evaluation.metrics import aggregate_metrics, metrics_for_case


class EvaluationMetricsTests(unittest.TestCase):
    def test_metrics_measure_ranked_relevant_results(self) -> None:
        case = EvaluationCase("q1", "test", {7, 9}, "ocr")
        values = metrics_for_case(case, [1, 7, 2, 9])
        self.assertEqual(values["recall_at_1"], 0.0)
        self.assertEqual(values["recall_at_5"], 1.0)
        self.assertEqual(values["mrr"], 0.5)

    def test_aggregate_groups_categories(self) -> None:
        report = aggregate_metrics([(EvaluationCase("q1", "x", {1}, "visual"), [1])])
        self.assertEqual(report["queries"], 1)
        self.assertEqual(report["by_category"]["visual"]["recall_at_1"], 1.0)


if __name__ == "__main__":
    unittest.main()
