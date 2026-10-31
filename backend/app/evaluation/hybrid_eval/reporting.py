"""Stable report serialization, timing summaries and paired query statistics."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping, Sequence
from importlib import metadata
from typing import Any, TypeAlias

from backend.app.evaluation import dataset as evaluation_dataset
from backend.app.evaluation import metrics
from backend.app.retrieval.hybrid import documents as corpus

# Report payloads are dynamic JSON at the serialization boundary.
JsonObject: TypeAlias = dict[str, Any]
Rankings: TypeAlias = dict[str, list[int]]


class CountingStore(corpus.MemoryDocumentStore):
    """Observe final-fetch batch sizes without changing in-memory store semantics."""

    def __init__(self, documents: Sequence[corpus.CorpusDocument]) -> None:
        """Initialize an in-memory store and zero its observation counters."""
        super().__init__(documents)
        self.calls = 0
        self.rows = 0
        self.max_rows = 0

    def fetch(self, ids: Sequence[int]) -> Mapping[int, corpus.CorpusDocument]:
        """Count the attempted final batch before resolving its documents."""
        self.calls += 1
        self.rows += len(ids)
        self.max_rows = max(self.max_rows, len(ids))
        return super().fetch(ids)


def digest(value: Any) -> str:
    """Hash JSON using the existing sorted-key Unicode serialization contract.

    Default JSON separators (including spaces) are deliberate: compacting them
    would invalidate stored identities. This checksum detects payload changes;
    it is neither a signature nor proof that the experiment was executed.
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def environment() -> dict[str, str]:
    """Record the selected interpreter and numeric distribution versions."""
    return {
        "python": platform.python_version(),
        **{name: metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")},
    }


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a timing order statistic in ms, or zero for no observations.

    fraction is in [0, 1]. Select sorted[int((n - 1) * fraction)], without
    interpolation, then convert seconds to milliseconds and round to 3 decimals.
    """
    return round(sorted(values)[int((len(values) - 1) * fraction)] * 1000, 3) if values else 0.0


def paired_comparison(
    cases: Sequence[evaluation_dataset.EvaluationCase],
    baseline: Mapping[str, list[int]],
    candidate: Mapping[str, list[int]],
) -> JsonObject:
    """Compare paired query nDCG differences with fixed-seed resampling.

    Args:
        cases: Nonempty judged queries, with one aligned ranking in each run.
        baseline: Query-ID-keyed baseline IDs, already truncated to the top 10.
        candidate: Query-ID-keyed candidate IDs, already truncated to the top 10.

    Returns:
        Mean nDCG@10 difference, 10000-sample percentile interval and query
        win/loss counts (absolute differences <= 1e-12 count as ties).
        This describes the selected corpus; it does not correct model selection.
        The interval is for nDCG, not Recall, and resamples queries rather than
        documents. An interval containing zero does not establish a positive gain.
    """
    import numpy as np

    differences = np.array(
        [
            metrics.metrics_for_case(case, candidate[case.query_id])["ndcg_at_10"]
            - metrics.metrics_for_case(case, baseline[case.query_id])["ndcg_at_10"]
            for case in cases
        ]
    )
    rng = np.random.default_rng(42)
    # Use the same sampled query indices for each paired difference. Batches
    # bound the largest draw to 100 * query_count rather than 10000 * query_count;
    # memory still scales with query count. Preserve draw order for fixed results.
    means = np.concatenate(
        [differences[rng.integers(0, len(cases), size=(100, len(cases)))].mean(axis=1) for _ in range(100)]
    )
    return {
        "mean_ndcg_delta": float(differences.mean()),
        "bootstrap_95_percentile_interval": np.quantile(means, [0.025, 0.975]).tolist(),
        "bootstrap_samples": 10000,
        "seed": 42,
        "improved": int((differences > 1e-12).sum()),
        "regressed": int((differences < -1e-12).sum()),
        "tied": int((abs(differences) <= 1e-12).sum()),
        "scope": "Paired query bootstrap for this fixed corpus/selection; no multiple-search correction.",
    }
