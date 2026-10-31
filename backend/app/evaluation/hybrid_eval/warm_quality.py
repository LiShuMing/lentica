"""Fixed cold/warm lifecycle cost controls with actual providers, never serving claims."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

from backend.app.evaluation.hybrid_eval import reporting
from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import execution, warm_execution


def _measure(searcher, queries, config):
    rows = []
    for query in queries:
        start = time.perf_counter()
        result = searcher.search(hybrid.SearchQuery(query), config)
        rows.append(
            {
                "query": query,
                "ids": [hit.asset_id for hit in result.result.hits],
                "seconds": time.perf_counter() - start,
                "route_pids": {worker.branch: worker.pid for worker in result.workers},
                "final_pid": result.finalization.pid if result.finalization else None,
            }
        )
    return rows


def run(
    providers: Mapping[str, execution.ProviderSpec],
    metadata: Mapping[int, Mapping[str, object]],
    queries: Sequence[str],
) -> reporting.JsonObject:
    """Measure five fixed queries repeated twice; assert cold/warm rank equality.

    Cold includes worker/model/index loading on every request. Warm includes the
    first cold load and keeps successful retrieval workers; final worker remains
    cold. The caller supplies public queries before seeing performance results.
    """
    if len(queries) != 5:
        raise ValueError("The fixed lifecycle control requires exactly five queries")
    config = hybrid.SearchConfig(tuple(hybrid.Branch(name, 50) for name in providers))
    repeated = list(queries) * 2
    cold = _measure(execution.ProcessSearch(providers, metadata), repeated, config)
    with warm_execution.WarmProcessSearch(providers, metadata) as searcher:
        warm = _measure(searcher, repeated, config)
    if [row["ids"] for row in cold] != [row["ids"] for row in warm]:
        raise ValueError("Cold/warm real provider rankings differ")
    if any(len({row["route_pids"][name] for row in warm}) != 1 for name in providers):
        raise ValueError("Warm workers were not reused across successful requests")
    return {
        "cold": cold,
        "warm": warm,
        "latency_ms": {
            "cold_all": {
                "p50": reporting.percentile([row["seconds"] for row in cold], 0.5),
                "p95": reporting.percentile([row["seconds"] for row in cold], 0.95),
            },
            "warm_including_first": {
                "p50": reporting.percentile([row["seconds"] for row in warm], 0.5),
                "p95": reporting.percentile([row["seconds"] for row in warm], 0.95),
            },
            "warm_after_first": {
                "p50": reporting.percentile([row["seconds"] for row in warm[1:]], 0.5),
                "p95": reporting.percentile([row["seconds"] for row in warm[1:]], 0.95),
            },
        },
        "rankings_equal": True,
        "startup_and_max_ms": {
            name: {
                "first": rows[0]["seconds"] * 1000,
                "max": max(row["seconds"] for row in rows) * 1000,
                "total_seconds": sum(row["seconds"] for row in rows),
            }
            for name, rows in (("cold", cold), ("warm", warm))
        },
        "scope": "Ten serial local requests per arm, five fixed queries twice; cold final worker and no document store; not deployed latency, concurrency or throughput.",
    }
