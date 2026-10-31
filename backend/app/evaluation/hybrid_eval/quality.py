"""Frozen public-corpus diagnostics and one predeclared rule-reranking control."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping

from backend.app.evaluation import metrics
from backend.app.evaluation.hybrid_eval import dataset as datasets
from backend.app.evaluation.hybrid_eval import diagnostics, execution, reporting, workflow
from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import adapters, reranking, stages, units


def _catalog(dataset: datasets.Dataset) -> units.UnitCatalog:
    return units.UnitCatalog(
        [
            units.RetrievalUnit(
                doc.asset_id,
                doc.asset_id,
                "text",
                evidence=(units.TextEvidence("native", doc.title + " " + doc.text, "beir-corpus", "v1"),),
            )
            for doc in dataset.documents
        ]
    )


def _cached_stages(
    dataset: datasets.Dataset,
    catalog: units.UnitCatalog,
    config: stages.StageConfig,
    pools: Mapping[str, Mapping[str, list[int] | tuple[int, ...]]],
    *,
    rerank: bool,
) -> tuple[reporting.Rankings, reporting.Rankings]:
    rankings, candidates = {}, {}
    for case in dataset.cases:
        pipeline = stages.StagedSearch(
            {name: adapters.RankedListRetriever(pool) for name, pool in pools[case.query_id].items()},
            catalog,
            reranker=reranking.ExactIdentifierReranker() if rerank else None,
        )
        result = pipeline.search(hybrid.SearchQuery(case.query), config)
        rankings[case.query_id] = [hit.asset_id for hit in result.hits]
        candidates[case.query_id] = [hit.asset_id for hit in result.candidates]
    return rankings, candidates


def _metric_summary(dataset: datasets.Dataset, rankings: reporting.Rankings) -> dict[str, float]:
    rows = [metrics.metrics_for_case(case, rankings[case.query_id]) for case in dataset.cases]
    return {key: sum(row[key] for row in rows) / len(rows) for key in execution.METRICS}


def _actual_pipeline(
    dataset: datasets.Dataset,
    catalog: units.UnitCatalog,
    routes: Mapping[str, hybrid.Retriever],
    config: stages.StageConfig,
    expected: reporting.Rankings,
    *,
    rerank: bool,
) -> reporting.JsonObject:
    store = reporting.CountingStore(dataset.documents)
    pipeline = stages.StagedSearch(
        routes, catalog, reranker=reranking.ExactIdentifierReranker() if rerank else None, store=store
    )
    elapsed, features, reranked = [], [], []
    for case in dataset.cases:
        started = time.perf_counter()
        result = pipeline.search(hybrid.SearchQuery(case.query), config)
        elapsed.append(time.perf_counter() - started)
        if [hit.asset_id for hit in result.hits] != expected[case.query_id]:
            raise ValueError("Actual staged ranking differs from cached candidate execution")
        features.append(result.feature_units)
        reranked.append(result.reranked_units)
    return {
        "latency_ms": {"p50": reporting.percentile(elapsed, 0.5), "p95": reporting.percentile(elapsed, 0.95)},
        "candidate_feature_units": {"total": sum(features), "max_per_query": max(features)},
        "reranked_units": {"total": sum(reranked), "max_per_query": max(reranked)},
        "final_fetch": {"calls": store.calls, "records": store.rows, "max_per_call": store.max_rows},
        "scope": "Actual serial local retrieval, feature access, optional rules, aggregation and in-memory final fetch; "
        "excludes index build; no pretrained inference.",
    }


def run(
    selection: reporting.JsonObject, train: datasets.Dataset, test: datasets.Dataset, *, progress: bool = False
) -> reporting.JsonObject:
    """Diagnose a locked historical selection without selecting new parameters.

    The fixed identifier reranker is an experimental control, including losses.
    Existing selection identity is validated before constructing any retriever.
    """
    configs = workflow.validate_lock(selection, train, test)
    selected = configs["selected"]
    names = {branch.name for name in ("baseline", "selected") for branch in configs[name].branches}
    routes, builds = execution.build_routes(test.documents, names, progress=progress)
    pools, route_latency = execution.cache_pools(routes, test.cases, progress=progress)
    catalog = _catalog(test)
    config = stages.StageConfig(selected, candidate_window=100, rerank_window=50, max_units_per_asset=1)
    baseline_report, baseline = execution.evaluate_config(configs["baseline"], test.cases, pools)
    selected_report, original = execution.evaluate_config(selected, test.cases, pools)
    staged, candidates = _cached_stages(test, catalog, config, pools, rerank=False)
    if original != staged:
        raise ValueError("Opt-in stages changed the frozen no-reranking baseline")
    reranked, _ = _cached_stages(test, catalog, config, pools, rerank=True)
    windows = {branch.name: branch.window for branch in selected.branches if branch.weight and branch.window}
    comparisons = {
        "baseline": {"metrics": baseline_report["metrics"], "ranking_sha256": reporting.digest(baseline)},
        "selected": {"metrics": selected_report["metrics"], "ranking_sha256": reporting.digest(staged)},
        "identifier-rerank": {"metrics": _metric_summary(test, reranked), "ranking_sha256": reporting.digest(reranked)},
    }
    actual = {
        "baseline": _actual_pipeline(
            test,
            catalog,
            routes,
            stages.StageConfig(configs["baseline"], candidate_window=100, rerank_window=50),
            baseline,
            rerank=False,
        ),
        "selected": _actual_pipeline(test, catalog, routes, config, staged, rerank=False),
        "identifier-rerank": _actual_pipeline(test, catalog, routes, config, reranked, rerank=True),
    }
    return {
        "schema": "lentica-hybrid-quality-v1",
        "selection_lock_sha256": selection["lock_sha256"],
        "dataset": {**test.provenance, "documents": len(test.documents), "queries": len(test.cases)},
        "environment": reporting.environment(),
        "stage_config": dataclasses.asdict(config),
        "comparisons": comparisons,
        "selected_diagnostics": diagnostics.candidate_report(
            test.cases, pools, staged, windows=windows, candidates=candidates
        ),
        "reranked_vs_selected": reporting.paired_comparison(test.cases, staged, reranked),
        "build": builds,
        "route_latency_ms": route_latency,
        "actual_pipeline": actual,
        "rankings": {"baseline": baseline, "selected": staged, "identifier-rerank": reranked},
        "ranking_sha256": reporting.digest({"baseline": baseline, "selected": staged, "identifier-rerank": reranked}),
        "test_exposure": selection["test_exposure"],
        "decision": "Fixed historical-test diagnostic control only; no retuning or default change.",
    }
