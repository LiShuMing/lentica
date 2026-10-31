"""Controlled multimodal retrieval ablations; no pretrained-model quality claim."""

from __future__ import annotations

import dataclasses
import hashlib
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from backend.app.evaluation import dataset as evaluation_dataset
from backend.app.evaluation import metrics
from backend.app.evaluation.hybrid_eval import media_fixture, reporting
from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import media, reranking, stages


def _measure(
    fixture: media_fixture.MediaFixture,
    routes: Mapping[str, hybrid.Retriever],
    *,
    grouped: bool,
    rerank: bool,
) -> reporting.JsonObject:
    search_config = hybrid.SearchConfig(tuple(hybrid.Branch(name, 50) for name in routes), rrf_constant=1)
    store = reporting.CountingStore(fixture.assets)
    pipeline = stages.StagedSearch(
        routes,
        fixture.catalog,
        store=store,
        reranker=reranking.ExactIdentifierReranker() if rerank else None,
    )
    flat = hybrid.HybridSearch(routes)
    config = stages.StageConfig(search_config, candidate_window=100, rerank_window=50, max_units_per_asset=2)
    rows, elapsed = [], []
    for case in fixture.cases:
        started = time.perf_counter()
        if grouped:
            result = pipeline.search(case.query, config)
            asset_ids = [hit.asset_id for hit in result.hits]
            unit_ids = [hit.evidence[0].unit.unit_id for hit in result.hits]
            evidence_ids = [evidence.unit.unit_id for hit in result.hits for evidence in hit.evidence]
            candidate_ids = [hit.asset_id for hit in result.candidates]
            feature_units = result.feature_units
            reranked_units = result.reranked_units
        else:
            result = flat.search(case.query, search_config)
            unit_ids = [hit.asset_id for hit in result.hits]
            asset_ids = [fixture.catalog.units[identifier].asset_id for identifier in unit_ids]
            evidence_ids, candidate_ids = unit_ids, unit_ids
            feature_units, reranked_units = 0, 0
        elapsed.append(time.perf_counter() - started)
        judged = evaluation_dataset.EvaluationCase(
            case.query_id, case.query.text, set(case.relevant_assets), case.category
        )
        values = metrics.metrics_for_case(judged, asset_ids)
        rows.append(
            {
                "query_id": case.query_id,
                "category": case.category,
                "asset_ids": asset_ids,
                "best_unit_ids": unit_ids,
                "locators": [dataclasses.asdict(fixture.catalog.units[identifier].locator) for identifier in unit_ids],
                "relevant_assets": sorted(case.relevant_assets),
                "relevant_units": sorted(case.relevant_units),
                "candidate_units": len(candidate_ids),
                "feature_units": feature_units,
                "reranked_units": reranked_units,
                "duplicate_fraction": 1 - len(set(asset_ids)) / len(asset_ids) if asset_ids else 0.0,
                "best_unit_hit_at_10": float(bool(case.relevant_units & set(unit_ids[:10]))),
                "evidence_hit_at_10_assets": float(bool(case.relevant_units & set(evidence_ids))),
                "metrics": values,
            }
        )

    def summarize(selected: Sequence[reporting.JsonObject]) -> reporting.JsonObject:
        return {
            "queries": len(selected),
            **{
                key: sum(row["metrics"][key] for row in selected) / len(selected)
                for key in ("recall_at_10", "ndcg_at_10", "mrr")
            },
            **{
                key: sum(row[key] for row in selected) / len(selected)
                for key in ("duplicate_fraction", "best_unit_hit_at_10", "evidence_hit_at_10_assets")
            },
        }

    return {
        "metrics": summarize(rows),
        "by_category": {
            category: summarize([row for row in rows if row["category"] == category])
            for category in sorted({row["category"] for row in rows})
        },
        "latency_ms": {"p50": reporting.percentile(elapsed, 0.5), "p95": reporting.percentile(elapsed, 0.95)},
        "final_fetch": {"calls": store.calls, "records": store.rows, "max_per_call": store.max_rows},
        "ranking_sha256": reporting.digest(rows),
        "per_query": rows,
    }


def run(directory: Path, *, groups: int = 12) -> reporting.JsonObject:
    """Generate media and run four fixed controls without tuning on their labels.

    Args:
        directory: Fresh output directory for generated PNG/PDF controls.
        groups: Number of independent generated groups, bounded by the generator.

    Returns:
        Quality, per-query evidence locations, timings and exact local identities.
        Synthetic labels describe generated conditions, not generalization.
    """
    fixture = media_fixture.generate(directory, groups=groups)
    started = time.perf_counter()
    flat = media.EvidenceTextRetriever(fixture.catalog, frozenset({"native", "ocr", "caption", "transcript"}))
    routes = {
        "text": media.EvidenceTextRetriever(fixture.catalog, frozenset({"native", "ocr", "transcript"})),
        "caption": media.EvidenceTextRetriever(fixture.catalog, frozenset({"caption"})),
        "color": media.ColorHistogramRetriever(fixture.catalog, fixture.images),
    }
    build_seconds = time.perf_counter() - started
    comparisons = {
        "flat-text": _measure(fixture, {"flat": flat}, grouped=False, rerank=False),
        "text-parent": _measure(fixture, {"flat": flat}, grouped=True, rerank=False),
        "multiroute": _measure(fixture, routes, grouped=True, rerank=False),
        "multiroute-identifier": _measure(fixture, routes, grouped=True, rerank=True),
    }
    generated = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.suffix in {".png", ".pdf"}
    }
    return {
        "schema": "lentica-multimodal-controls-v1",
        "scope": "Generated image/PDF controls, actual image pixels/native PDF extraction; supplied synthetic OCR/captions; "
        "no pretrained vision, semantic or OCR model and no unseen-test claim.",
        "dataset": {
            "groups": groups,
            "assets": len(fixture.assets),
            "units": len(fixture.catalog.units),
            "images": len(fixture.images),
            "queries": len(fixture.cases),
            "generated_files_sha256": reporting.digest(generated),
        },
        "environment": {**reporting.environment(), "pillow": _version("Pillow"), "pymupdf": _version("PyMuPDF")},
        "fixed_config": {
            "rrf_constant": 1,
            "branch_window": 50,
            "candidate_window": 100,
            "rerank_window": 50,
            "top_k": 10,
            "max_units_per_asset": 2,
            "color_encoder_id": media.ColorHistogramRetriever.encoder_id,
        },
        "build_seconds": build_seconds,
        "comparisons": comparisons,
        "ranking_sha256": reporting.digest({name: row["ranking_sha256"] for name, row in comparisons.items()}),
        "latency_scope": "Serial in-process retrieval/ranking; grouped controls include candidate feature access and "
        "final in-memory asset fetch. Flat control is ranking-only; excludes generation/index build and disk serving.",
        "decision": "Mechanism controls only; no product default or pretrained-quality conclusion.",
    }


def _version(name: str) -> str:
    from importlib import metadata

    return metadata.version(name)
