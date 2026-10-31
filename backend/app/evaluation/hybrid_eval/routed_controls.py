"""Fixed capability/family controls over generated media; no model claims."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from backend.app.evaluation import dataset as evaluation_dataset
from backend.app.evaluation import metrics
from backend.app.evaluation.hybrid_eval import localization, media_fixture, partitions, reporting
from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import media, querying, reranking, routing, stages


@dataclass(frozen=True)
class ControlCase:
    """Generated conditions, never supplied to route decisions or ranking."""

    query_id: str
    query: querying.MultimodalQuery
    category: str
    group: int
    judgment: localization.Judgment


def _cases(fixture: media_fixture.MediaFixture) -> list[ControlCase]:
    result = []
    for case in fixture.cases:
        locations = {
            fixture.catalog.units[identifier].asset_id: (fixture.catalog.units[identifier].locator,)
            for identifier in case.relevant_units
            if fixture.catalog.units[identifier].kind == "page"
        }
        result.append(
            ControlCase(
                case.query_id,
                querying.MultimodalQuery(case.query.text, filters=case.query.filters),
                case.category,
                case.query.filters.equals["group"],
                localization.Judgment(case.relevant_assets, case.relevant_units, locations),
            )
        )
    for base in [item for item in result if item.category == "combined"]:
        target = next(iter(base.judgment.unit_ids))
        color = next(item for item in result if item.group == base.group and item.category == "visual_color")
        result.extend(
            (
                ControlCase(
                    f"g{base.group}-image",
                    querying.MultimodalQuery(image=fixture.images[target], filters=base.query.filters),
                    "image",
                    base.group,
                    color.judgment,
                ),
                ControlCase(
                    f"g{base.group}-image-text",
                    dataclasses.replace(base.query, image=fixture.images[target]),
                    "image_text",
                    base.group,
                    base.judgment,
                ),
                ControlCase(
                    f"g{base.group}-no-answer",
                    querying.MultimodalQuery("application screen ERR_999999", filters=base.query.filters),
                    "no_answer",
                    base.group,
                    localization.Judgment(frozenset(), frozenset()),
                ),
            )
        )
    return result


def _summarize(rows: Sequence[reporting.JsonObject]) -> reporting.JsonObject:
    values = {key for row in rows for key in row["metrics"]}
    summary = {"queries": len(rows), "positive_queries": sum(row["answerable"] for row in rows)}
    for key in sorted(values):
        eligible = [row["metrics"][key] for row in rows if row["metrics"].get(key) is not None]
        summary[key] = sum(eligible) / len(eligible) if eligible else None
    return summary


def _measure(
    fixture: media_fixture.MediaFixture,
    cases: Sequence[ControlCase],
    *,
    routed: bool,
    grouped_families: bool,
    rerank: bool,
) -> reporting.JsonObject:
    text = media.EvidenceTextRetriever(fixture.catalog, frozenset({"native", "ocr", "transcript"}))
    caption = media.EvidenceTextRetriever(fixture.catalog, frozenset({"caption"}))
    pixel = media.ColorHistogramRetriever(fixture.catalog, fixture.images)
    all_text = frozenset({"text", "identifier", "color", "combined"})
    # Color-only requests are precisely the six supported color names. General
    # prose keeps text routes; this is a fixed limited capability policy.
    caption_kinds = all_text - {"color"} if routed else all_text
    specs = (
        querying.RouteSpec(hybrid.Branch("native", 50), all_text - {"color"} if routed else all_text),
        querying.RouteSpec(hybrid.Branch("caption", 50), caption_kinds, "generated" if grouped_families else None),
        querying.RouteSpec(hybrid.Branch("tags", 50), caption_kinds, "generated" if grouped_families else None),
        querying.RouteSpec(hybrid.Branch("color", 50), all_text, "pixels"),
        querying.RouteSpec(hybrid.Branch("image", 50), frozenset({"image", "combined"}), "pixels"),
    )
    providers = {
        "native": routing.TextRoute(text),
        "caption": routing.TextRoute(caption),
        # Deliberately cloned caption evidence tests that family caps, rather
        # than arbitrary weight tuning, prevent duplicate votes.
        "tags": routing.TextRoute(caption),
        "color": routing.TextRoute(pixel),
        "image": media.ImageHistogramRoute(pixel),
    }
    store = reporting.CountingStore(fixture.assets)
    searcher = routing.RoutedSearch(
        providers,
        fixture.catalog,
        routes=specs,
        store=store,
        reranker=reranking.ExactIdentifierReranker() if rerank else None,
    )
    config = stages.StageConfig(hybrid.SearchConfig(tuple(spec.branch for spec in specs), rrf_constant=1))
    rows, latencies = [], []
    for case in cases:
        started = time.perf_counter()
        result = searcher.search(case.query, config)
        latencies.append(time.perf_counter() - started)
        hits = result.result.hits if result.result is not None else ()
        predicted = [hit.evidence[0].unit for hit in hits]
        identifiers = [unit.asset_id for unit in predicted]
        values = localization.score(case.judgment, predicted)
        if case.judgment.assets:
            judged = evaluation_dataset.EvaluationCase(
                case.query_id, case.query.text, set(case.judgment.assets), case.category
            )
            values.update(metrics.metrics_for_case(judged, identifiers))
        rows.append(
            {
                "query_id": case.query_id,
                "category": case.category,
                "answerable": bool(case.judgment.assets),
                "asset_ids": identifiers,
                "best_unit_ids": [unit.unit_id for unit in predicted],
                "locators": [dataclasses.asdict(unit.locator) for unit in predicted],
                "metrics": values,
                "plan": [
                    {"route": item.route.branch.name, "window": item.window, "status": item.status}
                    for item in result.plan
                ],
                "candidate_units": result.result.feature_units if result.result is not None else 0,
            }
        )
    return {
        "metrics": _summarize(rows),
        "by_category": {
            category: _summarize([row for row in rows if row["category"] == category])
            for category in sorted({row["category"] for row in rows})
        },
        "per_query": rows,
        "ranking_sha256": reporting.digest(rows),
        "latency_ms": {"p50": reporting.percentile(latencies, 0.5), "p95": reporting.percentile(latencies, 0.95)},
        "final_fetch": {"calls": store.calls, "records": store.rows, "max_per_call": store.max_rows},
    }


def run(directory: Path, *, groups: int = 12) -> reporting.JsonObject:
    """Run predetermined controls, retaining difficult no-answer failures.

    Images/native PDF text are computed; OCR/captions are supplied annotations.
    Partitions demonstrate source isolation only. No parameters are selected on
    these fixtures and none of these numbers certify unseen model quality.
    """
    if groups < 3:
        raise ValueError("At least three source groups are required")
    fixture = media_fixture.generate(directory, groups=groups)
    cases = _cases(fixture)
    split = partitions.source_partitions(
        [partitions.SourceItem(case.query_id, frozenset({f"generated-source-{case.group}"})) for case in cases]
    )
    comparisons = {
        name: _measure(fixture, cases, routed=routed, grouped_families=family, rerank=rerank)
        for name, routed, family, rerank in (
            ("duplicated-evidence", False, False, False),
            ("family-capped", False, True, False),
            ("capability-routed", True, True, False),
            ("routed-identifier", True, True, True),
        )
    }
    return {
        "schema": "lentica-routed-controls-v1",
        "scope": "Generated mechanism controls; actual image queries/pixels/native PDF; supplied synthetic OCR/captions. "
        "Fixed narrow routing, no semantic model or unseen-test claim; no-answer failures retained.",
        "dataset": {
            "groups": groups,
            "queries": len(cases),
            "assets": len(fixture.assets),
            "units": len(fixture.catalog.units),
        },
        "source_partitions": split,
        "partition_sha256": reporting.digest(split),
        "comparisons": comparisons,
        "ranking_sha256": reporting.digest({name: row["ranking_sha256"] for name, row in comparisons.items()}),
        "environment": reporting.environment(),
        "decision": "Opt-in mechanism evidence only; default unchanged; no answer threshold has been trained or selected.",
    }
