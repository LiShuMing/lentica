"""Fixed real-table structure/text/location controls and actual final PDF reads."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

from backend.app.evaluation.hybrid_eval import localization, pdf_controls, reporting
from backend.app.retrieval.hybrid import contracts, media, snapshots, stages, units, unruled_tables


def _controls(path: Path, region: unruled_tables.TableRegion) -> reporting.JsonObject:
    """Record native baselines, including malformed headers/blank rows, unchanged."""
    import fitz

    results = {}
    with fitz.open(path) as document:
        page = document[region.page - 1]
        x0, y0, x1, y1 = region.bbox
        clip = fitz.Rect(x0 * page.rect.width, y0 * page.rect.height, x1 * page.rect.width, y1 * page.rect.height)
        for strategy in ("lines_strict", "text"):
            started = time.perf_counter()
            tables = page.find_tables(strategy=strategy, clip=clip).tables
            results[strategy] = {
                "seconds": time.perf_counter() - started,
                "tables": [
                    {"rows": table.row_count, "columns": table.col_count, "values": table.extract()} for table in tables
                ],
            }
    return results


def _structure(result: unruled_tables.UnruledTable, reference: reporting.JsonObject) -> reporting.JsonObject:
    """Every reference slot is scored; shape mismatch does not reduce denominators."""
    cells = {(cell.row, cell.column): cell for cell in result.table.cells}
    rows, columns = reference["rows"], reference["columns"]
    cases, correct, overlaps, nonempty, correct_nonempty = [], 0, [], 0, 0
    for row in range(rows):
        for column in range(columns):
            expected_text = reference["text"][row][column]
            cell = cells.get((row, column))
            text_ok = cell is not None and " ".join(cell.text.split()) == expected_text
            expected = units.Locator(
                page=reference["page"],
                bbox=(
                    reference["column_edges"][column],
                    reference["row_edges"][row],
                    reference["column_edges"][column + 1],
                    reference["row_edges"][row + 1],
                ),
            )
            overlap = (
                localization.locator_iou(units.Locator(page=result.table.page, bbox=cell.bbox), expected)
                if cell
                else 0.0
            )
            correct += text_ok
            nonempty += bool(expected_text)
            correct_nonempty += bool(expected_text) and text_ok
            overlaps.append(overlap)
            cases.append(
                {
                    "row": row,
                    "column": column,
                    "expected": expected_text,
                    "actual": cell.text if cell else None,
                    "text_correct": text_ok,
                    "bbox_iou": overlap,
                }
            )
    return {
        "expected_slots": rows * columns,
        "actual_slots": len(cells),
        "correct_text_slots": correct,
        "text_accuracy": correct / (rows * columns),
        "nonempty_text_cells": nonempty,
        "correct_nonempty_text_cells": correct_nonempty,
        "mean_cell_iou": sum(overlaps) / (rows * columns),
        "shape_correct": (result.table.rows, result.table.columns) == (rows, columns),
        "table_bbox_iou": localization.locator_iou(
            units.Locator(page=result.table.page, bbox=result.table.bbox),
            units.Locator(page=reference["page"], bbox=tuple(reference["table_bbox"])),
        ),
        "opaque_glyphs": len(result.glyphs),
        "expected_opaque_glyphs": reference["opaque_glyphs"],
        "cells": cases,
    }


def _validate_reference(reference: reporting.JsonObject) -> None:
    """Bound structural labels before scoring; invalid input cannot shrink totals."""
    try:
        rows, columns = reference["rows"], reference["columns"]
        if (
            type(rows) is not int
            or type(columns) is not int
            or not 2 <= rows <= 100
            or not 2 <= columns <= 50
            or rows * columns > 5000
        ):
            raise ValueError("Unruled reference dimensions exceed bounds")
        values = reference["text"]
        if (
            len(values) != rows
            or any(len(row) != columns for row in values)
            or any(not isinstance(text, str) or len(text) > 2048 for row in values for text in row)
            or reference["queries"] != [row[0] for row in values[1:]]
            or any(not query.strip() for query in reference["queries"])
        ):
            raise ValueError("Unruled reference text/query alignment differs")
        for key, count in (("row_edges", rows + 1), ("column_edges", columns + 1)):
            edges = reference[key]
            if (
                len(edges) != count
                or any(type(value) not in (int, float) or not 0 <= value <= 1 for value in edges)
                or any(left >= right for left, right in zip(edges, edges[1:]))
            ):
                raise ValueError("Unruled reference edges must be bounded and increasing")
        units.Locator(page=reference["page"], bbox=tuple(reference["table_bbox"]))
        if type(reference["opaque_glyphs"]) is not int or not 0 <= reference["opaque_glyphs"] <= 20000:
            raise ValueError("Unruled reference glyph count exceeds bounds")
    except (KeyError, TypeError) as error:
        raise ValueError("Malformed unruled reference fields") from error


def _retrieval(
    result: unruled_tables.UnruledTable, source: Path, reference: reporting.JsonObject
) -> reporting.JsonObject:
    """Compose native cell units with prefilters and exactly one actual PDF read."""
    catalog = units.UnitCatalog(result.units)
    retriever = media.EvidenceTextRetriever(catalog, frozenset({"native"}))
    store = pdf_controls.PublicPdfStore(source, reference["source_sha256"])
    search = stages.StagedSearch({"native": retriever}, catalog, store=store)
    config = stages.StageConfig(
        contracts.SearchConfig((contracts.Branch("native", 20),), top_k=1),
        candidate_window=20,
        rerank_window=10,
        max_units_per_asset=1,
    )
    cells = {(cell.row, cell.column): cell for cell in result.table.cells}
    rows = []
    for index, query in enumerate(reference["queries"]):
        expected = cells.get((index + 1, 0))
        target = expected.unit_id if expected and expected.text == query else None
        request = contracts.SearchQuery(query, contracts.SearchFilter(equals={"source": "public-survey"}))
        ranked = tuple(retriever.retrieve(request, 10))
        before = store.calls
        found = search.search(request, config)
        wrong = search.search(contracts.SearchQuery(query, contracts.SearchFilter(equals={"source": "other"})), config)
        first = found.hits[0].evidence[0].unit if found.hits else None
        expected_location = units.Locator(
            page=reference["page"],
            bbox=(
                reference["column_edges"][0],
                reference["row_edges"][index + 1],
                reference["column_edges"][1],
                reference["row_edges"][index + 2],
            ),
        )
        rows.append(
            {
                "query": query,
                "expected_unit": target,
                "ranked_units": ranked,
                "unit_recall_at_10": int(target is not None and target in ranked),
                "resource_recall_at_1": int(bool(found.hits) and found.hits[0].asset_id == 42),
                "unit_hit_at_1": int(first is not None and target is not None and first.unit_id == target),
                "evidence_bbox_iou": localization.locator_iou(first.locator, expected_location) if first else 0.0,
                "final_pdf_reads": store.calls - before,
                "wrong_filter_hits": len(wrong.hits),
            }
        )
    glyph_queries = []
    for query in sorted({glyph.raw_decode for glyph in result.glyphs}):
        before = store.calls
        found = search.search(contracts.SearchQuery(query), config)
        glyph_queries.append({"query": query, "hits": len(found.hits), "final_pdf_reads": store.calls - before})
    return {
        "queries": len(rows),
        "unit_recall_at_10": sum(row["unit_recall_at_10"] for row in rows) / len(rows),
        "resource_recall_at_1": sum(row["resource_recall_at_1"] for row in rows) / len(rows),
        "unit_hit_at_1": sum(row["unit_hit_at_1"] for row in rows) / len(rows),
        "mean_evidence_bbox_iou": sum(row["evidence_bbox_iou"] for row in rows) / len(rows),
        "actual_final_pdf_reads": store.calls,
        "actual_final_rows": store.rows,
        "per_query": rows,
        "opaque_query_controls": glyph_queries,
    }


def evaluate(path: Path, reference: reporting.JsonObject) -> reporting.JsonObject:
    """Run frozen public-region controls without changing references after failure."""
    if (
        reference["schema"] != "lentica-unruled-reference-v1"
        or Path(path).stat().st_size > 16 * 1024**2
        or snapshots.sha256_file(path) != reference["source_sha256"]
    ):
        raise ValueError("Public unruled source/reference identity differs")
    _validate_reference(reference)
    region = unruled_tables.TableRegion(reference["page"], tuple(reference["region"]))
    controls = _controls(path, region)
    started = time.perf_counter()
    result = unruled_tables.extract_region(
        path, region, asset_id=42, first_unit_id=10000, metadata_values={"source": "public-survey"}
    )
    extraction_seconds = time.perf_counter() - started
    structure = _structure(result, reference)
    retrieval = _retrieval(result, path, reference)
    source_unchanged = snapshots.sha256_file(path) == reference["source_sha256"]
    payload = dataclasses.asdict(result.table)
    payload["glyphs"] = [dataclasses.asdict(glyph) for glyph in result.glyphs]
    accepted = (
        structure["shape_correct"]
        and structure["text_accuracy"] == 1
        and structure["mean_cell_iou"] >= 0.5
        and structure["table_bbox_iou"] >= 0.5
        and structure["opaque_glyphs"] == reference["opaque_glyphs"]
        and retrieval["unit_recall_at_10"] == 1
        and retrieval["resource_recall_at_1"] == 1
        and retrieval["unit_hit_at_1"] == 1
        and retrieval["mean_evidence_bbox_iou"] >= 0.5
        and source_unchanged
        and all(row["hits"] == 0 and row["final_pdf_reads"] == 0 for row in retrieval["opaque_query_controls"])
        and all(row["final_pdf_reads"] == 1 and row["wrong_filter_hits"] == 0 for row in retrieval["per_query"])
    )
    return {
        "schema": "lentica-unruled-quality-v1",
        "source_sha256": reference["source_sha256"],
        "reference_sha256": reporting.digest(reference),
        "producer_version": result.producer_version,
        "controls": controls,
        "structure": structure,
        "retrieval": retrieval,
        "source_unchanged": source_unchanged,
        "extraction_seconds": extraction_seconds,
        "accepted": accepted,
        "table": payload,
        "result_sha256": reporting.digest(payload),
        "scope": "One previously inspected public native table region; geometric development control, no glyph state, image-table/discovery/generalization accuracy claim.",
    }
