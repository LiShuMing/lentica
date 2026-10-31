"""Reference-only scoring of explicit typed native-table row/answer bindings."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

from backend.app.evaluation.hybrid_eval import localization, pdf_controls, reporting
from backend.app.retrieval.hybrid import contracts, media, snapshots, stages, table_context, units, unruled_tables


def _validate(reference: reporting.JsonObject) -> None:
    """Reject malformed references; every declared row/field stays in denominators."""
    try:
        columns, labels, rows, queries = (reference[key] for key in ("columns", "headers", "rows", "queries"))
        if (
            not 3 <= len(columns) <= 50
            or len(labels) != len(columns)
            or len(set(columns)) != len(columns)
            or not 1 <= len(rows) <= 1000
            or len(rows) * len(columns) > 5000
            or any(len(row) != len(columns) for row in rows)
            or any(not isinstance(text, str) or len(text) > 2048 for row in rows for text in row)
            or not 1 <= len(queries) <= 1000
        ):
            raise ValueError("Table reference row/field/query dimensions differ")
        for key, count in (("row_edges", len(rows) + 1), ("column_edges", len(columns) + 1)):
            edges = reference[key]
            if (
                len(edges) != count
                or any(type(value) not in (int, float) or not 0 <= value <= 1 for value in edges)
                or any(left >= right for left, right in zip(edges, edges[1:]))
            ):
                raise ValueError("Table reference edges must be bounded and increasing")
        for query in queries:
            if (
                type(query["row"]) is not int
                or not 0 <= query["row"] < len(rows)
                or query["field"] not in columns
                or query["answer"] != rows[query["row"]][columns.index(query["field"])]
            ):
                raise ValueError("Table reference query/answer alignment differs")
    except (KeyError, TypeError) as error:
        raise ValueError("Malformed table reference") from error


def _structure(context: table_context.TableContext, reference: reporting.JsonObject) -> reporting.JsonObject:
    """Keep all expected expanded values, including missing rows/fields."""
    cases, correct = [], 0
    for index, expected in enumerate(reference["rows"]):
        actual = [cell.value for cell in context.rows[index].cells] if index < len(context.rows) else []
        matches = sum(column < len(actual) and value == actual[column] for column, value in enumerate(expected))
        correct += matches
        cases.append({"row": index, "expected": expected, "actual": actual, "correct_fields": matches})
    spans = {
        (cell.source_row, cell.column, cell.source_bbox)
        for row in context.rows
        for cell in row.cells
        if cell.method == "geometric-span"
    }
    slots = len(reference["rows"]) * len(reference["columns"])
    return {
        "expected_data_rows": len(reference["rows"]),
        "actual_data_rows": len(context.rows),
        "native_coarse_rows_including_headers": len(context.native_values),
        "expected_slots": slots,
        "actual_slots": sum(len(row.cells) for row in context.rows),
        "correct_values": correct,
        "value_accuracy": correct / slots,
        "geometric_span_source_cells": len(spans),
        "per_row": cases,
    }


def _expected_location(reference, row, column):
    return units.Locator(
        page=reference["page"],
        bbox=(
            reference["column_edges"][column],
            reference["row_edges"][row],
            reference["column_edges"][column + 1],
            reference["row_edges"][row + 1],
        ),
    )


def _retrieval(
    context: table_context.TableContext, path: Path, reference: reporting.JsonObject
) -> reporting.JsonObject:
    """Apply explicit first-three-field predicates before Top1 and real PDF reading."""
    catalog = units.UnitCatalog(context.units)
    retriever = media.EvidenceTextRetriever(catalog, frozenset({"native"}))
    store = pdf_controls.PublicPdfStore(path, reference["source_sha256"])
    searcher = stages.StagedSearch({"table": retriever}, catalog, store=store)
    config = stages.StageConfig(
        contracts.SearchConfig((contracts.Branch("table", 1),), top_k=1),
        candidate_window=1,
        rerank_window=1,
        max_units_per_asset=1,
    )
    row_by_unit = {row.unit_id: row for row in context.rows}
    cases = []
    for query in reference["queries"]:
        index, column = query["row"], reference["columns"].index(query["field"])
        conditions = {
            table_context.FIELD_PREFIX + key: value
            for key, value in zip(reference["columns"][:3], reference["rows"][index][:3])
        }
        request = contracts.SearchQuery(
            reference["headers"][column], contracts.SearchFilter(equals={"source": "public-table", **conditions})
        )
        before = store.calls
        result = searcher.search(request, config)
        found = row_by_unit[result.ordered_unit_ids[0]] if result.ordered_unit_ids else None
        answer = found.cells[column] if found else None
        wrong_source = searcher.search(
            contracts.SearchQuery(request.text, contracts.SearchFilter(equals={**conditions, "source": "other"})),
            config,
        )
        missing = searcher.search(
            contracts.SearchQuery(
                request.text,
                contracts.SearchFilter(equals={**request.filters.equals, table_context.FIELD_PREFIX + "missing": "x"}),
            ),
            config,
        )
        cases.append(
            {
                "conditions": conditions,
                "field": query["field"],
                "expected_answer": query["answer"],
                "actual_answer": answer.value if answer else None,
                "answer_correct": bool(answer and answer.value == query["answer"]),
                "expected_row": index,
                "actual_row": found.row if found else None,
                "row_hit_at_1": int(bool(found and found.row == index)),
                "answer_bbox_iou": localization.locator_iou(
                    units.Locator(page=reference["page"], bbox=answer.bbox),
                    _expected_location(reference, index, column),
                )
                if answer
                else 0.0,
                "answer_bbox": answer.bbox if answer else None,
                "source_bbox": answer.source_bbox if answer else None,
                "final_pdf_reads": store.calls - before,
                "wrong_source_hits": len(wrong_source.hits),
                "missing_field_hits": len(missing.hits),
            }
        )
    return {
        "queries": len(cases),
        "row_hit_at_1": sum(case["row_hit_at_1"] for case in cases) / len(cases),
        "answer_accuracy": sum(case["answer_correct"] for case in cases) / len(cases),
        "mean_answer_bbox_iou": sum(case["answer_bbox_iou"] for case in cases) / len(cases),
        "actual_final_pdf_reads": store.calls,
        "per_query": cases,
    }


def evaluate(path: Path, reference: reporting.JsonObject, *, header_rows: int) -> reporting.JsonObject:
    """Extract without answers, then score frozen values and typed query conditions.

    This is a numeric-table development control, not a free-language question
    answering model. The source and caller field schema are explicit; expected
    data values, query conditions and answers reach only scoring/retrieval.
    """
    if (
        reference["schema"] != "lentica-merged-table-reference-v1"
        or Path(path).stat().st_size > 16 * 1024**2
        or snapshots.sha256_file(path) != reference["source_sha256"]
    ):
        raise ValueError("Public table source/reference identity differs")
    _validate(reference)
    columns = tuple(
        table_context.TableColumn(key, label) for key, label in zip(reference["columns"], reference["headers"])
    )
    started = time.perf_counter()
    context = table_context.extract_rows(
        path,
        unruled_tables.TableRegion(reference["page"], tuple(reference["region"])),
        columns,
        header_rows=header_rows,
        asset_id=42,
        first_unit_id=10000,
        metadata_values={"source": "public-table"},
    )
    elapsed = time.perf_counter() - started
    structure = _structure(context, reference)
    retrieval = _retrieval(context, path, reference)
    payload = {"rows": [dataclasses.asdict(row) for row in context.rows], "native_values": context.native_values}
    source_unchanged = snapshots.sha256_file(path) == reference["source_sha256"]
    accepted = (
        structure["value_accuracy"] == 1
        and structure["actual_data_rows"] == structure["expected_data_rows"]
        and structure["actual_slots"] == structure["expected_slots"]
        and retrieval["row_hit_at_1"] == 1
        and retrieval["answer_accuracy"] == 1
        and retrieval["mean_answer_bbox_iou"] >= 0.5
        and source_unchanged
        and all(
            case["final_pdf_reads"] == 1 and case["wrong_source_hits"] == case["missing_field_hits"] == 0
            for case in retrieval["per_query"]
        )
    )
    return {
        "schema": "lentica-table-context-quality-v1",
        "source_sha256": reference["source_sha256"],
        "reference_sha256": reporting.digest(reference),
        "producer_version": context.producer_version,
        "header_rows": header_rows,
        "structure": structure,
        "retrieval": retrieval,
        "extraction_seconds": elapsed,
        "source_unchanged": source_unchanged,
        "accepted": accepted,
        "table": payload,
        "result_sha256": reporting.digest({"table": payload, "queries": retrieval["per_query"]}),
        "scope": "One visually frozen independent public numeric table development control; explicit field schema/typed filters and geometric spans, not held-out, free-language QA or unruled/image-table accuracy.",
    }
