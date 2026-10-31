"""Actual PDF merged-row controls; generated fixtures are not public accuracy."""

from __future__ import annotations

import hashlib
import subprocess
import sys

import pytest

from backend.app.evaluation.hybrid_eval import pdf_controls, table_quality
from backend.app.retrieval.hybrid import contracts, media, table_context, units, unruled_tables

COLUMNS = tuple(
    table_context.TableColumn(key, label)
    for key, label in zip(
        ("encoder", "decoder", "language", "score_a", "score_b", "seconds"),
        ("Encoder", "Decoder", "Language", "Score A", "Score B", "Time seconds"),
    )
)
EXPECTED = (
    ("6", "6", "OTSL", "0.9", "0.8", "2.73"),
    ("6", "6", "HTML", "0.8", "0.7", "5.39"),
    ("4", "2", "OTSL", "0.7", "0.6", "1.22"),
    ("4", "2", "HTML", "0.6", "0.5", "2"),
)


def _pdf(path, *, ambiguous=False, symbol=False):
    import fitz

    regions = []
    with fitz.open() as document:
        for rotation in (0, 90, 180, 270):
            page = document.new_page(width=520, height=180)
            xs = (20, 95, 170, 245, 320, 395, 470)
            for x in xs:
                page.draw_line((x, 20), (x, 120))
            for y in (20, 40, 80, 120):
                page.draw_line((20, y), (470, y))
            for x, column in zip(xs, COLUMNS):
                page.insert_text((x + 5, 33), column.label, fontsize=8)
            for group, (encoder, decoder) in enumerate((("6", "6"), ("4", "2"))):
                offset = group * 40
                page.insert_text((25, 63 + offset), encoder, fontsize=9)
                page.insert_text((100, 63 + offset), decoder, fontsize=9)
                for row in (0, 1):
                    for column in range(2, 6):
                        if ambiguous and group == 0 and column == 3 and row == 1:
                            continue
                        page.insert_text(
                            (xs[column] + 5, 53 + offset + row * 20), EXPECTED[group * 2 + row][column], fontsize=9
                        )
            if symbol:
                page.insert_text((465, 33), "o", fontname="ZaDb", fontsize=8)
            page.set_rotation(rotation)
            rect = fitz.Rect(18, 18, 472, 122) * page.rotation_matrix
            regions.append(
                unruled_tables.TableRegion(
                    page.number + 1,
                    (
                        rect.x0 / page.rect.width,
                        rect.y0 / page.rect.height,
                        rect.x1 / page.rect.width,
                        rect.y1 / page.rect.height,
                    ),
                )
            )
        document.save(path)
    return regions


def test_native_merged_rows_keep_binding_provenance_and_all_rotated_locations(tmp_path):
    path = tmp_path / "numeric.pdf"
    regions = _pdf(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for page, region in enumerate(regions, start=1):
        context = table_context.extract_rows(
            path, region, COLUMNS, header_rows=1, asset_id=42, first_unit_id=100, metadata_values={"tenant": "public"}
        )
        assert len(context.native_values) == 3 and context.native_values[1][2] == "OTSL\nHTML"
        assert tuple(tuple(cell.value for cell in row.cells) for row in context.rows) == EXPECTED
        assert (
            len({(cell.source_row, cell.column) for row in context.rows for cell in row.cells if cell.span_rows}) == 4
        )
        assert context.rows[2].cells[0].span_rows == (2, 3)
        assert context.rows[2].cells[0].method == "geometric-span"
        assert context.rows[2].cells[0].source_text == "4"
        assert context.rows[2].cells[5].source_text == "1.22\n2"
        assert all(unit.locator.page == page and unit.metadata["tenant"] == "public" for unit in context.units)
        retriever = media.EvidenceTextRetriever(units.UnitCatalog(context.units), frozenset({"native"}))
        request = contracts.SearchQuery(
            "Time seconds",
            contracts.SearchFilter(
                equals={"table.field.encoder": "4", "table.field.decoder": "2", "table.field.language": "OTSL"}
            ),
        )
        assert retriever.retrieve(request, 1) == (102,)
        assert context.rows[2].cells[5].value == "1.22"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_ambiguous_sparse_cells_and_symbol_encodings_fail_instead_of_inheriting_fake_values(tmp_path):
    for name, options, message in (
        ("ambiguous", {"ambiguous": True}, "Ambiguous"),
        ("symbol", {"symbol": True}, "Opaque symbol"),
    ):
        path = tmp_path / f"{name}.pdf"
        region = _pdf(path, **options)[0]
        with pytest.raises(ValueError, match=message):
            table_context.extract_rows(path, region, COLUMNS, header_rows=1, asset_id=42, first_unit_id=100)


def test_budgets_schema_and_reserved_filter_namespace_are_checked(tmp_path):
    path = tmp_path / "numeric.pdf"
    region = _pdf(path)[0]
    options = {"header_rows": 1, "asset_id": 42, "first_unit_id": 100}
    for change in (
        {"max_words": 1},
        {"max_cells": 20},
        {"max_rows": 3},
        {"header_rows": 3},
        {"metadata_values": {"table.field.encoder": "caller-injected"}},
    ):
        with pytest.raises(ValueError):
            table_context.extract_rows(path, region, COLUMNS, **{**options, **change})
    with pytest.raises(ValueError, match="fields"):
        table_context.extract_rows(path, region, COLUMNS[:-1], **options)
    with pytest.raises(ValueError):
        table_context.TableColumn("unsafe.key", "Field")


def test_reference_scoring_typed_top1_filters_and_actual_final_reads(tmp_path):
    path = tmp_path / "numeric.pdf"
    region = _pdf(path)[0]
    reference = {
        "schema": "lentica-merged-table-reference-v1",
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "page": 1,
        "region": region.bbox,
        "columns": [column.key for column in COLUMNS],
        "headers": [column.label for column in COLUMNS],
        "rows": EXPECTED,
        "row_edges": [40 / 180, 60 / 180, 80 / 180, 100 / 180, 120 / 180],
        "column_edges": [x / 520 for x in (20, 95, 170, 245, 320, 395, 470)],
        "queries": [{"row": row, "field": "seconds", "answer": values[-1]} for row, values in enumerate(EXPECTED)],
    }
    result = table_quality.evaluate(path, reference, header_rows=1)
    assert result["accepted"] and result["structure"]["correct_values"] == 24
    assert result["retrieval"]["actual_final_pdf_reads"] == 4
    assert result["retrieval"]["answer_accuracy"] == result["retrieval"]["row_hit_at_1"] == 1
    with pytest.raises(ValueError, match="alignment"):
        table_quality.evaluate(
            path, {**reference, "queries": [{"row": 0, "field": "seconds", "answer": "wrong"}]}, header_rows=1
        )
    store = pdf_controls.PublicPdfStore(path, reference["source_sha256"])
    path.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="changed"):
        store.fetch([42])
    assert store.calls == 0


def test_imports_and_help_need_no_optional_runtime():
    code = """
import sys
from backend.app.retrieval.hybrid import table_context
from backend.app.evaluation.hybrid_eval import table_quality
assert not any(name.startswith(('fitz','numpy','torch','transformers','PIL')) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-S", "-c", code], check=True, timeout=10)
    result = subprocess.run(
        [sys.executable, "-S", "-m", "backend.app.evaluation.table_benchmark", "--help"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert "--header-rows" in result.stdout
