"""Actual native PDF controls for table geometry/glyphs, not broad table accuracy."""

from __future__ import annotations

import hashlib
import subprocess
import sys

import pytest

from backend.app.evaluation.hybrid_eval import unruled_quality
from backend.app.retrieval.hybrid import unruled_tables


def _pdf(path):
    import fitz

    regions = []
    with fitz.open() as document:
        for rotation in (0, 90, 180, 270):
            page = document.new_page(width=600, height=400)
            for y, values in [
                (60, ("Category", "Value", "Status")),
                (100, ("Alpha", "10", None)),
                (140, ("Beta", "20", None)),
            ]:
                for x, text in zip((50, 260, 410), values):
                    if text is None:
                        page.insert_text((x, y), "o", fontname="ZaDb", fontsize=12)
                    else:
                        page.insert_text((x, y), text, fontsize=12)
            page.set_rotation(rotation)
            displayed = fitz.Rect(30, 20, 550, 200) * page.rotation_matrix
            regions.append(
                unruled_tables.TableRegion(
                    page.number + 1,
                    (
                        displayed.x0 / page.rect.width,
                        displayed.y0 / page.rect.height,
                        displayed.x1 / page.rect.width,
                        displayed.y1 / page.rect.height,
                    ),
                )
            )
        document.save(path)
    return regions


def test_actual_unruled_cells_keep_logical_order_under_rotations_and_do_not_index_symbol_fonts(tmp_path):
    path = tmp_path / "table.pdf"
    regions = _pdf(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = [["Category", "Value", "Status"], ["Alpha", "10", ""], ["Beta", "20", ""]]
    versions = set()
    for index, region in enumerate(regions):
        result = unruled_tables.extract_region(
            path, region, asset_id=42, first_unit_id=100, metadata_values={"tenant": "public"}
        )
        assert (result.table.rows, result.table.columns) == (3, 3)
        assert [[cell.text for cell in result.table.cells if cell.row == row] for row in range(3)] == expected
        assert len(result.glyphs) == 2 and all(glyph.column == 2 and glyph.row in (1, 2) for glyph in result.glyphs)
        assert all(unit.locator.page == index + 1 and unit.metadata["tenant"] == "public" for unit in result.units)
        assert [unit.unit_id for unit in result.units] == list(range(100, 107))
        assert all(cell.unit_id is None for cell in result.table.cells if not cell.text)
        assert all(any("dingbat" in font.lower() for font in glyph.fonts) for glyph in result.glyphs)
        versions.add(result.producer_version)
    assert len(versions) == 1 and hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_excess_words_cells_empty_region_and_missing_page_fail_without_truncation(tmp_path):
    path = tmp_path / "table.pdf"
    regions = _pdf(path)
    for budget in ({"max_words": 2}, {"max_cells": 2}):
        with pytest.raises(ValueError, match="budget"):
            unruled_tables.extract_region(path, regions[0], asset_id=42, first_unit_id=100, **budget)
    with pytest.raises(ValueError, match="empty"):
        unruled_tables.extract_region(
            path, unruled_tables.TableRegion(1, (0.8, 0.8, 0.9, 0.9)), asset_id=42, first_unit_id=100
        )
    with pytest.raises(ValueError, match="outside"):
        unruled_tables.extract_region(
            path, unruled_tables.TableRegion(10, (0.1, 0.1, 0.9, 0.9)), asset_id=42, first_unit_id=100
        )


def test_wrong_reference_source_is_rejected_before_native_extraction(tmp_path, monkeypatch):
    path = tmp_path / "wrong.pdf"
    path.write_bytes(b"wrong source")

    def unexpected(*args, **kwargs):
        raise RuntimeError("native extraction must not run")

    monkeypatch.setattr(unruled_tables, "extract_region", unexpected)
    with pytest.raises(ValueError, match="identity"):
        unruled_quality.evaluate(path, {"schema": "lentica-unruled-reference-v1", "source_sha256": "wrong"})


def test_table_import_and_help_need_no_native_dependencies():
    result = subprocess.run(
        [sys.executable, "-S", "-m", "backend.app.evaluation.unruled_benchmark", "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert "--reference" in result.stdout


def test_structure_localization_and_one_actual_final_read_per_query(tmp_path):
    path = tmp_path / "table.pdf"
    _pdf(path)
    reference = {
        "schema": "lentica-unruled-reference-v1",
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "page": 1,
        "region": [0.05, 0.05, 0.92, 0.5],
        "rows": 3,
        "columns": 3,
        "text": [["Category", "Value", "Status"], ["Alpha", "10", ""], ["Beta", "20", ""]],
        "table_bbox": [0.08, 0.115, 0.75, 0.36],
        "column_edges": [0.08, 0.30, 0.58, 0.75],
        "row_edges": [0.115, 0.19, 0.29, 0.36],
        "opaque_glyphs": 2,
        "queries": ["Alpha", "Beta"],
    }
    report = unruled_quality.evaluate(path, reference)
    assert report["accepted"] and report["structure"]["correct_nonempty_text_cells"] == 7
    assert report["retrieval"]["actual_final_pdf_reads"] == 2
    assert report["retrieval"]["unit_hit_at_1"] == 1
    assert all(row["hits"] == 0 and row["final_pdf_reads"] == 0 for row in report["retrieval"]["opaque_query_controls"])
    with pytest.raises(ValueError, match="alignment"):
        unruled_quality.evaluate(path, {**reference, "queries": ["Only one query"]})
