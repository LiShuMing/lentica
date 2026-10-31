"""Actual mixed PDF OCR/native-table controls; not broad document accuracy."""

from __future__ import annotations

import hashlib

import pytest

from backend.app.retrieval.hybrid import contracts, documents, extraction, media, pdf_layout, sparse, units


def _mixed_pdf(path, image_path):
    import fitz
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (700, 150), "white")
    ImageDraw.Draw(image).text((15, 35), "SCANNED ERROR 429", font=ImageFont.load_default(size=40), fill="black")
    image.save(image_path)
    with fitz.open() as document:
        for rotation in (0, 90):
            page = document.new_page(width=800, height=600)
            page.insert_text((50, 60), "NATIVE HEADER", fontsize=20)
            for x in (50, 300, 550):
                page.draw_line((x, 120), (x, 240))
            for y in (120, 180, 240):
                page.draw_line((50, y), (550, y))
            for row in range(2):
                for column in range(2):
                    page.insert_text((60 + 250 * column, 155 + 60 * row), f"R{row}C{column}", fontsize=20)
            page.insert_image(fitz.Rect(50, 350, 750, 500), filename=str(image_path))
            page.set_rotation(rotation)
        document.save(path)


def test_real_mixed_page_recalls_scanned_text_keeps_tables_rotation_and_filters(tmp_path):
    pytest.importorskip("fitz")
    pytest.importorskip("rapidocr_onnxruntime")
    path = tmp_path / "mixed.pdf"
    _mixed_pdf(path, tmp_path / "scan.png")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    ocr = extraction.RapidOcrExtractor()
    old = extraction.pdf_with_ocr(path, ocr, asset_id=30, first_unit_id=100)
    assert not any("SCANNED" in unit.text() for unit in old)
    result = pdf_layout.extract(path, ocr, asset_id=30, first_unit_id=100, metadata_values={"tenant": "a"})
    assert len(result.tables) == 2
    assert [unit.unit_id for unit in result.units] == list(range(100, 100 + len(result.units)))
    by_id = {unit.unit_id: unit for unit in result.units}
    for table in result.tables:
        assert (table.rows, table.columns) == (2, 2)
        assert [(cell.row, cell.column, cell.text) for cell in table.cells] == [
            (row, column, f"R{row}C{column}") for row in range(2) for column in range(2)
        ]
        assert [cell.unit_id for cell in table.cells] == sorted(cell.unit_id for cell in table.cells)
        assert all(by_id[cell.unit_id].locator.bbox == cell.bbox for cell in table.cells)
    for page in (1, 2):
        matches = [unit for unit in result.units if unit.locator.page == page and "SCANNED" in unit.text()]
        assert matches and all(unit.evidence[0].kind == "ocr" for unit in matches)
        header = [unit for unit in result.units if unit.locator.page == page and "NATIVE HEADER" in unit.text()]
        assert len(header) == 1 and header[0].evidence[0].kind == "native"
    retriever = media.EvidenceTextRetriever(units.UnitCatalog(result.units), frozenset({"native", "ocr"}))
    # Actual OCR joins words here (SCANNEDERROR429). Preserve that evidence and
    # use the existing character route, rather than changing source/expectations
    # or pretending word BM25 recognizes a token absent from the OCR output.
    character = sparse.SparseTfidfRetriever(
        [documents.CorpusDocument(unit.unit_id, unit.text(), metadata=unit.metadata) for unit in result.units],
        analyzer="char_wb",
        ngram_range=(3, 5),
    )
    hits = character.retrieve(contracts.SearchQuery("SCANNED", contracts.SearchFilter(equals={"tenant": "a"})), 2)
    assert len(hits) == 2 and {by_id[identifier].locator.page for identifier in hits} == {1, 2}
    assert (
        retriever.retrieve(contracts.SearchQuery("SCANNED", contracts.SearchFilter(equals={"tenant": "other"})), 10)
        == ()
    )
    assert (
        character.retrieve(contracts.SearchQuery("SCANNED", contracts.SearchFilter(equals={"tenant": "other"})), 2)
        == ()
    )
    assert all(unit.metadata["tenant"] == "a" and unit.locator.bbox for unit in result.units)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    with pytest.raises(ValueError, match="budget"):
        pdf_layout.extract(path, ocr, asset_id=30, first_unit_id=100, max_units=3)
    ocr.max_image_pixels = 100
    with pytest.raises(ValueError, match="pixel budget"):
        pdf_layout.extract(path, ocr, asset_id=30, first_unit_id=100)
