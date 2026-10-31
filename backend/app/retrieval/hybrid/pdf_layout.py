"""Opt-in mixed-page OCR and native ruled-table cells with explicit limitations."""

from __future__ import annotations

import dataclasses
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from . import extraction, units


@dataclass(frozen=True)
class TableCell:
    """Zero-based structural position; absent/merged slots have no unit identity."""

    row: int
    column: int
    bbox: tuple[float, float, float, float] | None
    text: str
    unit_id: int | None


@dataclass(frozen=True)
class PdfTable:
    """Detected native table; empty/merged slots are preserved without guessing."""

    page: int
    bbox: tuple[float, float, float, float]
    rows: int
    columns: int
    cells: tuple[TableCell, ...]


@dataclass(frozen=True)
class PdfLayout:
    """Bounded evidence and structural tables; unit order is geometric, not semantic."""

    units: tuple[units.RetrievalUnit, ...]
    tables: tuple[PdfTable, ...]
    ordering: str = "page then displayed y/x; table cells row-major within each table"


def _coverage(inner: Sequence[float], outer: Sequence[float]) -> float:
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    intersection = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0])) * max(
        0.0, min(inner[3], outer[3]) - max(inner[1], outer[1])
    )
    return intersection / area if area > 0 else 0.0


def _bounds(page, rect) -> tuple[float, float, float, float]:
    import fitz

    displayed = (fitz.Rect(rect) * page.rotation_matrix) & page.rect
    if displayed.is_empty:
        raise ValueError("Detected PDF cell has no displayed page area")
    return (
        displayed.x0 / page.rect.width,
        displayed.y0 / page.rect.height,
        displayed.x1 / page.rect.width,
        displayed.y1 / page.rect.height,
    )


def _table_units(page, asset_id, next_id, metadata_values, remaining):
    """Copy table data while the PyMuPDF page/document is alive."""
    records, tables = [], []
    slots = 0
    rotation = page.rotation
    raw_tables = []
    try:
        # Native table detection follows displayed axes. Temporarily unrotate
        # only this in-memory page so logical row/column identities survive a
        # viewer rotation; restore before OCR and coordinate normalization.
        page.set_rotation(0)
        for table in page.find_tables(strategy="lines_strict").tables:
            slots += table.row_count * table.col_count
            if slots > remaining:
                raise ValueError("PDF table cell slots exceed combined unit budget")
            raw_tables.append(
                (table.bbox, table.row_count, table.col_count, table.extract(), [row.cells for row in table.rows])
            )
    finally:
        page.set_rotation(rotation)
    for bounds, rows, columns, values, cell_rows in raw_tables:
        cells = []
        for row, cell_row in enumerate(cell_rows):
            for column, rect in enumerate(cell_row):
                text = values[row][column] or ""
                bbox = _bounds(page, rect) if rect is not None else None
                identifier = None
                if bbox is not None and text.strip():
                    identifier = next_id + len(records)
                    records.append(
                        units.RetrievalUnit(
                            identifier,
                            asset_id,
                            "page",
                            units.Locator(page=page.number + 1, bbox=bbox),
                            (
                                units.TextEvidence(
                                    "native", text, "pymupdf-ruled-table-cell", metadata.version("PyMuPDF")
                                ),
                            ),
                            metadata_values,
                        )
                    )
                cells.append(TableCell(row, column, bbox, text, identifier))
        tables.append(PdfTable(page.number + 1, _bounds(page, bounds), rows, columns, tuple(cells)))
    return records, tables, slots


def _ocr_page(page, directory, ocr, dpi, asset_id, next_id, metadata_values):
    """OCR logical orientation, then restore displayed-page region coordinates."""
    import fitz

    rotation = page.rotation
    try:
        page.set_rotation(0)
        width, height = page.rect.width, page.rect.height
        scale = dpi / 72
        if (int(width * scale) + 1) * (int(height * scale) + 1) > ocr.max_image_pixels:
            raise ValueError("Mixed PDF page exceeds OCR pixel budget")
        rendered = Path(directory) / "page.png"
        page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False).save(rendered)
        values = ocr.extract_units(
            rendered, asset_id=asset_id, first_unit_id=next_id, page=page.number + 1, metadata_values=metadata_values
        )
    finally:
        page.set_rotation(rotation)
    result = []
    for unit in values:
        x0, y0, x1, y1 = unit.locator.bbox
        bbox = _bounds(page, (x0 * width, y0 * height, x1 * width, y1 * height))
        result.append(dataclasses.replace(unit, locator=units.Locator(page=page.number + 1, bbox=bbox)))
    return tuple(result)


def _duplicate_ocr(candidate: units.RetrievalUnit, native: Sequence[units.RetrievalUnit]) -> bool:
    text = "".join(character.casefold() for character in candidate.text() if character.isalnum())
    if len(text) < 3:
        return False
    for unit in native:
        original = "".join(character.casefold() for character in unit.text() if character.isalnum())
        if text in original and _coverage(candidate.locator.bbox, unit.locator.bbox) >= 0.6:
            return True
    return False


def extract(
    path: Path,
    ocr: extraction.RapidOcrExtractor,
    *,
    asset_id: int,
    first_unit_id: int,
    max_pages: int = 100,
    max_units: int = 5000,
    dpi: int = 150,
    metadata_values: Mapping[str, str | int | float | bool | None] | None = None,
) -> PdfLayout:
    """Extract native cells/text plus new OCR evidence on every selected PDF page.

    Native blocks fully inside a table are replaced by its populated cells. OCR
    duplicates need both normalized-text containment and substantial spatial
    coverage; equal text elsewhere remains separate evidence. Table metadata is
    separate from inherited resource filters. No reading-order model, row spans,
    unruled-table inference, or OCR table reconstruction is claimed.

    Raises:
        ValueError: DPI, page, unit/cell-slot, file or raster budget is exceeded.
    """
    import fitz

    if type(dpi) is not int or not 72 <= dpi <= 300 or Path(path).stat().st_size > 64 * 1024**2:
        raise ValueError("PDF layout requires bounded DPI and at most64MiB input")
    native = extraction.pdf_blocks(
        path,
        asset_id=asset_id,
        first_unit_id=first_unit_id,
        max_pages=max_pages,
        max_blocks=max_units,
        metadata_values=metadata_values,
    )
    records, tables, keys = [], [], {}
    cell_slots = 0
    with fitz.open(path) as document, tempfile.TemporaryDirectory(prefix="lentica-pdf-layout-") as directory:
        for page in document:
            page_records, page_tables, slots = _table_units(
                page, asset_id, first_unit_id + len(records), metadata_values or {}, max_units - cell_slots
            )
            cell_slots += slots
            for table in page_tables:
                for cell in table.cells:
                    if cell.unit_id is not None:
                        keys[cell.unit_id] = (table.page, table.bbox[1], table.bbox[0], cell.row, cell.column)
            records.extend(page_records)
            tables.extend(page_tables)
            for unit in native:
                if unit.locator.page != page.number + 1:
                    continue
                if any(
                    _coverage(unit.locator.bbox, table.bbox) >= 0.95
                    and all(
                        word.casefold() in " ".join(cell.text for cell in table.cells).casefold()
                        for word in unit.text().split()
                    )
                    for table in page_tables
                ):
                    continue
                records.append(dataclasses.replace(unit, unit_id=first_unit_id + len(records)))
            page_native = [unit for unit in records if unit.locator.page == page.number + 1]
            ocr_units = _ocr_page(
                page, directory, ocr, dpi, asset_id, first_unit_id + len(records), metadata_values or {}
            )
            for unit in ocr_units:
                if not _duplicate_ocr(unit, page_native):
                    records.append(dataclasses.replace(unit, unit_id=first_unit_id + len(records)))
            if len(records) > max_units:
                raise ValueError("PDF layout exceeds combined native/table/OCR unit budget")
    records.sort(
        key=lambda unit: keys.get(unit.unit_id, (unit.locator.page, unit.locator.bbox[1], unit.locator.bbox[0], -1, -1))
    )
    reassigned = {unit.unit_id: first_unit_id + index for index, unit in enumerate(records)}
    return PdfLayout(
        tuple(dataclasses.replace(unit, unit_id=reassigned[unit.unit_id]) for unit in records),
        tuple(
            dataclasses.replace(
                table,
                cells=tuple(
                    dataclasses.replace(cell, unit_id=reassigned[cell.unit_id] if cell.unit_id is not None else None)
                    for cell in table.cells
                ),
            )
            for table in tables
        ),
    )
