"""Explicit native table rows with field binding and marked geometric spans."""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from . import artifacts, contracts, pdf_layout, units, unruled_tables

FIELD_PREFIX = "table.field."


@dataclass(frozen=True)
class TableColumn:
    """Caller-declared field key and evidence label; no automatic header inference."""

    key: str
    label: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.key, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.key)
            or not isinstance(self.label, str)
            or not self.label.strip()
            or len(self.label) > 256
        ):
            raise ValueError("Table fields require bounded explicit keys and labels")


@dataclass(frozen=True)
class FieldCell:
    """One row-bound value, retaining its full original cell and span provenance."""

    row: int
    column: int
    value: str
    bbox: tuple[float, float, float, float]
    source_bbox: tuple[float, float, float, float]
    source_text: str
    source_row: int
    span_rows: tuple[int, ...]
    method: str


@dataclass(frozen=True)
class TableRow:
    """A searchable data row; fields retain precise answer locations separately."""

    row: int
    unit_id: int
    cells: tuple[FieldCell, ...]


@dataclass(frozen=True)
class TableContext:
    """Explicit typed row units and unchanged coarse native-table diagnostics."""

    columns: tuple[TableColumn, ...]
    rows: tuple[TableRow, ...]
    units: tuple[units.RetrievalUnit, ...]
    native_values: tuple[tuple[str | None, ...], ...]
    producer_version: str


def _clusters(words: Sequence[unruled_tables._Word], tolerance: float) -> list[list[unruled_tables._Word]]:
    """Cluster word centers; use the current mean rather than chaining wide gaps."""
    groups: list[list[unruled_tables._Word]] = []
    for word in sorted(words, key=lambda item: ((item.box[1] + item.box[3]) / 2, item.box[0])):
        center = (word.box[1] + word.box[3]) / 2
        if not groups or abs(center - _center(groups[-1])) > tolerance:
            groups.append([])
        groups[-1].append(word)
    return groups


def _center(words: Sequence[unruled_tables._Word]) -> float:
    return statistics.mean((word.box[1] + word.box[3]) / 2 for word in words)


def _text(words: Sequence[unruled_tables._Word]) -> str:
    return " ".join(word.text for word in sorted(words, key=lambda item: item.box[0]))


def _contains(rect, word: unruled_tables._Word) -> bool:
    x, y = (word.box[0] + word.box[2]) / 2, (word.box[1] + word.box[3]) / 2
    return rect[0] <= x < rect[2] and rect[1] <= y < rect[3]


def _geometry(cell_rows, header_rows, column_count):
    """Accept rectangular coarse data groups; unsupported merges fail explicitly."""
    previous = None
    for rects in cell_rows[header_rows:]:
        if len(rects) != column_count or any(rect is None for rect in rects):
            raise ValueError("Coarse data rows require a detected rectangle for every field")
        if any(
            any(not math.isfinite(value) for value in rect)
            or rect[0] >= rect[2]
            or rect[1] >= rect[3]
            or abs(rect[1] - rects[0][1]) > 0.5
            or abs(rect[3] - rects[0][3]) > 0.5
            for rect in rects
        ) or any(abs(left[2] - right[0]) > 0.5 for left, right in zip(rects, rects[1:])):
            raise ValueError("Native data cells do not form aligned coarse rows/columns")
        if previous is not None and (
            abs(previous[0][3] - rects[0][1]) > 0.5
            or any(abs(old[0] - new[0]) > 0.5 or abs(old[2] - new[2]) > 0.5 for old, new in zip(previous, rects))
        ):
            raise ValueError("Native data row boundaries are discontinuous")
        previous = rects


def _baselines(by_column, tolerance):
    """Require repeated multi-column evidence; sparse centered conditions are spans."""
    words = [word for group in by_column for word in group]
    if not words:
        raise ValueError("Native data group has no words")
    centers = []
    for cluster in _clusters(words, tolerance):
        support = sum(any(word in cluster for word in group) for group in by_column)
        if support >= math.ceil(len(by_column) / 2):
            centers.append(_center(cluster))
    if not centers:
        raise ValueError("Native data group has insufficient repeated column baselines")
    return centers


def _field_lines(words, centers, tolerance):
    """Split complete lines or mark a centered all-row span; never forward-fill."""
    groups = _clusters(words, tolerance)
    if not groups:
        return [""] * len(centers), False
    if len(groups) == 1 and len(centers) > 1:
        y = _center(groups[0])
        if (
            min(abs(y - center) for center in centers) > tolerance
            and abs(y - (centers[0] + centers[-1]) / 2) <= tolerance / 2
        ):
            return [_text(groups[0])] * len(centers), True
        raise ValueError("Ambiguous single-line cell cannot imply a merged or missing value")
    if len(groups) != len(centers) or any(abs(_center(group) - y) > tolerance for group, y in zip(groups, centers)):
        raise ValueError("Cell text lines do not align with complete data subrows")
    return [_text(group) for group in groups], False


def _data_group(page, rects, all_words, first_row, source_row, tolerance):
    """Expand one coarse row while keeping each value's original native rectangle."""
    by_column = [[word for word in all_words if _contains(rect, word)] for rect in rects]
    centers = _baselines(by_column, tolerance)
    edges = [rects[0][1], *((left + right) / 2 for left, right in zip(centers, centers[1:])), rects[0][3]]
    fields = [_field_lines(words, centers, tolerance) for words in by_column]
    values = [[] for _ in centers]
    for column, (rect, words, (texts, span)) in enumerate(zip(rects, by_column, fields)):
        source_text = "\n".join(_text(group) for group in _clusters(words, tolerance))
        span_rows = tuple(range(first_row, first_row + len(centers))) if span else ()
        for offset, text in enumerate(texts):
            values[offset].append(
                FieldCell(
                    first_row + offset,
                    column,
                    text,
                    pdf_layout._bounds(page, (rect[0], edges[offset], rect[2], edges[offset + 1])),
                    pdf_layout._bounds(page, rect),
                    source_text,
                    source_row,
                    span_rows,
                    "geometric-span" if span else "native-line" if text else "empty",
                )
            )
    return values


def _records(page, values, columns, asset_id, first_unit_id, metadata_values, version):
    rows, records = [], []
    for index, cells in enumerate(values):
        identifier = first_unit_id + index
        boxes = [cell.bbox for cell in cells]
        bbox = (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )
        values_metadata = {FIELD_PREFIX + column.key: cell.value for column, cell in zip(columns, cells)}
        records.append(
            units.RetrievalUnit(
                identifier,
                asset_id,
                "page",
                units.Locator(page=page.number + 1, bbox=bbox),
                (
                    units.TextEvidence(
                        "native",
                        "\n".join(f"{column.label}: {cell.value}" for column, cell in zip(columns, cells)),
                        "pymupdf-explicit-table-row",
                        version,
                    ),
                ),
                {**metadata_values, **values_metadata},
            )
        )
        rows.append(TableRow(index, identifier, tuple(cells)))
    return tuple(rows), tuple(records)


def extract_rows(
    path: Path,
    region: unruled_tables.TableRegion,
    columns: Sequence[TableColumn],
    *,
    header_rows: int,
    asset_id: int,
    first_unit_id: int,
    metadata_values: Mapping[str, str | int | float | bool | None] | None = None,
    max_words: int = 5000,
    max_cells: int = 5000,
    max_rows: int = 1000,
) -> TableContext:
    """Bind explicit fields to native data rows, retaining geometric merge evidence.

    Args:
        path: Explicit local PDF, at most16MiB; original bytes remain unchanged.
        region: Selected table on one displayed page, with no automatic discovery.
        columns: Ordered caller-declared keys/labels, separate from reference answers.
        header_rows: Number of coarse native header rows to exclude from data.
        asset_id: Parent resource ID, distinct from row-unit IDs.
        first_unit_id: First ID of contiguous searchable data-row units.
        metadata_values: Inherited resource filters; table.field. is reserved.
        max_words: Native word bound, at most5000.
        max_cells: Both coarse and expanded cell-slot bound, at most5000.
        max_rows: Expanded data-row bound, at most1000.

    Returns:
        Row units, field/source positions, span methods and coarse native values.
        Field predicates are explicit exact filters, not natural-language parsing.

    Raises:
        ValueError: Invalid inputs, budgets, ambiguous sparse text, unsupported
            coarse geometry/symbol fonts, or nonunique detected table.
    """
    contracts.asset_id(asset_id)
    contracts.asset_id(first_unit_id)
    columns, inherited = tuple(columns), dict(metadata_values or {})
    if (
        not 2 <= len(columns) <= 50
        or any(not isinstance(column, TableColumn) for column in columns)
        or len({column.key for column in columns}) != len(columns)
        or type(header_rows) is not int
        or not 0 <= header_rows <= 100
        or any(type(value) is not int or not 1 <= value <= 5000 for value in (max_words, max_cells))
        or type(max_rows) is not int
        or not 1 <= max_rows <= 1000
        or any(not isinstance(key, str) or key.startswith(FIELD_PREFIX) for key in inherited)
        or Path(path).stat().st_size > 16 * 1024**2
    ):
        raise ValueError("Explicit table fields/metadata/header/file or budgets are invalid")
    import fitz

    version = artifacts.space_id(
        "native-table-context-v1",
        {},
        {
            "pymupdf": metadata.version("PyMuPDF"),
            "columns": [(column.key, column.label) for column in columns],
            "header_rows": header_rows,
            "line_height_factor": 0.35,
            "span_center_tolerance_factor": 0.5,
            "column_support_fraction": 0.5,
            "max_words": max_words,
            "max_cells": max_cells,
            "max_rows": max_rows,
        },
    )
    with fitz.open(path) as document:
        if region.page > len(document):
            raise ValueError("Table page is outside selected PDF")
        page = document[region.page - 1]
        if page.rotation not in (0, 90, 180, 270) or max(page.rect.width, page.rect.height) > 20000:
            raise ValueError("Unsupported table page rotation/dimensions")
        x0, y0, x1, y1 = region.bbox
        clip = fitz.Rect(x0 * page.rect.width, y0 * page.rect.height, x1 * page.rect.width, y1 * page.rect.height)
        clip = clip * page.derotation_matrix
        rotation = page.rotation
        try:
            page.set_rotation(0)
            words = unruled_tables._words(page, clip, max_words)
            if any(word.symbol_fonts for word in words):
                raise ValueError("Opaque symbol glyphs are unsupported in table row contexts")
            detected = page.find_tables(strategy="lines_strict", clip=clip).tables
            if len(detected) != 1:
                raise ValueError("Explicit region must contain exactly one native ruled table")
            table = detected[0]
            if table.col_count != len(columns) or table.row_count * table.col_count > max_cells:
                raise ValueError("Native table fields or cell budget differ")
            if header_rows >= table.row_count:
                raise ValueError("Table requires data rows after explicit headers")
            cell_rows = [tuple(row.cells) for row in table.rows]
            native_values = tuple(tuple(row) for row in table.extract())
        finally:
            page.set_rotation(rotation)
        _geometry(cell_rows, header_rows, len(columns))
        tolerance = statistics.median(word.box[3] - word.box[1] for word in words) * 0.35
        values = []
        for source_row, rects in enumerate(cell_rows[header_rows:], start=header_rows):
            values.extend(_data_group(page, rects, words, len(values), source_row, tolerance))
            if len(values) > max_rows or len(values) * len(columns) > max_cells:
                raise ValueError("Expanded table exceeds data-row/cell budget")
        rows, records = _records(page, values, columns, asset_id, first_unit_id, inherited, version)
    return TableContext(columns, rows, records, native_values, version)
