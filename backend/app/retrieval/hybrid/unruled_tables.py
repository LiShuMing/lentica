"""Explicit native PDF table regions with whitespace cells and opaque font glyphs."""

from __future__ import annotations

import bisect
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from . import artifacts, contracts, pdf_layout, units


@dataclass(frozen=True)
class TableRegion:
    """Caller-selected one-based page and normalized displayed table area."""

    page: int
    bbox: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        location = units.Locator(page=self.page, bbox=self.bbox)
        object.__setattr__(self, "bbox", location.bbox)


@dataclass(frozen=True)
class OpaqueGlyph:
    """An undecidable symbol-font word; raw_decode never implies glyph meaning."""

    row: int
    column: int
    bbox: tuple[float, float, float, float]
    raw_decode: str
    fonts: tuple[str, ...]


@dataclass(frozen=True)
class UnruledTable:
    """Logical cells plus searchable native text and separately retained glyphs."""

    table: pdf_layout.PdfTable
    units: tuple[units.RetrievalUnit, ...]
    glyphs: tuple[OpaqueGlyph, ...]
    producer_version: str


@dataclass(frozen=True)
class _Word:
    box: tuple[float, float, float, float]
    text: str
    symbol_fonts: tuple[str, ...]


def _intersects(first: Sequence[float], second: Sequence[float]) -> bool:
    return min(first[2], second[2]) > max(first[0], second[0]) and min(first[3], second[3]) > max(first[1], second[1])


def _words(page, clip, max_words: int) -> tuple[_Word, ...]:
    """Preserve mixed-font word uncertainty rather than indexing its fake letters."""
    import fitz

    raw = page.get_text("words", clip=clip)
    if not raw or len(raw) > max_words:
        raise ValueError("Table region is empty or exceeds word budget")
    blocks = page.get_text("dict", clip=clip, flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)["blocks"]
    spans = [span for block in blocks for line in block.get("lines", ()) for span in line["spans"]]
    if len(spans) > 20000:
        raise ValueError("Table region exceeds native span budget")
    symbols = [
        span
        for span in spans
        if any(marker in span["font"].casefold() for marker in ("wingdings", "dingbat", "webdings", "symbol"))
    ]
    result = []
    for word in raw:
        box = tuple(word[:4])
        if any(not math.isfinite(value) for value in box) or box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("Native table word has invalid geometry")
        fonts = tuple(sorted({span["font"] for span in symbols if _intersects(box, span["bbox"])}))
        result.append(_Word(box, word[4], fonts))
    return tuple(result)


def _edges(words: Sequence[_Word], axis: int, threshold: float) -> tuple[float, ...]:
    """Cut only projected empty space; never cut through a native word box."""
    intervals = sorted((word.box[axis], word.box[axis + 2]) for word in words)
    first, right = intervals[0]
    edges = [first]
    for start, end in intervals[1:]:
        if start - right >= threshold:
            edges.append((start + right) / 2)
        right = max(right, end)
    return tuple([*edges, right])


def _cells(page, words, xs, ys, asset_id, first_unit_id, metadata_values, version):
    """Keep empty/glyph-only slots without inventing searchable cell values."""
    grouped: dict[tuple[int, int], list[_Word]] = {}
    glyphs, records, cells = [], [], []
    for word in words:
        column = min(len(xs) - 2, bisect.bisect_right(xs, (word.box[0] + word.box[2]) / 2) - 1)
        row = min(len(ys) - 2, bisect.bisect_right(ys, (word.box[1] + word.box[3]) / 2) - 1)
        if word.symbol_fonts:
            glyphs.append(OpaqueGlyph(row, column, pdf_layout._bounds(page, word.box), word.text, word.symbol_fonts))
        else:
            grouped.setdefault((row, column), []).append(word)
    for row in range(len(ys) - 1):
        for column in range(len(xs) - 1):
            box = pdf_layout._bounds(page, (xs[column], ys[row], xs[column + 1], ys[row + 1]))
            text = " ".join(
                word.text
                for word in sorted(grouped.get((row, column), ()), key=lambda word: (word.box[1], word.box[0]))
            )
            identifier = None
            if text.strip():
                identifier = first_unit_id + len(records)
                records.append(
                    units.RetrievalUnit(
                        identifier,
                        asset_id,
                        "page",
                        units.Locator(page=page.number + 1, bbox=box),
                        (units.TextEvidence("native", text, "pymupdf-whitespace-table-cell", version),),
                        metadata_values,
                    )
                )
            cells.append(pdf_layout.TableCell(row, column, box, text, identifier))
    table = pdf_layout.PdfTable(
        page.number + 1,
        pdf_layout._bounds(page, (xs[0], ys[0], xs[-1], ys[-1])),
        len(ys) - 1,
        len(xs) - 1,
        tuple(cells),
    )
    return UnruledTable(table, tuple(records), tuple(glyphs), version)


def extract_region(
    path: Path,
    region: TableRegion,
    *,
    asset_id: int,
    first_unit_id: int,
    metadata_values: Mapping[str, str | int | float | bool | None] | None = None,
    max_words: int = 5000,
    max_cells: int = 5000,
) -> UnruledTable:
    """Infer a grid inside one explicit native-text region without modifying PDF.

    Args:
        path: Explicit local PDF, at most16MiB. No discovery/OCR/model runs occur.
        region: Displayed table area; logical row/column orientation survives rotation.
        asset_id: Parent resource identity, separate from the unit ID domain.
        first_unit_id: Start of contiguous IDs for nonempty searchable text cells.
        metadata_values: Parent scalar filters inherited by every returned unit.
        max_words: Positive native word bound, at most5000.
        max_cells: Positive logical slot bound, at most5000, including empty cells.

    Returns:
        All inferred slots, searchable native-text units and undecidable glyphs.
        Word-height-scaled gutters are a geometric baseline; merged cells and
        image-only/automatically discovered tables are not inferred. Symbol-font
        overlaps remain opaque, including when regular and symbol spans mix.

    Raises:
        ValueError: Invalid IDs, budgets, page/geometry or empty native region.
    """
    contracts.asset_id(asset_id)
    contracts.asset_id(first_unit_id)
    if (
        any(type(value) is not int or not 1 <= value <= 5000 for value in (max_words, max_cells))
        or Path(path).stat().st_size > 16 * 1024**2
    ):
        raise ValueError("Unruled table requires bounded word/cell/file budgets")
    import fitz

    version = artifacts.space_id(
        "native-unruled-v1",
        {},
        {
            "pymupdf": metadata.version("PyMuPDF"),
            "x_height_factor": 1.5,
            "y_height_factor": 0.35,
            "symbol_fonts": ["wingdings", "dingbat", "webdings", "symbol"],
            "max_words": max_words,
            "max_cells": max_cells,
            "max_spans": 20000,
        },
    )
    with fitz.open(path) as document:
        if region.page > len(document):
            raise ValueError("Table page is outside the selected PDF")
        page = document[region.page - 1]
        if page.rotation not in (0, 90, 180, 270) or max(page.rect.width, page.rect.height) > 20000:
            raise ValueError("Unsupported table page rotation/dimensions")
        x0, y0, x1, y1 = region.bbox
        clip = (
            fitz.Rect(x0 * page.rect.width, y0 * page.rect.height, x1 * page.rect.width, y1 * page.rect.height)
            * page.derotation_matrix
        )
        rotation = page.rotation
        try:
            page.set_rotation(0)
            words = _words(page, clip, max_words)
        finally:
            page.set_rotation(rotation)
        height = statistics.median(word.box[3] - word.box[1] for word in words)
        xs, ys = _edges(words, 0, height * 1.5), _edges(words, 1, height * 0.35)
        if (len(xs) - 1) * (len(ys) - 1) > max_cells:
            raise ValueError("Unruled table exceeds logical cell-slot budget")
        return _cells(page, words, xs, ys, asset_id, first_unit_id, metadata_values or {}, version)
