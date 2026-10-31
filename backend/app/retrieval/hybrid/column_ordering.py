"""Opt-in geometric XY-cut order for bounded page regions, without a layout model."""

from __future__ import annotations

import math
from collections.abc import Sequence

from . import units


def _geometric(records: Sequence[units.RetrievalUnit]) -> tuple[units.RetrievalUnit, ...]:
    return tuple(sorted(records, key=lambda unit: (unit.locator.bbox[1], unit.locator.bbox[0], unit.unit_id)))


def _gap(records: Sequence[units.RetrievalUnit], axis: int) -> tuple[float, float]:
    """Largest whitespace between unions of projected boxes; ties choose first."""
    intervals = sorted((unit.locator.bbox[axis], unit.locator.bbox[axis + 2]) for unit in records)
    right, width, cut = intervals[0][1], 0.0, 0.0
    for start, end in intervals[1:]:
        if start - right > width:
            width, cut = start - right, (start + right) / 2
        right = max(right, end)
    return width, cut


def _edge_band(records: Sequence[units.RetrievalUnit], minimum: float) -> float | None:
    """Isolate full-width edge bands before a middle gap can split both columns."""
    span = max(unit.locator.bbox[2] for unit in records) - min(unit.locator.bbox[0] for unit in records)
    narrow = [unit for unit in records if unit.locator.bbox[2] - unit.locator.bbox[0] < 0.8 * span]
    if not narrow:
        return None
    top, bottom = min(unit.locator.bbox[1] for unit in narrow), max(unit.locator.bbox[3] for unit in narrow)
    header = [unit for unit in records if unit.locator.bbox[3] <= top]
    footer = [unit for unit in records if unit.locator.bbox[1] >= bottom]
    for band, before in ((header, True), (footer, False)):
        if not band:
            continue
        ids = {unit.unit_id for unit in band}
        others = [unit for unit in records if unit.unit_id not in ids]
        left = max(unit.locator.bbox[3] for unit in (band if before else others))
        right = min(unit.locator.bbox[1] for unit in (others if before else band))
        if right - left >= minimum:
            return (left + right) / 2
    return None


def _cut(records, column_gap, row_gap, depth):
    if len(records) < 2 or depth >= 64:
        return _geometric(records)
    for axis, threshold in ((0, column_gap), (1, row_gap)):
        width, position = _gap(records, axis)
        if axis == 1:
            edge = _edge_band(records, row_gap)
            if edge is not None:
                width, position = row_gap, edge
        if width < threshold:
            continue
        first = [unit for unit in records if unit.locator.bbox[axis + 2] <= position]
        second = [unit for unit in records if unit.locator.bbox[axis] > position]
        if first and second and len(first) + len(second) == len(records):
            return _cut(first, column_gap, row_gap, depth + 1) + _cut(second, column_gap, row_gap, depth + 1)
    return _geometric(records)


def order_columns(
    records: Sequence[units.RetrievalUnit],
    *,
    min_column_gap: float = 0.03,
    min_row_gap: float = 0.008,
    max_units: int = 5000,
) -> tuple[units.RetrievalUnit, ...]:
    """Order explicit page regions while retaining their identities and evidence.

    Args:
        records: Unique units with normalized displayed boxes and one-based pages.
            Pass paragraph regions; this function does not infer table structure.
        min_column_gap: Minimum full-width-normalized whitespace for an x split.
        min_row_gap: Minimum full-height-normalized whitespace for a y split.
        max_units: Total input bound, at most10000. No filesystem reads occur.

    Returns:
        Original objects ordered by asset/page then recursive left/right or
        top/bottom cuts. Overlaps and depth64 fall back to stable y/x/ID order.
        This is a geometric heuristic, not certified semantic reading order.

    Raises:
        ValueError: Budgets, gaps, identities or page/box locations are invalid.
    """
    if (
        type(max_units) is not int
        or not 1 <= max_units <= 10000
        or len(records) > max_units
        or any(
            type(gap) not in (int, float) or not math.isfinite(gap) or not 0 < gap < 0.5
            for gap in (min_column_gap, min_row_gap)
        )
    ):
        raise ValueError("Column ordering requires bounded units and finite positive normalized gaps")
    if len({unit.unit_id for unit in records}) != len(records):
        raise ValueError("Column ordering requires unique unit IDs")
    pages: dict[tuple[int, int], list[units.RetrievalUnit]] = {}
    for unit in records:
        if unit.locator.page is None or unit.locator.bbox is None:
            raise ValueError("Column ordering requires explicit page and normalized bbox")
        pages.setdefault((unit.asset_id, unit.locator.page), []).append(unit)
    return tuple(unit for key in sorted(pages) for unit in _cut(pages[key], min_column_gap, min_row_gap, 0))
