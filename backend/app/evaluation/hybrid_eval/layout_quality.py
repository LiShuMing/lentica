"""Anchor recall and directed-pair ordering; missing references stay in denominator."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence

from backend.app.evaluation.hybrid_eval import reporting
from backend.app.retrieval.hybrid import column_ordering, units


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).casefold()


def _score(records: Sequence[units.RetrievalUnit], anchors: Sequence[str]) -> reporting.JsonObject:
    """Locate anchors within ordered text, allowing several anchors in one block."""
    text = _normalize("\n".join(unit.text() for unit in records))
    positions = [text.find(_normalize(anchor)) for anchor in anchors]
    pairs = len(anchors) * (len(anchors) - 1) // 2
    correct = sum(
        0 <= positions[first] < positions[second]
        for first in range(len(anchors))
        for second in range(first + 1, len(anchors))
    )
    found = sum(position >= 0 for position in positions)
    return {
        "anchors": len(anchors),
        "found": found,
        "anchor_recall": found / len(anchors),
        "directed_pairs": pairs,
        "correct_pairs": correct,
        "pair_accuracy": correct / pairs,
        "positions": positions,
        "missing": [anchor for anchor, position in zip(anchors, positions) if position < 0],
    }


def evaluate(
    records: Sequence[units.RetrievalUnit], anchors: Sequence[str], *, min_column_gap: float = 0.03
) -> reporting.JsonObject:
    """Compare fixed y/x baseline with fixed XY-cut; never select gaps from labels."""
    if (
        not 2 <= len(anchors) <= 200
        or len(set(anchors)) != len(anchors)
        or any(not isinstance(anchor, str) or not anchor.strip() or len(anchor) > 2048 for anchor in anchors)
    ):
        raise ValueError("Require2..200 unique bounded manual anchors")
    started = time.perf_counter()
    ordered = column_ordering.order_columns(records, min_column_gap=min_column_gap)
    elapsed = time.perf_counter() - started
    baseline = sorted(
        records,
        key=lambda unit: (unit.asset_id, unit.locator.page, unit.locator.bbox[1], unit.locator.bbox[0], unit.unit_id),
    )
    candidate = _score(ordered, anchors)
    rankings = {"baseline": [unit.unit_id for unit in baseline], "xy_cut": [unit.unit_id for unit in ordered]}
    return {
        "schema": "lentica-column-order-v1",
        "configuration": {
            "min_column_gap": min_column_gap,
            "min_row_gap": 0.008,
            "max_depth": 64,
            "edge_band_ratio": 0.8,
            "normalization": "whitespace-and-casefold",
        },
        "baseline": _score(baseline, anchors),
        "xy_cut": candidate,
        "units": len(records),
        "ordering_seconds": elapsed,
        "accepted": candidate["found"] == len(anchors) and candidate["correct_pairs"] == candidate["directed_pairs"],
        "rankings": rankings,
        "ranking_sha256": reporting.digest(rankings),
        "scope": "Fixed manually transcribed anchors; missing anchors fail; geometric ordering only, not general layout/semantic accuracy.",
    }
