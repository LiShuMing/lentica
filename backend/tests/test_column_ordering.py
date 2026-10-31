"""Geometric controls and honest missing-anchor accounting, not document accuracy."""

from __future__ import annotations

import subprocess
import sys

import pytest

from backend.app.evaluation.hybrid_eval import layout_quality
from backend.app.retrieval.hybrid import column_ordering, units


def _region(identifier, box, text, *, page=1, asset=1):
    return units.RetrievalUnit(
        identifier,
        asset,
        "page",
        units.Locator(page=page, bbox=box),
        (units.TextEvidence("native", text, "fixture", "v1"),),
        {"tenant": "public"},
    )


def test_wide_header_and_footer_do_not_force_column_interleaving_or_change_evidence():
    regions = [
        _region(1, (0.1, 0.05, 0.9, 0.1), "header"),
        _region(2, (0.1, 0.2, 0.45, 0.3), "left first"),
        _region(3, (0.55, 0.2, 0.9, 0.3), "right first"),
        _region(4, (0.1, 0.4, 0.45, 0.5), "left second"),
        _region(5, (0.55, 0.4, 0.9, 0.5), "right second"),
        _region(6, (0.1, 0.9, 0.9, 0.95), "footer"),
    ]
    expected = [1, 2, 4, 3, 5, 6]
    result = column_ordering.order_columns(tuple(reversed(regions)))
    assert [unit.unit_id for unit in result] == expected
    assert all(unit is regions[unit.unit_id - 1] for unit in result)
    assert all(unit.metadata["tenant"] == "public" for unit in result)


def test_pages_and_assets_are_isolated_and_overlapping_ties_use_stable_ids():
    box = (0.1, 0.1, 0.9, 0.9)
    regions = [
        _region(5, box, "last", asset=2),
        _region(4, box, "later", page=2),
        _region(3, box, "tie"),
        _region(2, box, "first tie"),
    ]
    assert [unit.unit_id for unit in column_ordering.order_columns(regions)] == [2, 3, 4, 5]
    assert column_ordering.order_columns([]) == ()


def test_duplicate_missing_locations_and_excess_budgets_fail():
    region = _region(1, (0.1, 0.1, 0.4, 0.4), "fixture")
    with pytest.raises(ValueError, match="unique"):
        column_ordering.order_columns([region, region])
    with pytest.raises(ValueError, match="page and"):
        column_ordering.order_columns([units.RetrievalUnit(1, 1, "text")])
    with pytest.raises(ValueError, match="bounded"):
        column_ordering.order_columns([region], max_units=0)
    with pytest.raises(ValueError, match="bounded"):
        column_ordering.order_columns([region], min_column_gap=float("nan"))


def test_missing_anchors_and_incorrect_pairs_cannot_be_accepted_by_reducing_denominator():
    records = [_region(1, (0.1, 0.1, 0.9, 0.2), "first"), _region(2, (0.1, 0.3, 0.9, 0.4), "third")]
    report = layout_quality.evaluate(records, ["first", "missing", "third"])
    assert report["xy_cut"]["found"] == 2
    assert report["xy_cut"]["directed_pairs"] == 3 and report["xy_cut"]["correct_pairs"] == 1
    assert report["xy_cut"]["pair_accuracy"] == pytest.approx(1 / 3)
    assert not report["accepted"]
    assert not layout_quality.evaluate(records, ["third", "first"])["accepted"]


def test_layout_help_is_safe_without_optional_dependencies():
    subprocess.run(
        [sys.executable, "-S", "-m", "backend.app.evaluation.layout_benchmark", "--help"],
        check=True,
        capture_output=True,
        timeout=15,
    )
