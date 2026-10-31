"""Ordered time support, shared domains and budgets, without model-quality claims."""

import pytest

from backend.app.retrieval.hybrid import contracts, temporal, units


def item(identifier, start, end, asset=1, tenant="public"):
    return units.RetrievalUnit(
        identifier, asset, "video", units.Locator(start_seconds=start, end_seconds=end), metadata={"tenant": tenant}
    )


def test_sequence_preserves_explicit_step_order_resource_and_covering_support():
    first, second, third = item(1, 0, 1), item(2, 2, 3), item(3, 4, 5)
    result = temporal.sequence(
        [("z-first", [first]), ("a-second", [second]), ("middle", [third])],
        contracts.SearchFilter(),
        min_gap_seconds=1,
        max_gap_seconds=1,
    )
    assert len(result) == 1 and result[0].evidence == (first, second, third)
    assert result[0].locator == units.Locator(start_seconds=0, end_seconds=5)
    assert temporal.sequence([("a", [second]), ("b", [first])], contracts.SearchFilter()) == ()
    assert temporal.sequence([("a", [first]), ("b", [item(4, 2, 3, asset=2)])], contracts.SearchFilter()) == ()
    assert temporal.sequence([("a", [first]), ("b", [item(5, 0.5, 2)])], contracts.SearchFilter()) == ()
    assert temporal.intersect({"a": [first], "b": [second]}, contracts.SearchFilter()) == ()


def test_sequence_touching_gap_bounds_and_stable_ids():
    first, touching, later = item(1, 0, 1), item(3, 1, 2), item(2, 1, 2)
    groups = [("first", [first]), ("second", [touching, later])]
    results = temporal.sequence(groups, contracts.SearchFilter())
    assert [r.evidence[-1].unit_id for r in results] == [2, 3]
    assert temporal.sequence(groups, contracts.SearchFilter(), min_gap_seconds=0.01) == ()
    assert (
        temporal.sequence([("a", [first]), ("b", [item(5, 10, 11)])], contracts.SearchFilter(), max_gap_seconds=2) == ()
    )


def test_sequence_rejects_filter_leaks_identity_and_intermediate_excess():
    first = item(1, 0, 1)
    groups = [("a", [first]), ("b", [item(2, 2, 3), item(3, 4, 5)])]
    for kwargs in ({"max_matches": 1}, {"max_combinations": 1}):
        with pytest.raises(ValueError, match="budget"):
            temporal.sequence(groups, contracts.SearchFilter(), **kwargs)
    with pytest.raises(ValueError, match="filters"):
        temporal.sequence(groups, contracts.SearchFilter(equals={"tenant": "other"}))
    with pytest.raises(ValueError, match="identity"):
        temporal.sequence([("a", [first]), ("b", [item(1, 2, 3)])], contracts.SearchFilter())
    with pytest.raises(ValueError, match="unique"):
        temporal.sequence([("a", [first]), ("a", [])], contracts.SearchFilter())
    assert temporal.sequence([("a", []), ("b", [])], contracts.SearchFilter()) == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_gap_seconds": True},
        {"min_gap_seconds": -1},
        {"max_gap_seconds": float("inf")},
        {"min_gap_seconds": 2, "max_gap_seconds": 1},
        {"max_combinations": 0},
        {"max_matches": True},
    ],
)
def test_sequence_invalid_constraints_fail(kwargs):
    with pytest.raises(ValueError):
        temporal.sequence([("a", []), ("b", [])], contracts.SearchFilter(), **kwargs)
