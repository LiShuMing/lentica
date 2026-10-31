"""Source leakage, location misses and no-answer metric denominators."""

from __future__ import annotations

import pytest

from backend.app.evaluation.hybrid_eval import localization, partitions, routed_controls
from backend.app.retrieval.hybrid import units


def test_source_components_keep_pages_crops_multi_positive_and_duplicate_queries_together():
    items = [
        partitions.SourceItem("page1", frozenset({"pdf"})),
        partitions.SourceItem("page2", frozenset({"pdf"})),
        partitions.SourceItem("multi-positive", frozenset({"pdf", "image"})),
        partitions.SourceItem("crop", frozenset({"image"}), "same-query"),
        partitions.SourceItem("duplicate", frozenset({"other"}), "same-query"),
        partitions.SourceItem("independent1", frozenset({"s1"})),
        partitions.SourceItem("independent2", frozenset({"s2"})),
    ]
    actual = partitions.source_partitions(items)
    assert actual == partitions.source_partitions(list(reversed(items)))
    assert all(actual.values()) and set().union(*map(set, actual.values())) == {item.item_id for item in items}
    group = next(set(values) for values in actual.values() if "page1" in values)
    assert group == {"page1", "page2", "multi-positive", "crop", "duplicate"}
    with pytest.raises(ValueError, match="Three independent"):
        partitions.source_partitions(items[:5])
    with pytest.raises(ValueError, match="unique"):
        partitions.source_partitions([items[0], items[0]])


def test_location_iou_uses_original_page_box_and_aligned_time():
    label = units.Locator(page=3, bbox=(0, 0, 1, 1))
    assert localization.locator_iou(units.Locator(page=3, bbox=(0, 0, 0.5, 1)), label) == 0.5
    assert localization.locator_iou(units.Locator(page=2, bbox=(0, 0, 1, 1)), label) == 0
    assert localization.locator_iou(units.Locator(page=3), label) == 0
    interval = units.Locator(start_seconds=10, end_seconds=20)
    assert localization.locator_iou(units.Locator(start_seconds=15, end_seconds=25), interval) == pytest.approx(1 / 3)
    assert localization.locator_iou(units.Locator(start_seconds=20, end_seconds=30), interval) == 0
    with pytest.raises(ValueError, match="Localization labels"):
        localization.locator_iou(label, units.Locator())


def test_resource_and_unit_recall_and_no_answer_are_separate():
    label = localization.Judgment(frozenset({10}), frozenset({2}), {10: (units.Locator(page=3),)})
    predictions = [
        units.RetrievalUnit(1, 10, "page", units.Locator(page=2)),
        units.RetrievalUnit(2, 10, "page", units.Locator(page=3)),
    ]
    actual = localization.score(label, predictions, k=1)
    assert actual["resource_recall"] == 1 and actual["unit_recall"] == 0
    assert actual["localization_iou"] == 0  # Page 3 below the unit window cannot rescue page 2.
    assert actual["no_answer_false_positive"] is None
    empty = localization.Judgment(frozenset(), frozenset())
    assert localization.score(empty, predictions)["no_answer_false_positive"] == 1
    assert localization.score(empty, [])["no_answer_false_positive"] == 0
    assert localization.score(empty, predictions)["resource_recall"] is None
    assert localization.score(label, [])["localization_iou"] == 0
    with pytest.raises(ValueError):
        localization.Judgment(frozenset({10}), frozenset())


def test_routed_controls_repeat_and_keep_no_answer_denominator_separate(tmp_path):
    first = routed_controls.run(tmp_path / "first", groups=3)
    second = routed_controls.run(tmp_path / "second", groups=3)
    assert first["ranking_sha256"] == second["ranking_sha256"]
    assert first["partition_sha256"] == second["partition_sha256"]
    assert first["dataset"]["queries"] == 21
    for report in first["comparisons"].values():
        assert report["metrics"]["positive_queries"] == 18
        no_answer = report["by_category"]["no_answer"]
        assert no_answer["queries"] == 3 and no_answer["positive_queries"] == 0
        assert no_answer["resource_recall"] is None
        assert no_answer["no_answer_false_positive"] is not None
        assert report["final_fetch"]["max_per_call"] <= 10
    parts = first["source_partitions"]
    for group in range(3):
        assert sum(any(item.startswith(f"g{group}-") for item in values) for values in parts.values()) == 1
