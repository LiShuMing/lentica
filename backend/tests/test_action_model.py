"""Ordered temporal input/control provenance; fixtures do not prove action quality."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from backend.app.retrieval.hybrid import action_model, action_retrieval, contracts, units, video_frames


def _frames():
    return tuple(
        video_frames.FrameSample(
            units.RetrievalUnit(index, 1, "video", units.Locator(start_seconds=index, end_seconds=index + 1)),
            Path(f"frame-{index}.png"),
            index,
            "fixture-hash",
        )
        for index in range(16)
    )


def test_original_order_and_control_provenance_cannot_be_interchanged():
    frames = _frames()
    action_model._validate(frames, "original")
    action_model._validate(frames[::-1], "reverse_control")
    action_model._validate((frames[0],) * 16, "static_control")
    for values, kind in (
        (frames[::-1], "original"),
        (frames, "reverse_control"),
        (frames, "static_control"),
        (frames[:15], "original"),
        ((frames[0],) * 16, "original"),
        (frames, "unknown"),
    ):
        with pytest.raises(ValueError):
            action_model._validate(values, kind)
    different = dataclasses.replace(frames[0], unit=dataclasses.replace(frames[0].unit, asset_id=2))
    with pytest.raises(ValueError, match="same video"):
        action_model._validate((different,) + frames[1:], "original")
    wrong_time = dataclasses.replace(
        frames[1], unit=dataclasses.replace(frames[1].unit, locator=units.Locator(start_seconds=0, end_seconds=1))
    )
    with pytest.raises(ValueError, match="nonoverlapping"):
        action_model._validate((frames[0], wrong_time) + frames[2:], "original")


def test_class_ties_are_stable_and_help_does_not_load_a_model():
    scored = action_model.ActionScores(("a", "b", "c"), (0.0, 1.0, 1.0), "fixture", "original")
    assert scored.ranking() == (1, 2, 0)
    subprocess.run(
        [sys.executable, "-S", "-m", "backend.app.evaluation.action_benchmark", "--help"],
        check=True,
        capture_output=True,
        timeout=10,
    )


def test_action_features_keep_controls_out_and_apply_filters_before_window():
    catalog = units.UnitCatalog(
        [
            units.RetrievalUnit(
                identifier,
                identifier,
                "video",
                units.Locator(start_seconds=0, end_seconds=1),
                metadata={"tenant": tenant},
            )
            for identifier, tenant in ((1, "other"), (2, "a"), (3, "a"), (4, "a"))
        ]
    )
    rows = {
        identifier: action_model.ActionScores(("eat", "run"), (value, 0.0), "model", "original")
        for identifier, value in ((1, 99.0), (2, 2.0), (3, 3.0), (4, 3.0))
    }
    retriever = action_retrieval.ActionLabelRetriever(catalog, rows)
    rows.clear()
    query = contracts.SearchQuery(" EAT ", contracts.SearchFilter(equals={"tenant": "a"}))
    assert retriever.retrieve(query, 1) == (3,)
    assert retriever.retrieve(query, 4) == (3, 4, 2)
    assert retriever.retrieve(contracts.SearchQuery("eat", contracts.SearchFilter(allowed_ids=frozenset({2}))), 1) == (
        2,
    )
    with pytest.raises(ValueError, match="exact native"):
        retriever.retrieve(contracts.SearchQuery("arbitrary natural language"), 1)
    with pytest.raises(ValueError, match="original"):
        action_retrieval.ActionLabelRetriever(
            catalog, {2: action_model.ActionScores(("eat", "run"), (1.0, 0.0), "model", "static_control")}
        )
    mutable = [1.0, 0.0]
    captured = action_model.ActionScores(["eat", "run"], mutable, "model", "original")
    mutable[0] = 99.0
    assert captured.logits == (1.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        action_model.ActionScores(("eat",), (float("nan"),), "model", "original")
