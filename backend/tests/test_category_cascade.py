"""Cascade call paths, frozen selection and cost checks; fixtures are not accuracy."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys

import pytest

from backend.app.evaluation.hybrid_eval import cascade_quality, reporting, rescue_quality
from backend.app.retrieval.hybrid import category_gating
from backend.tests import test_image_quality_protocol


def test_slow_model_only_sees_fast_accepted_positions_and_returns_complete_decisions(tmp_path):
    import numpy as np

    paths = [tmp_path / f"{index}.png" for index in range(3)]
    for path in paths:
        path.write_bytes(b"explicit fixture")

    class Encoder:
        def __init__(self, name, rows):
            self.encoder_id, self.rows, self.seen = name, rows, []

        def encode_texts(self, texts):
            return np.eye(3)

        def encode_images(self, chosen):
            self.seen.extend(path.name for path in chosen)
            return np.eye(3)[[self.rows[int(path.stem)] for path in chosen]]

    fast, slow = Encoder("fast", [0, 2, 0]), Encoder("slow", [0, 1, 2])
    prompts = {"present": "present", "absent": "absent", "other": "other"}
    gate = category_gating.CascadedCategoryGate(fast, slow, prompts, frozenset({"present"}))
    prompts["present"] = "caller mutation"
    result = gate.score_images(paths)
    assert slow.seen == ["0.png", "2.png"]
    assert fast.seen == ["0.png", "1.png", "2.png"]
    assert result.fast_accepts == (True, False, True)
    assert result.decisions(-1) == (True, False, False)
    assert result.decisions(1) == (True, False, False)
    assert result.margins == (1.0, None, -1.0)
    assert gate.prompts[0] == "present"
    assert gate.score_images([]).fast_vectors.shape == (0, 3)
    assert slow.seen == ["0.png", "2.png"]
    slow.encoder_id = "changed"
    with pytest.raises(ValueError, match="identity"):
        gate.score_images(paths)


def _fixture(directory):
    source = test_image_quality_protocol.fixture_source(directory)
    fast, slow = (test_image_quality_protocol.ScoreFixture(source) for _ in range(2))
    fast.encoder_id, slow.encoder_id = "fixture-fast", "fixture-slow"
    gate = category_gating.CascadedCategoryGate(fast, slow, cascade_quality.PROMPTS, cascade_quality.GALLERY)
    return source, fast, slow, gate


def test_uncertain_fast_negative_is_rescued_and_slow_vectors_preserve_positions(tmp_path):
    import numpy as np

    path = tmp_path / "query.png"
    path.write_bytes(b"explicit fixture")

    class Encoder:
        encoder_id = "fixture"

        def __init__(self, row):
            self.row, self.seen = row, []

        def encode_texts(self, texts):
            return np.eye(2)

        def encode_images(self, paths):
            self.seen.extend(paths)
            return np.tile(self.row, (len(paths), 1))

    fast = Encoder(np.asarray([0.7, 0.71]) / np.linalg.norm([0.7, 0.71]))
    slow = Encoder(np.asarray([1.0, 0.0]))
    prompts = {"gallery": "present", "other": "absent"}
    strict = category_gating.CascadedCategoryGate(fast, slow, prompts, frozenset({"gallery"}))
    assert strict.score_images([path]).decisions(-1) == (False,)
    assert not slow.seen
    rescue = category_gating.CascadedCategoryGate(
        fast, slow, prompts, frozenset({"gallery"}), fast_negative_margin=0.02
    )
    result = rescue.score_images([path])
    assert result.decisions(-1) == (True,) and slow.seen == [path]
    assert result.slow_vectors.tolist() == [[1.0, 0.0]]
    fast.row = np.asarray([0.0, 1.0])
    rejected = rescue.score_images([path])
    assert rejected.decisions(-1) == (False,) and slow.seen == [path]
    assert rejected.slow_vectors.tolist() == [[0.0, 0.0]]
    assert strict.gate_id != rescue.gate_id


@pytest.mark.parametrize("fast_margin", [rescue_quality.FAST_NEGATIVE_MARGIN, rescue_quality.TIGHT_NEGATIVE_MARGIN])
def test_rescue_gallery_is_separate_from_actual_queries_and_selection_is_frozen(tmp_path, fast_margin):
    source, fast, slow, _ = _fixture(tmp_path)
    gate = category_gating.CascadedCategoryGate(
        fast,
        slow,
        cascade_quality.PROMPTS,
        cascade_quality.GALLERY,
        fast_negative_margin=fast_margin,
    )
    lock = rescue_quality.select(source, gate, slow, fast_negative_margin=fast_margin)
    assert lock["accepted"] and set(fast.partitions_seen) == {"dev", "validation"}
    assert set(slow.partitions_seen) == {"dev", "validation"}
    assert len(lock["validation"]["thresholds"]) == 1
    fast.partitions_seen.clear()
    slow.partitions_seen.clear()
    result = rescue_quality.evaluate(source, gate, slow, json.loads(json.dumps(lock)), fast_negative_margin=fast_margin)
    assert result["eligible_for_promotion"] and result["gallery_images"] == 12
    assert result["cost"]["slow_images"] == 12
    assert len(fast.partitions_seen) == 20 and len(slow.partitions_seen) == 44
    assert set(slow.partitions_seen) == {"test"}
    assert all(int(key) not in ids for ids_by_query in result["rankings"].values() for key, ids in ids_by_query.items())
    slow.partitions_seen.clear()
    with pytest.raises(ValueError, match="identity"):
        rescue_quality.evaluate(
            dataclasses.replace(source, identity="wrong"), gate, slow, lock, fast_negative_margin=fast_margin
        )
    assert not slow.partitions_seen
    with pytest.raises(ValueError, match="identity"):
        rescue_quality.evaluate(source, gate, slow, lock, fast_negative_margin=0.01)
    assert not slow.partitions_seen
    lock["accepted"] = False
    lock.pop("sha256")
    lock["sha256"] = reporting.digest(lock)
    assert not rescue_quality.evaluate(source, gate, slow, lock, fast_negative_margin=fast_margin)[
        "eligible_for_promotion"
    ]


def test_selection_has_no_test_inference_and_frozen_identity_is_checked_before_pixels(tmp_path):
    source, fast, slow, gate = _fixture(tmp_path)
    lock = cascade_quality.select(source, gate, slow)
    assert lock["accepted"]
    assert set(fast.partitions_seen) == {"dev", "validation"}
    assert set(slow.partitions_seen) == {"dev", "validation"}
    assert len(lock["validation"]["thresholds"]) == 1
    fast.partitions_seen.clear()
    slow.partitions_seen.clear()
    frozen = json.loads(json.dumps(lock))
    result = cascade_quality.evaluate(source, gate, slow, frozen)
    assert set(fast.partitions_seen) == {"test"} and set(slow.partitions_seen) == {"test"}
    assert result["cost"]["fast_images"] == 20 and result["cost"]["slow_images"] == 12
    assert result["full_slow_reference"]["images"] == 20
    assert len(result["thresholds"]) == 1 and result["eligible_for_promotion"]
    for name in ("fast_full", "fast_gallery", "fast_gate"):
        assert all(int(key) not in values for key, values in result["rankings"][name].items())
    fast.partitions_seen.clear()
    slow.partitions_seen.clear()
    with pytest.raises(ValueError, match="identity"):
        cascade_quality.evaluate(dataclasses.replace(source, identity="changed"), gate, slow, frozen)
    assert not fast.partitions_seen and not slow.partitions_seen


def test_cost_failure_does_not_become_promotion_or_disappear_from_selection(tmp_path, monkeypatch):
    source, fast, slow, gate = _fixture(tmp_path)
    actual_score = gate.score_images

    def expensive(paths):
        return dataclasses.replace(actual_score(paths), total_seconds=len(paths) * 0.3)

    monkeypatch.setattr(gate, "score_images", expensive)
    lock = cascade_quality.select(source, gate, slow)
    assert not lock["accepted"]
    result = cascade_quality.evaluate(source, gate, slow, json.loads(json.dumps(lock)))
    assert result["quality_gate_passed"] and not result["cost_gate_passed"]
    assert not result["heldout_gate_passed"] and not result["eligible_for_promotion"]


def test_test_success_cannot_reverse_a_rejected_selection(tmp_path):
    source, _, slow, gate = _fixture(tmp_path)
    lock = cascade_quality.select(source, gate, slow)
    lock.update(accepted=False, selected_margin=-1.0)
    lock.pop("sha256")
    lock["sha256"] = reporting.digest(lock)
    result = cascade_quality.evaluate(source, gate, slow, lock)
    assert result["heldout_gate_passed"] and not result["eligible_for_promotion"]


def test_model_failure_is_not_silently_replaced_and_cli_help_is_stdlib_only(tmp_path, monkeypatch):
    source, fast, slow, gate = _fixture(tmp_path)

    def failure(paths):
        raise RuntimeError("slow runtime failure")

    monkeypatch.setattr(slow, "encode_images", failure)
    with pytest.raises(RuntimeError, match="slow runtime"):
        gate.score_images([source.directory / source.items[0].filename])
    subprocess.run(
        [sys.executable, "-S", "-m", "backend.app.evaluation.cascade_benchmark", "--help"],
        check=True,
        timeout=10,
        capture_output=True,
    )
