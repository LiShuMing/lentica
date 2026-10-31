"""Calibration data/call boundaries; mocked quality is not model accuracy evidence."""

from __future__ import annotations

import dataclasses
import json

import pytest

from backend.app.evaluation.hybrid_eval import calibrated_rescue, cascade_quality, image_quality, rescue_quality
from backend.app.retrieval.hybrid import category_gating
from backend.tests import test_category_cascade


def test_joint_calibration_runs_all_dev_candidates_but_only_one_validation_pair(tmp_path):
    import numpy as np

    source, fast, slow, _ = test_category_cascade._fixture(tmp_path)
    original = fast.encode_images

    def uncertain(paths):
        vectors = original(paths)
        for index, path in enumerate(paths):
            if fast.by_file[path.name].label == "airplane":
                vectors[index] = 0
                vectors[index, 0], vectors[index, 9] = 1, 1.01
                vectors[index] /= np.linalg.norm(vectors[index])
        return vectors

    fast.encode_images = uncertain
    lock = calibrated_rescue.select(source, fast, slow)
    assert lock["accepted"] and lock["selected_fast_margin"] == 0.01
    assert len(lock["development"]) == 4 and len(lock["validation"]["thresholds"]) == 1
    assert all(len(report["thresholds"]) == len(image_quality.MARGINS) for report in lock["development"].values())
    assert fast.partitions_seen.count("dev") == 80 and fast.partitions_seen.count("validation") == 20
    # dev:12-gallery+20-reference+10+12+12+12 query calls. Validation:12+12+20.
    assert slow.partitions_seen.count("dev") == 78 and slow.partitions_seen.count("validation") == 44
    assert set(slow.partitions_seen) == {"dev", "validation"}
    assert lock["development"]["0.005"]["cost"]["slow_images"] == 10
    assert lock["development"]["0.01"]["cost"]["slow_images"] == 12
    assert all(report["shared_preparation"] for report in lock["development"].values())
    fast.partitions_seen.clear()
    slow.partitions_seen.clear()
    result = calibrated_rescue.evaluate(source, fast, slow, json.loads(json.dumps(lock)))
    assert result["eligible_for_promotion"] and len(result["thresholds"]) == 1
    assert fast.partitions_seen == ["test"] * 20 and slow.partitions_seen == ["test"] * 44
    fast.partitions_seen.clear()
    slow.partitions_seen.clear()
    with pytest.raises(ValueError, match="identity"):
        calibrated_rescue.evaluate(dataclasses.replace(source, identity="wrong"), fast, slow, lock)
    assert not fast.partitions_seen and not slow.partitions_seen


def test_prepared_gallery_is_readonly_partition_bound_and_not_a_query_cache(tmp_path):
    source, fast, slow, _ = test_category_cascade._fixture(tmp_path)
    prepared = rescue_quality.prepare_gallery(source, slow, "dev")
    assert not prepared.vectors.flags.writeable and len(slow.partitions_seen) == 32
    gate = category_gating.CascadedCategoryGate(
        fast, slow, cascade_quality.PROMPTS, cascade_quality.GALLERY, fast_negative_margin=0.01
    )
    rescue_quality.measure(source, gate, slow, "dev", margins=(-1.0,), fast_negative_margin=0.01, prepared=prepared)
    assert len(slow.partitions_seen) == 44 and len(fast.partitions_seen) == 20
    fast.partitions_seen.clear()
    slow.partitions_seen.clear()
    with pytest.raises(ValueError, match="partition/model/ID"):
        rescue_quality.measure(
            source, gate, slow, "test", margins=(-1.0,), fast_negative_margin=0.01, prepared=prepared
        )
    assert not fast.partitions_seen and not slow.partitions_seen


def test_failed_validation_cannot_be_reversed_by_successful_test(tmp_path, monkeypatch):
    source, fast, slow, _ = test_category_cascade._fixture(tmp_path)
    original = rescue_quality.measure

    def expensive_validation(*args, **kwargs):
        report = original(*args, **kwargs)
        if report["partition"] == "validation":
            report["cost"]["seconds_per_query_image"] = 0.3
        return report

    monkeypatch.setattr(rescue_quality, "measure", expensive_validation)
    lock = calibrated_rescue.select(source, fast, slow)
    assert not lock["accepted"] and len(lock["validation"]["thresholds"]) == 1
    result = calibrated_rescue.evaluate(source, fast, slow, lock)
    assert result["heldout_gate_passed"] and not result["eligible_for_promotion"]


def test_gallery_preparation_rejects_model_identity_drift(tmp_path):
    source, _, slow, _ = test_category_cascade._fixture(tmp_path)
    original = slow.encode_images

    def changing_model(paths):
        result = original(paths)
        slow.encoder_id = "replacement-model"
        return result

    slow.encode_images = changing_model
    with pytest.raises(ValueError, match="identity changed during preparation"):
        rescue_quality.prepare_gallery(source, slow, "dev")
