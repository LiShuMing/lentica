"""Development runtime/language protocol checks; fixtures are not model quality."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys

import pytest

from backend.app.evaluation.hybrid_eval import image_quality, language_quality, reporting, vision_runtime
from backend.tests import test_image_quality_protocol


def test_runtime_profile_only_encodes_development_and_lock_checks_before_loading(tmp_path, monkeypatch):
    source = test_image_quality_protocol.fixture_source(tmp_path)
    models = []

    def build(directory, *, threads, batch_size):
        model = test_image_quality_protocol.ScoreFixture(source)
        model.encoder_id = f"mock:{threads}:{batch_size}"
        models.append(model)
        return model

    monkeypatch.setattr(vision_runtime.siglip_onnx, "SiglipOnnxEncoder", build)
    times = iter((0, 3, 4, 7, 10, 11, 12, 13, 20, 22, 23, 25))
    monkeypatch.setattr(vision_runtime.time, "perf_counter", lambda: next(times))
    lock = vision_runtime.profile(source, tmp_path)
    assert lock["selected_index"] == 1
    assert all(len(model.partitions_seen) == 40 and set(model.partitions_seen) == {"dev"} for model in models)
    frozen = json.loads(json.dumps(lock))
    model = vision_runtime.frozen_model(source, tmp_path, frozen)
    assert model.encoder_id == "mock:1:1"
    count = len(models)
    with pytest.raises(ValueError, match="identity"):
        vision_runtime.frozen_model(dataclasses.replace(source, identity="changed"), tmp_path, frozen)
    assert len(models) == count


def test_all_fixed_language_prompts_run_and_only_test_images_are_encoded(tmp_path):
    import numpy as np

    source = test_image_quality_protocol.fixture_source(tmp_path)
    model = test_image_quality_protocol.ScoreFixture(source)
    seen_prompts = []

    def encode(texts):
        seen_prompts.append(tuple(texts))
        return np.eye(10)

    model.encode_texts = encode
    report = language_quality.evaluate(source, model)
    assert set(model.partitions_seen) == {"test"}
    assert seen_prompts == list(language_quality.PROMPT_GROUPS.values())
    assert set(report["groups"]) == set(language_quality.PROMPT_GROUPS)
    assert all(len(row["per_prompt"]) == 10 for row in report["groups"].values())
    assert all(row["metrics"]["positive"]["recall_at_10"] == 1 for row in report["groups"].values())


def test_failed_image_selection_is_not_reversed_by_good_test(tmp_path):
    source = test_image_quality_protocol.fixture_source(tmp_path)
    model = test_image_quality_protocol.ScoreFixture(source)
    lock = image_quality.select(source, model, protocol="margin-v2")
    lock.update(accepted=False, selected_threshold=-1.0)
    lock.pop("sha256")
    lock["sha256"] = reporting.digest(lock)
    result = image_quality.evaluate(source, model, lock)
    assert result["heldout_gate_passed"]
    assert not result["selection_accepted"] and not result["eligible_for_promotion"]


def test_iteration_help_is_stdlib_only():
    subprocess.run(
        [sys.executable, "-S", "-m", "backend.app.evaluation.vision_iteration", "--help"],
        check=True,
        capture_output=True,
        timeout=10,
    )
