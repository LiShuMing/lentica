"""Mocked-score protocol checks are not evidence of real CLIP accuracy."""

from __future__ import annotations

import dataclasses
import json

import pytest

from backend.app.evaluation.hybrid_eval import image_quality, mirror_images


def fixture_source(directory):
    from PIL import Image

    items = []
    for partition in ("dev", "validation", "test"):
        for label in mirror_images.CLASSES:
            for _ in range(2):
                identifier = len(items)
                filename = f"{identifier:05d}.png"
                Image.new("RGB", (32, 32), (identifier * 3, 40, 180)).save(directory / filename)
                items.append(
                    mirror_images.ImageItem(
                        identifier,
                        f"fixture/{identifier}",
                        partition,
                        label,
                        filename,
                        f"pixel-{identifier}",
                        "fixture",
                    )
                )
    return mirror_images.ImageDataset(
        directory, tuple(items), "source-identity", {"kind": "synthetic protocol fixture"}
    )


class ScoreFixture:
    encoder_id = "mock-protocol-only"

    def __init__(self, source):
        self.by_file = {item.filename: item for item in source.items}
        self.partitions_seen = []

    def encode_images(self, paths):
        import numpy as np

        items = [self.by_file[path.name] for path in paths]
        self.partitions_seen.extend(item.partition for item in items)
        return np.eye(10)[[mirror_images.CLASSES.index(item.label) for item in items]]

    def encode_texts(self, prompts):
        import numpy as np

        assert tuple(prompts) == image_quality.PROMPTS
        return np.eye(10)


@pytest.mark.parametrize("protocol", ["similarity-v1", "margin-v2"])
def test_selection_never_scores_test_and_test_only_scores_frozen_threshold(tmp_path, protocol):
    source = fixture_source(tmp_path)
    model = ScoreFixture(source)
    selection = image_quality.select(source, model, protocol=protocol)
    assert set(model.partitions_seen) == {"dev", "validation"}
    assert len(selection["validation"]["open_gallery"]["thresholds"]) == 1
    model.partitions_seen.clear()
    frozen = json.loads(json.dumps(selection))
    result = image_quality.evaluate(source, model, frozen)
    assert set(model.partitions_seen) == {"test"}
    assert len(result["open_gallery"]["thresholds"]) == 1
    assert result["selected_no_answer"]["positive_queries"] == 12
    assert result["selected_no_answer"]["no_answer_queries"] == 8
    for key, ids in result["control_rankings"]["clip_image"].items():
        assert int(key) not in ids
    model.partitions_seen.clear()
    with pytest.raises(ValueError, match="identity"):
        image_quality.evaluate(dataclasses.replace(source, identity="changed"), model, frozen)
    assert not model.partitions_seen


def test_siglip_keeps_its_identity_and_control_names_without_scoring_test_in_select(tmp_path):
    source = fixture_source(tmp_path)
    model = ScoreFixture(source)
    model.kind, model.encoder_id = "siglip", "mock-siglip-protocol-only"
    selection = image_quality.select(source, model, protocol="margin-v2")
    assert "test" not in model.partitions_seen
    result = image_quality.evaluate(source, model, json.loads(json.dumps(selection)))
    assert "siglip_image" in result["quality"] and "clip_image" not in result["quality"]
    assert "siglip" in result["scope"]
