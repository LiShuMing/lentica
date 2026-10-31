"""Frozen relation inputs and honest scoring; fixtures do not prove model quality."""

from __future__ import annotations

import copy
import dataclasses
import json
import subprocess
import sys

import pytest

from backend.app.evaluation.hybrid_eval import relation_quality, relation_source, reporting
from backend.app.retrieval.hybrid import blip_native


@pytest.fixture
def relation_inputs(tmp_path):
    from PIL import Image

    images = tmp_path / "raw"
    images.mkdir()
    for index in range(6):
        Image.new("RGB", (8, 8), (index * 30, 30, 200)).save(images / f"{index:012d}.jpg")
    paths = {}
    for kind_index, kind in enumerate(relation_source.KINDS):
        ids = range(3) if kind_index == 0 else range(2, 6)
        rows = {
            str(index): {
                "filename": f"{index:012d}.jpg",
                "caption": "cat follows dog",
                "negative_caption": "dog follows cat",
            }
            for index in ids
        }
        path = tmp_path / (kind + ".json")
        path.write_text(json.dumps(rows))
        paths[kind] = path
    return paths, images, tmp_path / "prepared"


@pytest.fixture
def dataset(relation_inputs):
    annotations, images, target = relation_inputs
    return relation_source.prepare(annotations, images, target, per_kind=3)


def test_selection_is_source_grouped_counterbalanced_and_frozen_before_availability(relation_inputs):
    annotations, images, target = relation_inputs
    selected = relation_source.select_cases(annotations, per_kind=3)
    assert [case.source_image for case in selected] == [f"{index:012d}.jpg" for index in range(6)]
    assert [case.partition for case in selected] == list(relation_source.PARTITIONS) * 2
    assert [case.positive_index for case in selected] == [0, 1, 0, 0, 1, 0]
    (images / selected[0].source_image).unlink()
    with pytest.raises(FileNotFoundError):
        relation_source.prepare(annotations, images, target, per_kind=3)
    frozen = json.loads((target / "source-selection.json").read_text())
    assert frozen["cases"] == [dataclasses.asdict(case) for case in selected]
    assert not (target / "manifest.json").exists()  # Missing inputs never cause sample replacement.


def test_duplicate_pixels_fail_without_replacement(relation_inputs):
    annotations, images, target = relation_inputs
    (images / "000000000001.jpg").write_bytes((images / "000000000000.jpg").read_bytes())
    with pytest.raises(ValueError, match="Duplicate selected pixel"):
        relation_source.prepare(annotations, images, target, per_kind=3)


def test_preparation_rejects_symlink_sources_and_image_byte_budget(relation_inputs):
    annotations, images, target = relation_inputs
    first = images / "000000000000.jpg"
    first.unlink()
    first.symlink_to(images / "000000000001.jpg")
    with pytest.raises(ValueError, match="path escapes"):
        relation_source.prepare(annotations, images, target, per_kind=3)
    first.unlink()
    first.write_bytes(b"x" * (relation_source.MAX_FILE_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds4MiB"):
        relation_source.prepare(annotations, images, target.with_name("oversize"), per_kind=3)


def test_snapshot_bytes_selection_and_manifest_are_checked(dataset):
    manifest_path = dataset.directory / "manifest.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    manifest["cases"][0]["positive_index"] = 1
    manifest.pop("identity")
    manifest["identity"] = reporting.digest(manifest)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="frozen annotation-only selection"):
        relation_source.load(dataset.directory)
    manifest_path.write_bytes(original)
    assert relation_source.load(dataset.directory).identity == dataset.identity
    image = dataset.directory / dataset.cases[0].filename
    image.write_bytes(image.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="identity differs"):
        relation_source.load(dataset.directory)


def test_fresh_source_selection_excludes_prior_sources_and_pixels(dataset, relation_inputs):
    from PIL import Image

    annotations, images, target = relation_inputs
    for kind_index, kind in enumerate(relation_source.KINDS):
        rows = json.loads(annotations[kind].read_text())
        for index in range(6 + kind_index * 3, 9 + kind_index * 3):
            filename = f"{index:012d}.jpg"
            Image.new("RGB", (8, 8), (index * 17, 100, 25)).save(images / filename)
            rows[str(index)] = {
                "filename": filename,
                "caption": "cat follows dog",
                "negative_caption": "dog follows cat",
            }
        annotations[kind].write_text(json.dumps(rows))
    fresh = relation_source.prepare(annotations, images, target.with_name("fresh"), per_kind=3, exclude=dataset)
    assert not {c.source_image for c in fresh.cases} & {c.source_image for c in dataset.cases}
    assert not {c.pixel_sha256 for c in fresh.cases} & {c.pixel_sha256 for c in dataset.cases}
    assert [c.source_image for c in fresh.cases] == [f"{i:012d}.jpg" for i in range(6, 12)]
    assert relation_source.load(fresh.directory).identity == fresh.identity
    manifest = json.loads((fresh.directory / "manifest.json").read_text())
    assert manifest["excluded_dataset"]["identity"] == dataset.identity
    # A third round inherits first-round exclusions instead of resampling old IDs.
    with pytest.raises(ValueError, match="Insufficient distinct source-image groups"):
        relation_source.prepare(annotations, images, target.with_name("third"), per_kind=3, exclude=fresh)
    # A different filename with previously observed pixels fails; it is not skipped.
    (images / "000000000006.jpg").write_bytes((images / dataset.cases[0].source_image).read_bytes())
    with pytest.raises(ValueError, match="Duplicate selected pixel"):
        relation_source.prepare(annotations, images, target.with_name("same-pixels"), per_kind=3, exclude=dataset)
    frozen = json.loads((target.with_name("same-pixels") / "source-selection.json").read_text())
    assert frozen["cases"][0]["source_image"] == "000000000006.jpg"
    changed = dataclasses.replace(dataset, identity="f" * 64)
    with pytest.raises(ValueError, match="explicit exclusion"):
        relation_source.prepare(annotations, images, target.with_name("invalid-prior"), per_kind=3, exclude=changed)


class PairFixture:
    scorer_id = "fixture-joint-not-accuracy"
    last_unique_images = 1

    def __init__(self, *, tie=False, fail=False, drift=False):
        self.calls = []
        self.tie, self.fail, self.drift = tie, fail, drift

    def score_pairs(self, texts, paths):
        self.calls.append((tuple(texts), tuple(paths)))
        assert paths[0] == paths[1] and paths[0].name.startswith("image-")
        if self.drift:
            self.scorer_id = "changed"
        if self.fail:
            raise RuntimeError("fixture provider failure")
        return [0.0 if self.tie else float(text.startswith("cat")) for text in texts]


def test_joint_scores_each_case_fresh_in_input_order_and_retains_ties_failures(dataset):
    for scorer, correct, ties, failures in (
        (PairFixture(), 6, 0, 0),
        (PairFixture(tie=True), 0, 6, 0),
        (PairFixture(fail=True), 0, 0, 6),
    ):
        report = relation_quality.measure_joint(dataset, scorer)
        assert len(scorer.calls) == 6
        assert report["overall"]["cases"] == 6
        assert (report["overall"]["correct"], report["overall"]["ties"], report["overall"]["failures"]) == (
            correct,
            ties,
            failures,
        )
        assert [texts for texts, _ in scorer.calls] == [case.captions for case in dataset.cases]
        assert report["overall"]["same_word_bag_cases"] == 6
    with pytest.raises(ValueError, match="identity changed during"):
        relation_quality.measure_joint(dataset, PairFixture(drift=True))


def test_dual_runs_image_and_both_texts_per_case(dataset):
    import numpy as np

    class Dual:
        encoder_id = "fixture-dual-not-accuracy"

        def __init__(self):
            self.images, self.texts = [], []

        def encode_images(self, paths):
            self.images.append(tuple(paths))
            return np.array([[1.0, 0.0]])

        def encode_texts(self, texts):
            self.texts.append(tuple(texts))
            return np.array([[1.0, 0.0] if text.startswith("cat") else [0.0, 1.0] for text in texts])

    model = Dual()
    report = relation_quality.measure_dual(dataset, model)
    assert report["overall"]["accuracy"] == 1
    assert len(model.images) == len(model.texts) == 6
    assert model.texts == [case.captions for case in dataset.cases]


def test_compare_recomputes_decisions_and_test_cannot_reverse_dev_failure(dataset):
    # Different fixed fixtures exercise policy, not real model gains or latency.
    clip = relation_quality.measure_joint(dataset, PairFixture(tie=True))
    siglip = copy.deepcopy(clip)
    blip = relation_quality.measure_joint(dataset, PairFixture())
    reports = {"clip": clip, "siglip": siglip, "blip": blip}
    assert relation_quality.compare(dataset, reports)["eligible_for_promotion"]
    for field, value in (("captions", ["forged", "label"]), ("correct", False), ("seconds", -1)):
        changed = copy.deepcopy(reports)
        changed["blip"]["per_case"][0][field] = value
        with pytest.raises(ValueError):
            relation_quality.compare(dataset, changed)
    changed = copy.deepcopy(reports)
    changed["blip"]["overall"]["correct"] += 1
    with pytest.raises(ValueError, match="summaries differ"):
        relation_quality.compare(dataset, changed)
    changed = copy.deepcopy(reports)
    changed["blip"]["dataset_identity"] = "other"
    with pytest.raises(ValueError, match="identity differs"):
        relation_quality.compare(dataset, changed)

    class DevFailure(PairFixture):
        def score_pairs(self, texts, paths):
            return (
                [0.0, 0.0]
                if paths[0].name in {"image-000000.jpg", "image-000003.jpg"}
                else super().score_pairs(texts, paths)
            )

    reports["blip"] = relation_quality.measure_joint(dataset, DevFailure())
    comparison = relation_quality.compare(dataset, reports)
    assert comparison["partitions"]["test"]["gate_passed"]
    assert not comparison["partitions"]["dev"]["gate_passed"]
    assert not comparison["eligible_for_promotion"]


def test_blip_empty_and_token_budget_reject_before_image_access():
    scorer = object.__new__(blip_native.BlipPairScorer)
    scorer.max_pairs, scorer.max_tokens = 4, 128
    assert scorer.score_pairs([], []) == ()
    for invalid in ("", b"", None):
        with pytest.raises(ValueError, match="aligned bounded"):
            scorer.score_pairs(invalid, invalid)

    class Tokens:
        shape = (1, 129)

    def tokenize(texts, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": Tokens()}

    scorer.tokenizer = tokenize
    with pytest.raises(ValueError, match="token budget"):
        scorer.score_pairs(["x"], ["never-open-a-file"])


def test_relation_imports_and_cli_help_are_stdlib_only():
    code = """
import sys
from backend.app.evaluation import relation_benchmark
from backend.app.evaluation.hybrid_eval import relation_source, relation_quality
from backend.app.retrieval.hybrid import blip_native, vision_reranking
assert not any(name.startswith(('numpy', 'PIL', 'torch', 'transformers', 'onnxruntime')) for name in sys.modules)
sys.argv = ['relation_benchmark', '--help']
relation_benchmark.main()
"""
    subprocess.run([sys.executable, "-S", "-c", code], check=True, capture_output=True, timeout=10)


def test_blip_resolution_is_explicit_bounded_and_unique_paths_share_only_one_decode(tmp_path):
    from PIL import Image

    for invalid in (True, 225, 224.0):
        with pytest.raises(ValueError, match="bounded native profile"):
            blip_native.BlipPairScorer(tmp_path / "must-not-load", image_size=invalid)

    image_path = tmp_path / "input.jpg"
    Image.new("RGB", (9, 8), "red").save(image_path)
    scorer = object.__new__(blip_native.BlipPairScorer)
    scorer.max_pairs, scorer.max_tokens, scorer.image_size = 4, 128, 224

    class Tokens:
        shape = (2, 5)

    scorer.tokenizer = lambda texts, **kwargs: {"input_ids": Tokens()}
    decoded, options = [], []

    def process(images, **kwargs):
        decoded.extend(images)
        options.append(kwargs)
        assert len(images) == 1 and images[0].size == (9, 8)
        return {"pixel_values": "fixture-pixels"}

    scorer.processor = process
    _, _, positions = scorer._inputs(["cat", "dog"], [image_path, image_path])
    assert positions == [0, 0]
    assert options == [{"return_tensors": "pt", "size": {"height": 224, "width": 224}}]
    with pytest.raises(ValueError):
        decoded[0].getpixel((0, 0))  # Converted images are closed after preprocessing.
    scorer.image_size = 384
    scorer._inputs(["cat", "dog"], [image_path, image_path])
    assert options[-1] == {"return_tensors": "pt"}  # Existing processor defaults are preserved.


def test_cli_rejects_blip_size_for_other_models_before_dataset_or_model_loading(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "backend.app.evaluation.relation_benchmark",
            "evaluate",
            str(tmp_path / "unreadable-dataset"),
            "--model",
            "clip",
            "--model-dir",
            str(tmp_path / "unreadable-model"),
            "--image-size",
            "224",
            "--output",
            str(tmp_path / "must-not-write.json"),
        ],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert b"applies only to the explicit BLIP profile" in result.stderr
    assert not (tmp_path / "must-not-write.json").exists()
