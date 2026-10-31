"""Sound protocol and graph controls; synthetic scores are not CLAP accuracy."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import types
import wave

import pytest

from backend.app.evaluation.hybrid_eval import reporting, sound_quality, sound_source
from backend.app.retrieval.hybrid import clap_onnx


def test_sound_import_and_help_need_no_optional_runtime():
    subprocess.run(
        [sys.executable, "-S", "-m", "backend.app.evaluation.sound_benchmark", "--help"],
        check=True,
        capture_output=True,
        timeout=10,
    )


def _wave(path, *, rate=48000, channels=1, frames=48):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\0\0" * frames * channels)


def test_clap_rejects_random_crop_inputs_and_truncated_pcm(tmp_path):
    path = tmp_path / "sound.wav"
    _wave(path)
    assert clap_onnx.load_wave(path).shape == (48,)
    for kwargs in ({"rate": 44100}, {"channels": 2}, {"frames": 0}, {"frames": 480001}):
        _wave(path, **kwargs)
        with pytest.raises(ValueError):
            clap_onnx.load_wave(path)
    _wave(path)
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ValueError, match="truncated"):
        clap_onnx.load_wave(path)


def test_clap_text_export_never_pads_or_combines_different_length_queries():
    import numpy as np

    model = object.__new__(clap_onnx.ClapOnnxEncoder)
    model.dimensions = 512
    seen = []

    def tokenize(texts, **kwargs):
        assert len(texts) == 1 and kwargs["padding"] is False
        seen.append(texts[0])
        return {"input_ids": np.ones((1, len(texts[0]) + 2), dtype=np.int64)}

    def run(names, inputs):
        assert names == ["text_embeds"] and set(inputs) == {"input_ids"}
        assert inputs["input_ids"].shape[0] == 1
        return [np.ones((1, 512), dtype=np.float32)]

    model.tokenizer = tokenize
    model.text_session = types.SimpleNamespace(run=run)
    rows = model.encode_texts(["dog", "longer rain prompt"])
    assert seen == ["dog", "longer rain prompt"]
    assert np.array_equal(rows[0], model.encode_query("dog"))
    assert model.encode_texts([]).shape == (0, 512)
    with pytest.raises(ValueError):
        model.encode_texts("dog")


def _source(directory):
    items = []
    for partition in ("dev", "validation", "test"):
        for label in sound_source.CLASSES:
            for _ in range(2):
                index = len(items)
                items.append(
                    sound_source.SoundItem(
                        index, str(index), f"original-{index}", partition, label, f"clip-{index}.wav", "raw", "clip"
                    )
                )
    return sound_source.SoundDataset(directory, tuple(items), "fixture-source", {})


class ScoreFixture:
    encoder_id = "synthetic-sound-protocol"

    def __init__(self, source):
        self.by_file = {item.filename: item for item in source.items}
        self.seen = []

    def encode_audios(self, paths):
        import numpy as np

        items = [self.by_file[path.name] for path in paths]
        self.seen.extend(item.partition for item in items)
        return np.eye(5)[[sound_source.CLASSES.index(item.label) for item in items]]

    def encode_texts(self, texts):
        import numpy as np

        assert tuple(texts) == sound_quality.PROMPTS
        return np.eye(5)


def test_sound_selection_freezes_without_test_and_rejects_changed_source(tmp_path):
    source = _source(tmp_path)
    model = ScoreFixture(source)
    selection = sound_quality.select(source, model)
    assert set(model.seen) == {"dev", "validation"}
    assert selection["accepted"]
    assert len(selection["validation"]["thresholds"]) == 1
    model.seen.clear()
    lock = json.loads(json.dumps(selection))
    report = sound_quality.evaluate(source, model, lock)
    assert set(model.seen) == {"test"}
    assert report["eligible_for_promotion"]
    assert len(report["rankings"]["thresholds"]) == 1
    assert all(int(key) not in ids for key, ids in report["rankings"]["sound"].items())
    model.seen.clear()
    with pytest.raises(ValueError, match="identity"):
        sound_quality.evaluate(dataclasses.replace(source, identity="changed"), model, lock)
    assert not model.seen


def test_good_test_cannot_reverse_failed_validation(tmp_path):
    source = _source(tmp_path)
    model = ScoreFixture(source)
    lock = sound_quality.select(source, model)
    lock.update(accepted=False, selected_margin=-1.0)
    lock.pop("sha256")
    lock["sha256"] = reporting.digest(lock)
    result = sound_quality.evaluate(source, model, lock)
    assert result["heldout_gate_passed"]
    assert not result["eligible_for_promotion"]
