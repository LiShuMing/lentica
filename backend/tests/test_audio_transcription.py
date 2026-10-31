"""WAV bounds and native greedy-loop controls; fixtures are not speech quality."""

from __future__ import annotations

import types
import wave

import pytest

from backend.app.retrieval.hybrid import audio_onnx


def _wave(path, *, rate=16000, channels=1, frames=160):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\0\0" * frames * channels)


def test_wave_format_duration_and_truncated_data_are_rejected(tmp_path):
    path = tmp_path / "input.wav"
    _wave(path)
    assert audio_onnx.load_wave(path).shape == (160,)
    for kwargs in ({"rate": 8000}, {"channels": 2}, {"frames": 0}, {"frames": 16001}):
        _wave(path, **kwargs)
        with pytest.raises(ValueError):
            audio_onnx.load_wave(path, max_seconds=1)
    _wave(path)
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ValueError, match="truncated"):
        audio_onnx.load_wave(path)


def test_greedy_honors_suppression_and_requires_eos_without_partial_results():
    import numpy as np

    model = object.__new__(audio_onnx.WhisperOnnxTranscriber)
    model.features = lambda samples, **kwargs: {"input_features": np.zeros((1, 80, 3000), dtype=np.float32)}
    model.encoder = types.SimpleNamespace(run=lambda *args: [np.zeros((1, 1500, 384), dtype=np.float32)])
    model.prefix, model.eos, model.timestamp_begin = (50257, 50362), 50256, 50363
    model.suppress, model.begin_suppress, model.max_tokens = (1,), (220, 50256), 8
    model.tokenizer = types.SimpleNamespace(decode=lambda ids, **kwargs: ",".join(map(str, ids)))

    def decode(names, inputs):
        size = inputs["input_ids"].shape[1]
        logits = np.zeros((1, size, 51864), dtype=np.float32)
        logits[0, -1, [1, 220, 50363, 50256, 7]] = [20, 10, 30, 9, 8]
        return [logits]

    model.decoder = types.SimpleNamespace(run=decode)
    # A begin-only token220 may be emitted later, so explicitly remove it from
    # subsequent candidates for the EOS check rather than changing model policy.
    model.suppress = (1, 220)
    assert model._transcribe_chunk(np.zeros(160, dtype=np.float32)) == "7"
    model.suppress = (1, 220, 50256)
    with pytest.raises(ValueError, match="without EOS"):
        model._transcribe_chunk(np.zeros(160, dtype=np.float32))
