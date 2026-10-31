"""Reference-only WER, retained failures and explicit embedded-source controls."""

from __future__ import annotations

import io
import subprocess
import sys
import types
import wave

import pytest

from backend.app.evaluation.hybrid_eval import speech_quality, speech_source


def _wav(*, rate=16000):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(rate)
        target.writeframes(b"\0\0" * 160)
    return stream.getvalue()


def test_wer_counts_insertions_deletions_substitutions_and_normalizes():
    assert speech_quality.words("Ｈｅｌｌｏ, WORLD!  don't") == ("hello", "world", "don", "t")
    reference = ("a", "b", "c")
    assert speech_quality.edit_distance(reference, ("a", "z", "c", "extra")) == 2
    assert speech_quality.edit_distance(reference, ()) == 3
    assert speech_quality.edit_distance((), reference) == 3
    assert speech_quality.edit_distance(reference, reference) == 0
    with pytest.raises(ValueError, match="budget"):
        speech_quality.edit_distance(["word"] * 2001, [])


def test_preparation_uses_fixed_sorted_embedded_audio_and_detects_mutation(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("av")
    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": f"id-{index:03d}",
                    "text": "independent reference",
                    "audio": {"bytes": _wav(), "path": "/must/not/open"},
                }
                for index in reversed(range(14))
            ]
        ),
        source,
    )
    directory = tmp_path / "prepared"
    manifest = speech_source.prepare(source, directory)
    assert [item["source_id"] for item in manifest["items"]] == [f"id-{index:03d}" for index in range(12)]
    assert manifest["source_rows"] == 14
    assert all(item["duration_seconds"] == 0.01 for item in manifest["items"])
    seen = []

    def transcribe(path):
        seen.append(path.name)
        if path.name == "clip-00000.wav":
            raise RuntimeError("provider failed")
        return (types.SimpleNamespace(text="independent reference"),)

    model = types.SimpleNamespace(version="fixture-not-accuracy", transcribe=transcribe)
    report = speech_quality.evaluate(directory, manifest, model)
    assert len(seen) == 12 and report["failures"] == 1
    assert report["reference_words"] == 24 and report["errors"] == 2
    assert report["micro_wer"] == pytest.approx(1 / 12)
    path = directory / "clip-00000.wav"
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ValueError, match="identity"):
        speech_quality.evaluate(directory, manifest, model)
    with pytest.raises(ValueError, match="hash"):
        speech_source.load(directory)


def test_native_decoder_rejects_implicit_resampling(tmp_path):
    pytest.importorskip("av")
    with pytest.raises(ValueError, match="native16k"):
        speech_source._decode(_wav(rate=44100), tmp_path / "not-written.wav")
    assert not (tmp_path / "not-written.wav").exists()


def test_speech_cli_help_imports_without_optional_dependencies():
    subprocess.run(
        [sys.executable, "-S", "-m", "backend.app.evaluation.speech_benchmark", "--help"],
        check=True,
        timeout=10,
        capture_output=True,
    )
