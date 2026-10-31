"""Actual local speech inference controls; unscored without reference transcripts."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from backend.app.retrieval.hybrid import audio_onnx, snapshots


def main() -> None:
    """Load explicit model/wave paths and preserve actual repeated inference costs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wave", type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeat", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh speech report")
    digest = snapshots.sha256_file(args.wave)
    started = time.perf_counter()
    model = audio_onnx.WhisperOnnxTranscriber(args.model_dir)
    load_seconds = time.perf_counter() - started
    rows = []
    for index in range(args.repeat):
        started = time.perf_counter()
        segments = model.transcribe(args.wave)
        rows.append(
            {
                "repeat": index + 1,
                "seconds": time.perf_counter() - started,
                "segments": [dataclasses.asdict(segment) for segment in segments],
            }
        )
    if snapshots.sha256_file(args.wave) != digest:
        raise ValueError("Source waveform changed during inference")
    report = {
        "schema": "lentica-speech-inference-v1",
        "wave_sha256": digest,
        "model_version": model.version,
        "model_load_seconds": load_seconds,
        "runs": rows,
        "repeated_transcripts_equal": all(row["segments"] == rows[0]["segments"] for row in rows),
        "quality_scored": False,
        "scope": "Actual CPU decoding, coarse chunk locators; no checked reference transcript, speech accuracy, speaker/sound event or word-time alignment claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
