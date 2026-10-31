"""Explicit reference speech preparation and fixed fresh-inference WER report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.evaluation.hybrid_eval import speech_quality, speech_source
from backend.app.retrieval.hybrid import audio_onnx


def main() -> None:
    """Keep reference preparation, model construction and fresh report IO apart."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("source", type=Path)
    prepare.add_argument("--media-dir", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("--model-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = speech_source.prepare(args.source, args.media_dir)
        print(json.dumps({"identity": manifest["identity"], "clips": len(manifest["items"])}))
        return
    if args.output.exists():
        parser.error("Choose a fresh speech quality report")
    manifest = speech_source.load(args.dataset)
    model = audio_onnx.WhisperOnnxTranscriber(args.model_dir)
    report = speech_quality.evaluate(args.dataset, manifest, model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("clips", "failures", "micro_wer", "transcript_sha256")}))


if __name__ == "__main__":
    main()
