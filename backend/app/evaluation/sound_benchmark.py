"""Prepare explicit public sounds, freeze development selection, then test once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.evaluation.hybrid_eval import sound_quality, sound_source
from backend.app.retrieval.hybrid import clap_onnx


def main() -> None:
    """Keep source preparation/selection/test writes separate and non-overwriting."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("source", type=Path)
    prepare.add_argument("--media-dir", required=True, type=Path)
    for name in ("select", "evaluate"):
        child = commands.add_parser(name)
        child.add_argument("dataset", type=Path)
        child.add_argument("--model-dir", required=True, type=Path)
        child.add_argument("--selection", required=True, type=Path)
        if name == "evaluate":
            child.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        source = sound_source.prepare(args.source, args.media_dir)
        print(
            json.dumps(
                {
                    "directory": str(source.directory),
                    "identity": source.identity,
                    "counts": {name: len(source.partition(name)) for name in ("dev", "validation", "test")},
                }
            )
        )
        return
    target = args.selection if args.command == "select" else args.output
    if target.exists():
        parser.error("Choose a fresh sound selection/report")
    source = sound_source.load(args.dataset)
    model = clap_onnx.ClapOnnxEncoder(args.model_dir)
    if args.command == "select":
        report = sound_quality.select(source, model)
        summary = {key: report[key] for key in ("accepted", "selected_margin", "sha256")}
    else:
        if args.selection.stat().st_size > 8 * 1024**2:
            parser.error("Selection exceeds8MiB")
        report = sound_quality.evaluate(source, model, json.loads(args.selection.read_bytes()))
        summary = {
            key: report[key]
            for key in (
                "quality",
                "selected_no_answer",
                "heldout_gate_passed",
                "eligible_for_promotion",
                "ranking_sha256",
            )
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(target), **summary}))


if __name__ == "__main__":
    main()
