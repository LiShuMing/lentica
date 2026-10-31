"""Prepare selected public images, lock selection, then evaluate held-out sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.evaluation.hybrid_eval import image_quality, mirror_images
from backend.app.retrieval.hybrid import clip_onnx, siglip_onnx


def main() -> None:
    """Keep preparation/selection/test IO explicit and refuse artifact overwrites."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("archive", type=Path)
    prepare.add_argument("--media-dir", required=True, type=Path)
    prepare.add_argument("--exclude-dataset", action="append", default=[], type=Path)
    for command in ("select", "evaluate"):
        child = commands.add_parser(command)
        child.add_argument("dataset", type=Path)
        child.add_argument("--model-dir", required=True, type=Path)
        child.add_argument("--encoder", choices=("clip", "siglip"), default="clip")
        child.add_argument("--threads", type=int, default=2)
        child.add_argument("--batch-size", type=int, default=8)
        child.add_argument("--selection", required=True, type=Path)
        if command == "select":
            child.add_argument("--protocol", choices=("similarity-v1", "margin-v2"), default="similarity-v1")
        if command == "evaluate":
            child.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        source = mirror_images.prepare(
            args.archive, args.media_dir, exclude=[mirror_images.load(path) for path in args.exclude_dataset]
        )
        print(
            json.dumps(
                {
                    "dataset": str(source.directory),
                    "identity": source.identity,
                    "counts": {name: len(source.partition(name)) for name in ("dev", "validation", "test")},
                }
            )
        )
        return
    target = args.selection if args.command == "select" else args.output
    if target.exists():
        parser.error("Choose a fresh selection/report path")
    source = mirror_images.load(args.dataset)
    model_type = siglip_onnx.SiglipOnnxEncoder if args.encoder == "siglip" else clip_onnx.ClipOnnxEncoder
    model = model_type(args.model_dir, threads=args.threads, batch_size=args.batch_size)
    if args.command == "select":
        report = image_quality.select(source, model, protocol=args.protocol)
        summary = {key: report[key] for key in ("selected_threshold", "accepted", "sha256")}
    else:
        if args.selection.stat().st_size > 8 * 1024**2:
            parser.error("Image selection artifact exceeds 8 MiB")
        selection = json.loads(args.selection.read_bytes())
        report = image_quality.evaluate(source, model, selection)
        summary = {
            key: report[key]
            for key in (
                "quality",
                "selected_no_answer",
                "heldout_gate_passed",
                "image_to_label_accuracy",
                "ranking_sha256",
            )
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(target), **summary}))


if __name__ == "__main__":
    main()
