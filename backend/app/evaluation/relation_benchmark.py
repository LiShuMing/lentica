"""Prepare explicit relation pixels or measure fixed local dual/joint models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from backend.app.evaluation.hybrid_eval import relation_quality, relation_source
from backend.app.retrieval.hybrid import blip_native, clip_onnx, siglip_onnx


def _progress(done: int, total: int) -> None:
    if done % 10 == 0 or done == total:
        print(json.dumps({"completed_cases": done, "total_cases": total}), file=sys.stderr, flush=True)


def _evaluate(args, source):
    started = time.perf_counter()
    if args.model == "blip":
        model = blip_native.BlipPairScorer(args.model_dir, image_size=args.image_size or 384)
        loading = time.perf_counter() - started
        first = source.cases[0]
        if first.partition != "dev":
            raise ValueError("Native equivalence control must be on development")
        equivalence = model.verify_native_equivalence(first.captions, [source.directory / first.filename] * 2)
        report = relation_quality.measure_joint(source, model, progress=_progress if args.progress else None)
        report["native_equivalence"] = equivalence
        report["model_encoding"] = dict(model.encoding)
        report["model_input_size"] = model.image_size
    else:
        cls = clip_onnx.ClipOnnxEncoder if args.model == "clip" else siglip_onnx.SiglipOnnxEncoder
        model = cls(args.model_dir)
        loading = time.perf_counter() - started
        report = relation_quality.measure_dual(source, model, progress=_progress if args.progress else None)
    report["model_loading_seconds"] = loading
    report["model_kind"] = args.model
    return report


def main() -> None:
    """Keep model construction, explicit paths and fresh output outside library logic."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--replace-rel", required=True, type=Path)
    prepare.add_argument("--swap-obj", required=True, type=Path)
    prepare.add_argument("--images", required=True, type=Path)
    prepare.add_argument("--exclude-dataset", type=Path, help="Verify and exclude a previous source/pixel snapshot")
    prepare.add_argument("--output", required=True, type=Path)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("--model", required=True, choices=("clip", "siglip", "blip"))
    evaluate.add_argument("--model-dir", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--progress", action="store_true")
    evaluate.add_argument(
        "--image-size",
        type=int,
        choices=(224, 256, 288, 320, 384),
        help="Explicit BLIP input size; other models reject this option",
    )
    compare = commands.add_parser("compare")
    compare.add_argument("dataset", type=Path)
    for name in ("clip", "siglip", "blip"):
        compare.add_argument("--" + name + "-report", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh dataset/report path")
    if args.command == "evaluate" and args.model != "blip" and args.image_size is not None:
        parser.error("--image-size applies only to the explicit BLIP profile")
    if args.command == "prepare":
        source = relation_source.prepare(
            {"replace_rel": args.replace_rel, "swap_obj": args.swap_obj},
            args.images,
            args.output,
            exclude=relation_source.load(args.exclude_dataset) if args.exclude_dataset else None,
        )
        print(json.dumps({"dataset": str(source.directory), "identity": source.identity, "cases": len(source.cases)}))
        return
    started = time.perf_counter()
    source = relation_source.load(args.dataset)
    checking = time.perf_counter() - started
    if args.command == "evaluate":
        report = _evaluate(args, source)
        report["data_verification_seconds"] = checking
        summary = report["overall"]
    else:
        reports = {}
        for name in ("clip", "siglip", "blip"):
            path = getattr(args, name + "_report")
            if path.stat().st_size > 4 * 1024**2:
                parser.error("Relation report exceeds4MiB")
            reports[name] = json.loads(path.read_text(encoding="utf-8"))
        report = relation_quality.compare(source, reports)
        summary = {"eligible_for_promotion": report["eligible_for_promotion"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), **summary}))


if __name__ == "__main__":
    main()
