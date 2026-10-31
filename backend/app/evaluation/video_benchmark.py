"""Prepare frozen public multi-action videos and blind fixed-time proposals."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from backend.app.evaluation.hybrid_eval import video_quality, video_source, video_window_io


def _evaluate(args: argparse.Namespace) -> dict:
    """Construct only the explicitly selected local CPU provider after parsing."""
    source = video_source.load(args.dataset)
    prepared = video_window_io.load(source, args.windows)
    started = time.perf_counter()
    if args.route == "videomae":
        from backend.app.retrieval.hybrid import action_model

        model = action_model.VideoMaeClassifier(args.model_dir)
        providers = {"action": model}
    else:
        from backend.app.retrieval.hybrid import siglip_onnx

        model = siglip_onnx.SiglipOnnxEncoder(args.model_dir)
        providers = {"image": model}
    loading = time.perf_counter() - started
    report = video_quality.evaluate(
        source,
        prepared,
        route=args.route,
        progress=lambda identifier: print(json.dumps({"route": args.route, "finished_source": identifier}), flush=True),
        **providers,
    )
    report["model_loading_seconds"] = loading
    video_source.load(args.dataset)
    return report


def main() -> None:
    """Keep explicit artifact IO separate from reusable source/window contracts."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--selection", type=Path, required=True)
    prepare.add_argument("--sources", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    windows = commands.add_parser("windows")
    windows.add_argument("dataset", type=Path)
    windows.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("--windows", type=Path, required=True)
    evaluate.add_argument("--route", choices=("videomae", "siglip"), required=True)
    evaluate.add_argument("--model-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--native", type=Path, required=True)
    compare.add_argument("--single-frame", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh video artifact path")
    if args.command in {"evaluate", "compare"}:
        if args.command == "evaluate":
            report = _evaluate(args)
        else:
            reports = []
            for path in (args.native, args.single_frame):
                if path.stat().st_size > 32 * 1024**2:
                    parser.error("Video report exceeds32MiB")
                reports.append(json.loads(path.read_text(encoding="utf-8")))
            report = video_quality.compare(*reports)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        summary = report.get("gates") or {name: row["summary"] for name, row in report["partitions"].items()}
        print(json.dumps({"output": str(args.output), "results": summary}))
        return
    if args.command == "prepare":
        source = video_source.prepare(args.selection, args.sources, args.output)
        summary = {"identity": source.identity, "videos": len(source.cases)}
    else:
        source = video_source.load(args.dataset)
        result = video_window_io.prepare(source, args.output)
        summary = {
            "dataset_identity": source.identity,
            "videos": len(result),
            "windows": sum(len(row.proposals.windows) for row in result),
            "sampled_frames": sum(len(row.sampled.samples) for row in result),
        }
    print(json.dumps({"output": str(args.output), **summary}))


if __name__ == "__main__":
    main()
