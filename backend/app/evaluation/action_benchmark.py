"""One public source-labelled temporal-model demonstration and fixed controls."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from backend.app.evaluation.hybrid_eval import action_quality, reporting
from backend.app.retrieval.hybrid import action_model, snapshots, video_frames


def run(source: Path, directory: Path, model_dir: Path) -> reporting.JsonObject:
    """Construct explicit source/model adapters and delegate reusable evaluation."""
    sampled = video_frames.sample(source, directory, asset_id=1, first_unit_id=100, interval_seconds=0.2)
    started = time.perf_counter()
    model = action_model.VideoMaeClassifier(model_dir)
    loading = time.perf_counter() - started
    report = action_quality.evaluate(sampled, model)
    if snapshots.sha256_file(source) != sampled.source_sha256:
        raise ValueError("Action source changed after frame sampling")
    report["model_loading_seconds"] = loading
    return report


def main() -> None:
    """Keep downloads and model work out of import/help; refuse old outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--media-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh action demonstration report")
    report = run(args.source, args.media_dir, args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "results": report["results"]}))


if __name__ == "__main__":
    main()
