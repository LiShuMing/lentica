"""Profile development runtimes or measure fixed bilingual held-out prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.evaluation.hybrid_eval import language_quality, mirror_images, vision_runtime


def main() -> None:
    """Require explicit local paths and fresh profile/language output artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("profile", "language"))
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh vision iteration report")
    source = mirror_images.load(args.dataset)
    if args.command == "profile":
        if args.runtime:
            parser.error("Runtime profiling does not consume an existing lock")
        report = vision_runtime.profile(source, args.model_dir)
        summary = {"selected_index": report["selected_index"]}
    else:
        if not args.runtime or args.runtime.stat().st_size > 1024**2:
            parser.error("Language evaluation requires an explicit bounded frozen runtime")
        model = vision_runtime.frozen_model(source, args.model_dir, json.loads(args.runtime.read_bytes()))
        report = language_quality.evaluate(source, model)
        summary = {name: row["metrics"] for name, row in report["groups"].items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), **summary}))


if __name__ == "__main__":
    main()
