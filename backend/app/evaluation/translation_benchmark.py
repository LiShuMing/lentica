"""Evaluate explicit local Chinese query adaptation against fixed image controls."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from backend.app.evaluation.hybrid_eval import mirror_images, translation_quality
from backend.app.retrieval.hybrid import marian_onnx, siglip_onnx


def main() -> None:
    """Keep local model construction and fresh report IO outside evaluation logic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--translation-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh translation report")
    source = mirror_images.load(args.dataset)
    started = time.perf_counter()
    translator = marian_onnx.MarianOnnxTranslator(args.translation_dir)
    model = siglip_onnx.SiglipOnnxEncoder(args.model_dir)
    loading_seconds = time.perf_counter() - started
    report = translation_quality.evaluate(source, model, translator)
    report["model_loading_seconds"] = loading_seconds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "failed_queries": report["failed_queries"],
                "metrics": {name: group["metrics"] for name, group in report["groups"].items()},
                "ranking_sha256": report["ranking_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
