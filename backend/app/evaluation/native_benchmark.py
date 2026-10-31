"""Measure fixed local pretrained text routes without changing product defaults."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.evaluation.hybrid_eval import dataset, native_quality


def main() -> None:
    """Require explicit local model/cache/data and a fresh report path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--corpus-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", default=2, type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh report path")
    if not 1 <= args.threads <= 32:
        parser.error("CPU threads must be between 1 and 32")
    import torch

    torch.set_num_threads(args.threads)
    report = native_quality.run(dataset.load_dataset(args.dataset), args.model_dir, args.corpus_cache, progress=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "metrics": {name: value["metrics"] for name, value in report["comparisons"].items()},
            }
        )
    )


if __name__ == "__main__":
    main()
