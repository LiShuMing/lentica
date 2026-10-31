"""Run frozen candidate-coverage and staged-reranking diagnostics on local data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.evaluation.hybrid_eval import dataset, quality


def main() -> None:
    """Read explicit inputs and write a new report, refusing accidental overwrite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new explicit report path")
    if args.selection.stat().st_size > 1024 * 1024:
        parser.error("Selection exceeds 1 MiB")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    train = dataset.load_dataset(args.dataset, split="train")
    test = dataset.load_dataset(args.dataset)
    report = quality.run(selection, train, test, progress=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "comparisons": report["comparisons"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
