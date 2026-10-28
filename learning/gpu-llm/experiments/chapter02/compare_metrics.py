"""Compare metrics.json files produced by train_digits.py."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"metrics file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    header = f"{'run':<18} {'precision':<10} {'val_acc':>9} {'test_acc':>9} {'median_s':>10} {'peak_mib':>10}"
    print(header)
    print("-" * len(header))

    for path in args.metrics:
        metrics = load_metrics(path)
        history = metrics["history"]
        epoch_seconds = [float(epoch["seconds"]) for epoch in history]
        median_seconds = statistics.median(epoch_seconds)
        peak_memory = metrics.get("peak_gpu_memory_mib")
        peak_display = f"{peak_memory:.1f}" if peak_memory is not None else "n/a"
        print(
            f"{path.parent.name:<18} "
            f"{metrics['config']['precision']:<10} "
            f"{metrics['best_validation_accuracy']:>9.3f} "
            f"{metrics['test_accuracy']:>9.3f} "
            f"{median_seconds:>10.3f} "
            f"{peak_display:>10}"
        )


if __name__ == "__main__":
    main()
