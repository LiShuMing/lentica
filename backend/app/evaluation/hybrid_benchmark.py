"""Explicit, offline CPU benchmark for a local fixture or a selected BEIR archive.

No data/model download or application database access occurs on import or help.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.evaluation.hybrid_eval.benchmark import configurations as configurations
from backend.app.evaluation.hybrid_eval.benchmark import run as run
from backend.app.evaluation.hybrid_eval.dataset import Dataset as Dataset
from backend.app.evaluation.hybrid_eval.dataset import load_dataset as load_dataset
from backend.app.evaluation.hybrid_eval.encoders import LatentSemanticEncoder as LatentSemanticEncoder
from backend.app.evaluation.hybrid_eval.reporting import CountingStore as CountingStore
from backend.app.evaluation.hybrid_eval.reporting import percentile as percentile


def main() -> None:
    """Parse explicit local CLI inputs, run the experiment and write its report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", type=Path, help="Explicit BEIR zip or local fixture JSON; never downloaded automatically"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dimensions", type=int, default=128)
    args = parser.parse_args()
    if args.dimensions < 1:
        parser.error("dimensions must be positive")
    report = run(load_dataset(args.dataset), seed=args.seed, dimensions=args.dimensions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset": report["dataset"],
                "ranking_sha256": report["ranking_sha256"],
                "metrics": {name: row["metrics"]["overall"] for name, row in report["configurations"].items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
