"""Verify an explicit numeric table with frozen visual values and typed queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.evaluation.hybrid_eval import table_quality


def main() -> None:
    """Read explicit bounded references and write a fresh, independently scored report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--header-rows", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.reference.stat().st_size > 1024**2:
        parser.error("Choose fresh output and at most1MiB reference")
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    report = table_quality.evaluate(args.source, reference, header_rows=args.header_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "accepted": report["accepted"],
                "structure": {key: value for key, value in report["structure"].items() if key != "per_row"},
                "retrieval": {key: value for key, value in report["retrieval"].items() if key != "per_query"},
            }
        )
    )


if __name__ == "__main__":
    main()
