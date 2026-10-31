"""Run explicit generated multimodal capability and evidence-family controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.evaluation.hybrid_eval import routed_controls


def main() -> None:
    """Generate into a fresh local directory and refuse report replacement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--groups", default=12, type=int)
    args = parser.parse_args()
    if args.output.exists() or args.media_dir.exists():
        parser.error("Choose fresh media and output paths")
    report = routed_controls.run(args.media_dir, groups=args.groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "metrics": {name: row["metrics"] for name, row in report["comparisons"].items()},
            }
        )
    )


if __name__ == "__main__":
    main()
