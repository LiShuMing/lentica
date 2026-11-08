"""Run isolated product benchmarks without reading a private product database."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    """Select explicit public/synthetic input and fresh paths before product imports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("documents", "scale", "ingestion", "mixed", "api"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 1000, 5000])
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    if work.exists() or args.output.exists():
        parser.error("Choose fresh work and output paths")
    if args.mode not in {"scale", "mixed"} and args.dataset is None:
        parser.error("Public document/ingestion benchmarks require --dataset")
    work.mkdir(parents=True, exist_ok=False)
    # Set isolation before importing any product module: configuration and the
    # legacy session factory are initialized at import time.
    os.environ.update(
        LENTICA_DATA_DIR=str(work / "runtime"),
        LENTICA_DOCUMENTS_SOURCE=str(work / "empty-documents"),
        LENTICA_PHOTOS_SOURCE=str(work / "empty-photos"),
        LENTICA_QUEUE_ENABLED="false",
        LENTICA_ENABLE_FACES="false",
        LENTICA_WORKER_MODE="smoke",
    )
    from backend.app.evaluation.hybrid_eval import dataset, product_controls, product_quality, reporting

    if args.mode == "scale":
        report = product_quality.scale(work, args.sizes, queries=args.queries)
    elif args.mode == "mixed":
        report = product_controls.mixed(work)
    else:
        source = dataset.load_dataset(args.dataset)
        if args.mode == "documents":
            report = product_quality.documents(source, work, repeats=args.repeats, progress=True)
        elif args.mode == "ingestion":
            report = product_quality.ingestion(source, work)
        else:
            report = product_controls.api(source, work)
    report["environment"] = reporting.environment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "schema": report["schema"]}))


if __name__ == "__main__":
    main()
