"""Explicit CPU selection, then locked evaluation; no downloads or application IO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.evaluation.hybrid_eval.dataset import Dataset as Dataset
from backend.app.evaluation.hybrid_eval.dataset import case_identity as case_identity
from backend.app.evaluation.hybrid_eval.dataset import dataset_identity as dataset_identity
from backend.app.evaluation.hybrid_eval.dataset import load_dataset as load_dataset
from backend.app.evaluation.hybrid_eval.encoders import LatentSemanticEncoder as LatentSemanticEncoder
from backend.app.evaluation.hybrid_eval.execution import BASELINE as BASELINE
from backend.app.evaluation.hybrid_eval.execution import METRICS as METRICS
from backend.app.evaluation.hybrid_eval.execution import build_routes as build_routes
from backend.app.evaluation.hybrid_eval.execution import cache_pools as cache_pools
from backend.app.evaluation.hybrid_eval.execution import config_for as config_for
from backend.app.evaluation.hybrid_eval.execution import evaluate_config as evaluate_config
from backend.app.evaluation.hybrid_eval.execution import live_pipeline as live_pipeline
from backend.app.evaluation.hybrid_eval.execution import read_config as read_config
from backend.app.evaluation.hybrid_eval.execution import route_definitions as route_definitions
from backend.app.evaluation.hybrid_eval.reporting import CountingStore as CountingStore
from backend.app.evaluation.hybrid_eval.reporting import digest as digest
from backend.app.evaluation.hybrid_eval.reporting import environment as environment
from backend.app.evaluation.hybrid_eval.reporting import paired_comparison as paired_comparison
from backend.app.evaluation.hybrid_eval.reporting import percentile as percentile
from backend.app.evaluation.hybrid_eval.workflow import SCHEMA as SCHEMA
from backend.app.evaluation.hybrid_eval.workflow import best_name as best_name
from backend.app.evaluation.hybrid_eval.workflow import choose_validation as choose_validation
from backend.app.evaluation.hybrid_eval.workflow import evaluate as evaluate
from backend.app.evaluation.hybrid_eval.workflow import normalized_query as normalized_query
from backend.app.evaluation.hybrid_eval.workflow import partition_training as partition_training
from backend.app.evaluation.hybrid_eval.workflow import select as select
from backend.app.evaluation.hybrid_eval.workflow import validate_lock as validate_lock


def main() -> None:
    """Parse explicit local CLI inputs, run the experiment and write its report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("select", "evaluate"))
    parser.add_argument("dataset", type=Path, help="Explicit local BEIR archive, with train and test qrels")
    parser.add_argument("--selection", type=Path, required=True, help="Selection output/input JSON")
    parser.add_argument("--output", type=Path, help="Required evaluation output JSON")
    args = parser.parse_args()
    if args.action == "evaluate" and not args.output:
        parser.error("evaluate requires --output")
    if args.action == "select" and args.selection.exists():
        parser.error("selection already exists; use a new path to preserve the locked experiment")
    if args.action == "evaluate" and args.output.resolve() in {args.dataset.resolve(), args.selection.resolve()}:
        parser.error("evaluation output must preserve dataset and selection artifact")
    train = load_dataset(args.dataset, split="train")
    test = load_dataset(args.dataset, split="test")
    if args.action == "select":
        report = select(train, test, progress=True)
        destination = args.selection
        summary = {key: report[key] for key in ("partition", "validation", "selected_family", "lock_sha256")}
    else:
        if args.selection.stat().st_size > 8 * 1024 * 1024:
            parser.error("selection artifact exceeds 8 MiB")
        report = evaluate(json.loads(args.selection.read_text(encoding="utf-8")), train, test, progress=True)
        destination = args.output
        summary = {key: report[key] for key in ("comparisons", "selected_vs_baseline", "ranking_sha256")}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x" if args.action == "select" else "w", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
