"""Evaluate fixed manually annotated page-order anchors on an explicit public PDF."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from backend.app.evaluation.hybrid_eval import layout_quality, reporting
from backend.app.retrieval.hybrid import extraction, snapshots


def main() -> None:
    """Verify selected source/annotation identities before native PDF extraction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-column-gap", type=float, choices=(0.015, 0.03), default=0.03)
    args = parser.parse_args()
    if args.output.exists() or args.annotations.stat().st_size > 1024**2:
        parser.error("Choose fresh output and at most1MiB annotations")
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    if annotations["schema"] != "lentica-public-layout-anchors-v1" or (
        args.source.stat().st_size > 16 * 1024**2 or snapshots.sha256_file(args.source) != annotations["source_sha256"]
    ):
        parser.error("Public PDF/annotation source identity differs")
    started = time.perf_counter()
    records = extraction.pdf_blocks(args.source, asset_id=1, first_unit_id=100, max_pages=20, max_blocks=5000)
    extraction_seconds = time.perf_counter() - started
    chosen = [unit for unit in records if unit.locator.page == annotations["page"]]
    report = layout_quality.evaluate(chosen, annotations["anchors"], min_column_gap=args.min_column_gap)
    report.update(
        source_sha256=annotations["source_sha256"],
        annotation_sha256=reporting.digest(annotations),
        page=annotations["page"],
        extraction_seconds=extraction_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps({"output": str(args.output), **{name: report[name] for name in ["baseline", "xy_cut", "accepted"]}})
    )


if __name__ == "__main__":
    main()
