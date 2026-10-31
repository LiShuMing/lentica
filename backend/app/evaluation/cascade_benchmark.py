"""Freeze source-isolated fast/slow image-gate selection before held-out testing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from backend.app.evaluation.hybrid_eval import (
    calibrated_rescue,
    cascade_quality,
    mirror_images,
    reporting,
    rescue_quality,
)
from backend.app.retrieval.hybrid import category_gating, clip_onnx, siglip_onnx


def _run(
    args: argparse.Namespace,
    source: mirror_images.ImageDataset,
    fast: clip_onnx.ClipOnnxEncoder,
    slow: siglip_onnx.SiglipOnnxEncoder,
    selection: reporting.JsonObject | None,
) -> reporting.JsonObject:
    """Dispatch explicit policies while preserving the original CLI defaults."""
    if args.command == "evaluate" and selection is None:
        raise ValueError("Cascade evaluation requires an explicit frozen selection")
    if args.policy == "calibrated-rescue":
        return (
            calibrated_rescue.select(source, fast, slow)
            if args.command == "select"
            else calibrated_rescue.evaluate(source, fast, slow, selection)
        )
    library = cascade_quality if args.policy == "strict-fast" else rescue_quality
    fast_margin = (
        args.fast_negative_margin if args.fast_negative_margin is not None else rescue_quality.FAST_NEGATIVE_MARGIN
    )
    options = {} if args.policy == "strict-fast" else {"fast_negative_margin": fast_margin}
    gate = category_gating.CascadedCategoryGate(
        fast,
        slow,
        cascade_quality.PROMPTS,
        cascade_quality.GALLERY,
        fast_negative_margin=None if args.policy == "strict-fast" else fast_margin,
    )
    return (
        library.select(source, gate, slow, **options)
        if args.command == "select"
        else library.evaluate(source, gate, slow, selection, **options)
    )


def main() -> None:
    """Construct explicit local adapters; never overwrite selection or reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "evaluate"))
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--clip-dir", required=True, type=Path)
    parser.add_argument("--siglip-dir", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--policy", choices=("strict-fast", "rescue-slow", "calibrated-rescue"), default="strict-fast")
    parser.add_argument("--fast-negative-margin", type=float, choices=(0.005, 0.02))
    args = parser.parse_args()
    if args.fast_negative_margin is not None and args.policy != "rescue-slow":
        parser.error("Fast negative margin requires the explicit rescue-slow policy")
    target = args.selection if args.command == "select" else args.output
    if target is None or target.exists():
        parser.error("Choose an explicit fresh selection/report")
    source = mirror_images.load(args.dataset)
    started = time.perf_counter()
    fast, slow = clip_onnx.ClipOnnxEncoder(args.clip_dir), siglip_onnx.SiglipOnnxEncoder(args.siglip_dir)
    loading = time.perf_counter() - started
    selection = None
    if args.command == "evaluate":
        if args.selection.stat().st_size > 8 * 1024**2:
            parser.error("Cascade selection exceeds8MiB")
        selection = json.loads(args.selection.read_bytes())
    report = _run(args, source, fast, slow, selection)
    if args.command == "select":
        summary = {key: report[key] for key in ("accepted", "selected_margin", "sha256")}
    else:
        report["model_loading_seconds"] = loading
        summary = {
            key: report[key] for key in ("selected_no_answer", "cost", "eligible_for_promotion", "ranking_sha256")
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(target), **summary}))


if __name__ == "__main__":
    main()
