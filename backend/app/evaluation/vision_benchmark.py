"""Run the fixed third-source text/image controls after a SigLIP selection lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.evaluation.hybrid_eval import image_quality, mirror_images, vision_quality
from backend.app.retrieval.hybrid import clip_onnx, siglip_onnx


def main() -> None:
    """Load explicit local models/data; output one non-overwriting report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--clip-dir", required=True, type=Path)
    parser.add_argument("--siglip-dir", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.selection.stat().st_size > 8 * 1024**2:
        parser.error("Choose a fresh output and bounded frozen selection")
    source = mirror_images.load(args.dataset)
    model = siglip_onnx.SiglipOnnxEncoder(args.siglip_dir)
    selection = json.loads(args.selection.read_bytes())
    # Evaluate validates the frozen identity before encoding any test image.
    refusal = image_quality.evaluate(source, model, selection)
    report = vision_quality.compare(source, clip_onnx.ClipOnnxEncoder(args.clip_dir), model)
    report["frozen_refusal"] = refusal
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "quality_text": report["quality_text"],
                "quality_image": report["quality_image"],
                "refusal_passed": refusal["heldout_gate_passed"],
                "ranking_sha256": report["ranking_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
