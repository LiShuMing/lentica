"""Explicit local VideoMAE temporal classification, separate from frame retrieval."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from . import artifacts, snapshots, video_frames


@dataclass(frozen=True)
class ActionScores:
    """Native400-class logits; uncalibrated scores and explicit input provenance."""

    labels: tuple[str, ...]
    logits: tuple[float, ...]
    version: str
    input_kind: str

    def __post_init__(self) -> None:
        labels, logits = tuple(self.labels), tuple(self.logits)
        if (
            not 0 < len(labels) == len(logits) <= 400
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or any(type(value) not in (float, int) or not math.isfinite(value) for value in logits)
            or not isinstance(self.version, str)
            or not self.version
            or self.input_kind not in {"original", "reverse_control", "static_control"}
        ):
            raise ValueError(
                "Action scores require aligned bounded labels, finite logits and explicit model/input identity"
            )
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "logits", logits)

    def ranking(self) -> tuple[int, ...]:
        """Order class indices by native score with stable integer ties."""
        return tuple(sorted(range(len(self.labels)), key=lambda index: (-self.logits[index], index)))


def _validate(frames: Sequence[video_frames.FrameSample], input_kind: str) -> None:
    if len(frames) != 16 or input_kind not in {"original", "reverse_control", "static_control"}:
        raise ValueError("VideoMAE requires16 explicitly marked original/control frames")
    assets = {frame.unit.asset_id for frame in frames}
    indices = [frame.frame_index for frame in frames]
    if len(assets) != 1 or any(frame.unit.kind != "video" for frame in frames):
        raise ValueError("Temporal frames must belong to the same video resource")
    if any(frame.unit.locator.start_seconds is None for frame in frames):
        raise ValueError("Temporal frames require explicit time support")
    if input_kind == "original" and any(second <= first for first, second in zip(indices, indices[1:])):
        raise ValueError("Original temporal frames must have increasing indices")
    if input_kind == "original" and any(
        second.unit.locator.start_seconds < first.unit.locator.end_seconds for first, second in zip(frames, frames[1:])
    ):
        raise ValueError("Original temporal frame support must be ordered and nonoverlapping")
    if input_kind == "reverse_control" and any(second >= first for first, second in zip(indices, indices[1:])):
        raise ValueError("Reverse control frames must have decreasing indices")
    if input_kind == "static_control" and len(set(indices)) != 1:
        raise ValueError("Static control must explicitly repeat one frame")


class VideoMaeClassifier:
    """Native16-frame CPU model with mean-pooling/class head from local config.

    The class vocabulary is fixed. Whole-clip logits do not provide free-text
    embeddings, calibrated event probability or precise event-time boundaries.
    No model code or weights are downloaded during construction or inference.
    """

    def __init__(self, directory: Path, *, threads: int = 2, max_frame_pixels: int = 4_000_000) -> None:
        """Load verified local native weights; explicitly set process Torch threads."""
        if (
            type(threads) is not int
            or not 1 <= threads <= 32
            or type(max_frame_pixels) is not int
            or not 1 <= max_frame_pixels <= 16_000_000
        ):
            raise ValueError("VideoMAE CPU/frame budgets exceed supported limits")
        files = artifacts.verify(directory, ("config.json", "preprocessor_config.json", "model.safetensors"))
        import torch
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

        torch.set_num_threads(threads)
        self.processor = VideoMAEImageProcessor.from_pretrained(str(directory), local_files_only=True)
        self.model = (
            VideoMAEForVideoClassification.from_pretrained(
                str(directory),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                attn_implementation="eager",
            )
            .to("cpu")
            .eval()
        )
        config = self.model.config
        if (config.num_frames, config.image_size, config.num_channels, config.num_labels) != (16, 224, 3, 400):
            raise ValueError("Selected VideoMAE native temporal/image/class contract differs")
        self.labels = tuple(config.id2label[index] for index in range(config.num_labels))
        self.max_frame_pixels = max_frame_pixels
        self.version = artifacts.space_id(
            "videomae-native-v1",
            files,
            {
                "device": "cpu",
                "threads": threads,
                "frames": 16,
                "attention": "eager",
                "processor": "native-resize-center-crop-normalize",
                "max_frame_pixels": max_frame_pixels,
                "scores": "native-uncalibrated-class-logits",
                **{name: metadata.version(name) for name in ("torch", "transformers", "pillow")},
            },
        )

    def score(self, frames: Sequence[video_frames.FrameSample], *, input_kind: str = "original") -> ActionScores:
        """Read16 verified frames in declared order and run fresh native inference.

        Reversal/repeated-frame inputs must be marked as controls. Their outputs
        cannot be promoted into real ordered-sequence accuracy evidence.
        Frame files are bounded and checked again after reading; all images close
        even when processing or inference fails. Source videos remain read-only.
        """
        _validate(frames, input_kind)
        import torch
        from PIL import Image

        images = []
        try:
            for frame in frames:
                if frame.path.stat().st_size > 8 * 1024**2 or snapshots.sha256_file(frame.path) != frame.sha256:
                    raise ValueError("Selected temporal frame file identity/budget differs")
                with Image.open(frame.path) as image:
                    if image.width * image.height > self.max_frame_pixels:
                        raise ValueError("Selected temporal frame exceeds pixel budget")
                    images.append(image.convert("RGB"))
                if snapshots.sha256_file(frame.path) != frame.sha256:
                    raise ValueError("Selected temporal frame changed while reading")
            inputs = self.processor(images, return_tensors="pt")
            if tuple(inputs["pixel_values"].shape) != (1, 16, 3, 224, 224):
                raise ValueError("VideoMAE processor returned invalid temporal/image axes")
            with torch.inference_mode():
                values = self.model(**inputs).logits
            if tuple(values.shape) != (1, 400) or not torch.isfinite(values).all():
                raise ValueError("VideoMAE returned invalid native class scores")
            return ActionScores(self.labels, tuple(values[0].tolist()), self.version, input_kind)
        finally:
            for image in images:
                image.close()
