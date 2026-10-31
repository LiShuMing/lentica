"""Explicit sequential local video sampling with checked frame/time budgets."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from . import artifacts, contracts, snapshots, units


@dataclass(frozen=True)
class FrameSample:
    """One actual decoded frame and its narrow frame-period support interval."""

    unit: units.RetrievalUnit
    path: Path
    frame_index: int
    sha256: str


@dataclass(frozen=True)
class VideoFrames:
    """Completed explicit sampling artifact; partial directories are not readers."""

    directory: Path
    source_sha256: str
    version: str
    frames_per_second: float
    decoded_frames: int
    samples: tuple[FrameSample, ...]


def _properties(capture, cv2, *, max_seconds, max_decoded_frames, max_frame_pixels):
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if (
        not all(math.isfinite(value) for value in (fps, frames, width, height))
        or not 0 < fps <= 120
        or frames != int(frames)
        or not 0 < frames <= max_decoded_frames
        or frames / fps > max_seconds
        or min(width, height) <= 0
        or width * height > max_frame_pixels
    ):
        raise ValueError("Video requires bounded frame rate/count/duration/pixels")
    return fps, int(frames), (int(width), int(height))


def _decode(
    capture,
    cv2,
    directory,
    fps,
    expected,
    size,
    stride,
    max_samples,
    asset_id,
    first_unit_id,
    metadata_values,
    max_output_bytes,
):
    """Decode exactly the advertised count, checking each timestamp before save."""
    samples, written = [], 0
    for index in range(expected + 1):
        valid, pixels = capture.read()
        if not valid:
            if index != expected:
                raise ValueError("Video ended before its advertised frame count")
            return tuple(samples)
        if index >= expected or pixels.shape[:2] != (size[1], size[0]):
            raise ValueError("Video frame count/dimensions changed during decoding")
        timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        # A zero/unsupported timestamp stream fails from the second frame. The
        # bound detects observed VFR/drift; it is not a container-level proof of
        # constant physical capture rate or a calibrated time measurement.
        if not math.isfinite(timestamp) or abs(timestamp - index / fps) > 0.5 / fps:
            raise ValueError("Video timestamps cannot verify the selected constant-rate timeline")
        if index % stride:
            continue
        if len(samples) >= max_samples:
            raise ValueError("Video sample budget exceeded")
        path = directory / f"frame-{len(samples):05d}.png"
        if max_output_bytes is None:
            if not cv2.imwrite(str(path), pixels):
                raise OSError("Could not write selected decoded frame")
        else:
            valid_png, encoded = cv2.imencode(".png", pixels)
            if not valid_png:
                raise OSError("Could not encode selected decoded frame")
            payload = encoded.tobytes()
            if written + len(payload) > max_output_bytes:
                raise ValueError("Video frame output byte budget exceeded")
            with path.open("xb") as output:
                output.write(payload)
            written += len(payload)
        unit = units.RetrievalUnit(
            first_unit_id + len(samples),
            asset_id,
            "video",
            units.Locator(start_seconds=index / fps, end_seconds=(index + 1) / fps),
            metadata=metadata_values,
        )
        samples.append(FrameSample(unit, path, index, snapshots.sha256_file(path)))
    raise ValueError("Video did not end within its declared frame budget")


def sample(
    path: Path,
    directory: Path,
    *,
    asset_id: int,
    first_unit_id: int,
    interval_seconds: float = 1.0,
    max_seconds: int = 120,
    max_samples: int = 120,
    max_decoded_frames: int = 3600,
    max_frame_pixels: int = 4_000_000,
    max_output_bytes: int | None = None,
    interval_rounding: str = "ceil",
    metadata_values: Mapping[str, str | int | float | bool | None] | None = None,
) -> VideoFrames:
    """Decode explicit local AVI/MP4 sequentially into a fresh artifact directory.

    Source bytes are read-only. Timestamp/count/dimension drift, failed decode,
    changed source or budget excess fails without publishing a complete manifest.
    A failed new directory remains explicitly incomplete; old artifacts are not
    removed. No implicit media traversal, audio transcription or semantic action
    model is involved. Sparse sampling can miss short-lived events.
    Optional max_output_bytes bounds encoded PNG bytes before each file write;
    the original default/write path and version identity remain unchanged.
    Explicit interval_rounding='nearest' uses nearest integer frame spacing
    (half upward, at least one frame) for low-frame-rate temporal windows.
    The default 'ceil' retains the historical minimum frame separation.
    """
    contracts.asset_id(asset_id)
    contracts.asset_id(first_unit_id)
    if (
        type(interval_seconds) not in (int, float)
        or not math.isfinite(interval_seconds)
        or not 0.2 <= interval_seconds <= 30
        or interval_rounding not in {"ceil", "nearest"}
        or any(
            type(value) is not int or value <= 0
            for value in (max_seconds, max_samples, max_decoded_frames, max_frame_pixels)
        )
        or max_seconds > 120
        or max_samples > 600
        or max_decoded_frames > 14400
        or max_frame_pixels > 16_000_000
        or max_output_bytes is not None
        and (type(max_output_bytes) is not int or not 1 <= max_output_bytes <= 2 * 1024**3)
        or Path(directory).exists()
        or not Path(path).is_file()
        or Path(path).stat().st_size > 128 * 1024**2
    ):
        raise ValueError("Video requires explicit fresh paths and bounded sampling budgets")
    path, directory = Path(path), Path(directory)
    with path.open("rb") as source:
        header = source.read(12)
    if not (header[:4] == b"RIFF" and header[8:12] == b"AVI " or header[4:8] == b"ftyp"):
        raise ValueError("Only explicit local AVI/MP4 containers are supported")
    import cv2

    source_digest = snapshots.sha256_file(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("Selected local video cannot be decoded")
    created = False
    try:
        fps, frames, size = _properties(
            capture,
            cv2,
            max_seconds=max_seconds,
            max_decoded_frames=max_decoded_frames,
            max_frame_pixels=max_frame_pixels,
        )
        spacing = interval_seconds * fps
        stride = max(1, math.ceil(spacing) if interval_rounding == "ceil" else math.floor(spacing + 0.5))
        if math.ceil(frames / stride) > max_samples:
            raise ValueError("Requested video sampling exceeds sample budget")
        contracts.asset_id(first_unit_id + math.ceil(frames / stride) - 1)
        inherited = contracts.SearchFilter(equals=metadata_values or {}).equals
        directory.mkdir(parents=True, exist_ok=False)
        created = True
        samples = _decode(
            capture,
            cv2,
            directory,
            fps,
            frames,
            size,
            stride,
            max_samples,
            asset_id,
            first_unit_id,
            inherited,
            max_output_bytes,
        )
        if snapshots.sha256_file(path) != source_digest:
            raise ValueError("Video source changed during sampling")
        config = {
            "opencv": cv2.__version__,
            "interval_seconds": interval_seconds,
            "stride": stride,
            "fps": fps,
            "max_seconds": max_seconds,
            "max_samples": max_samples,
            "max_decoded_frames": max_decoded_frames,
            "max_frame_pixels": max_frame_pixels,
            "timeline": "sequential-checked-pos-msec-frame-period-v1",
            "asset_id": asset_id,
            "first_unit_id": first_unit_id,
            "metadata": dict(inherited),
        }
        if max_output_bytes is not None:
            config["max_output_bytes"] = max_output_bytes
        if interval_rounding != "ceil":
            config["interval_rounding"] = interval_rounding
        version = artifacts.space_id(
            "video-frames-v1", {"source": source_digest, **{row.path.name: row.sha256 for row in samples}}, config
        )
        manifest = {
            "schema": "lentica-video-frames-v1",
            "complete": True,
            "source_sha256": source_digest,
            "version": version,
            "config": config,
            "decoded_frames": frames,
            "samples": [
                {
                    "unit_id": row.unit.unit_id,
                    "asset_id": row.unit.asset_id,
                    "metadata": dict(row.unit.metadata),
                    "file": row.path.name,
                    "sha256": row.sha256,
                    "frame_index": row.frame_index,
                    "start_seconds": row.unit.locator.start_seconds,
                    "end_seconds": row.unit.locator.end_seconds,
                }
                for row in samples
            ],
        }
        with (directory / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        return VideoFrames(directory, source_digest, version, fps, frames, samples)
    except BaseException as exc:
        # Artifact publication/cleanup boundary: retain partial files and re-raise
        # the original failure, including cancellation. A marker is best effort;
        # its I/O failure must not replace the actual decode/publication error.
        if created:
            try:
                (directory / "incomplete.json").write_text(
                    json.dumps({"complete": False, "error": type(exc).__name__}) + "\n"
                )
            except OSError as marker_error:
                exc.add_note(f"Incomplete artifact marker failed: {type(marker_error).__name__}")
        raise
    finally:
        capture.release()
