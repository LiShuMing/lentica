"""Explicit bounded video frame/window artifacts, separate from model execution."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from backend.app.retrieval.hybrid import artifacts, snapshots, units, video_frames, video_windows

from . import reporting, video_source

MAX_FRAME_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class PreparedVideo:
    """Blind candidate windows linked to a separately stored reference source."""

    source: video_source.VideoCase
    sampled: video_frames.VideoFrames
    proposals: video_windows.WindowSet


def _read(path: Path) -> reporting.JsonObject:
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("Video window metadata exceeds4MiB")
    return json.loads(path.read_text(encoding="utf-8"))


def _record(case: video_source.VideoCase, proposals: video_windows.WindowSet) -> reporting.JsonObject:
    return {
        "asset_id": case.asset_id,
        "source_sha256": case.sha256,
        "directory": f"source-{case.asset_id:03d}",
        "window_version": proposals.version,
        "duration_seconds": proposals.duration_seconds,
        "windows": [
            {
                "unit_id": w.unit.unit_id,
                "start": w.unit.locator.start_seconds,
                "end": w.unit.locator.end_seconds,
                "frame_indices": [frame.frame_index for frame in w.frames],
            }
            for w in proposals.windows
        ],
    }


def prepare(source: video_source.VideoDataset, target: Path) -> tuple[PreparedVideo, ...]:
    """Actually decode frozen videos into fresh frame/window artifacts.

    References do not select times or frames. All input originals are read-only;
    any incomplete video, observed timeline drift or exhausted global2GiB encoded
    frame budget fails before publishing a complete top-level manifest.
    """
    checked = video_source.load(source.directory)
    if checked.identity != source.identity or checked.cases != source.cases:
        raise ValueError("Video window source snapshot changed")
    target = Path(target)
    target.mkdir()
    rows, prepared, total_bytes = [], [], 0
    started = time.perf_counter()
    for case in checked.cases:
        sampled = video_frames.sample(
            checked.directory / case.filename,
            target / f"source-{case.asset_id:03d}",
            asset_id=case.asset_id,
            first_unit_id=case.asset_id * 1000,
            interval_seconds=0.2,
            interval_rounding="nearest",
            max_seconds=42,
            max_samples=240,
            max_decoded_frames=6000,
            max_frame_pixels=16_000_000,
            max_output_bytes=MAX_FRAME_BYTES - total_bytes,
            metadata_values={"source": "public-charades", "partition": case.reference.partition},
        )
        total_bytes += sum(frame.path.stat().st_size for frame in sampled.samples)
        proposals = video_windows.build(sampled, first_unit_id=case.asset_id * 1000 + 500)
        rows.append(_record(case, proposals))
        prepared.append(PreparedVideo(case, sampled, proposals))
    manifest = {
        "schema": "lentica-video-proposals-v1",
        "dataset_identity": checked.identity,
        "encoded_frame_bytes": total_bytes,
        "decoding_seconds": time.perf_counter() - started,
        "videos": rows,
    }
    manifest["identity"] = reporting.digest(manifest)
    with (target / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return tuple(prepared)


def _sampled(directory: Path, case: video_source.VideoCase) -> video_frames.VideoFrames:
    manifest = _read(directory / "manifest.json")
    config = manifest["config"]
    expected_metadata = {"source": "public-charades", "partition": case.reference.partition}
    if (
        manifest["schema"] != "lentica-video-frames-v1"
        or manifest["complete"] is not True
        or manifest["source_sha256"] != case.sha256
        or config["asset_id"] != case.asset_id
        or config["metadata"] != expected_metadata
        or config["interval_seconds"] != 0.2
        or config.get("interval_rounding") != "nearest"
        or config["first_unit_id"] != case.asset_id * 1000
        or (config["max_seconds"], config["max_samples"], config["max_decoded_frames"], config["max_frame_pixels"])
        != (42, 240, 6000, 16_000_000)
        or not 16 <= len(manifest["samples"]) <= 240
    ):
        raise ValueError("Video sampled source/configuration differs")
    frames = []
    for index, row in enumerate(manifest["samples"]):
        name = row["file"]
        if not re.fullmatch(r"frame-\d{5}\.png", name) or (directory / name).is_symlink():
            raise ValueError("Video frame path escapes explicit sampling output")
        path = directory / name
        if path.stat().st_size > 8 * 1024**2 or snapshots.sha256_file(path) != row["sha256"]:
            raise ValueError("Video frame bytes differ")
        if (
            row["asset_id"] != case.asset_id
            or row["unit_id"] != case.asset_id * 1000 + index
            or row["frame_index"] != index * config["stride"]
            or row["metadata"] != expected_metadata
        ):
            raise ValueError("Video sampled frame filter identity differs")
        unit = units.RetrievalUnit(
            row["unit_id"],
            row["asset_id"],
            "video",
            units.Locator(start_seconds=row["start_seconds"], end_seconds=row["end_seconds"]),
            metadata=row["metadata"],
        )
        frames.append(video_frames.FrameSample(unit, path, row["frame_index"], row["sha256"]))
    version = artifacts.space_id(
        "video-frames-v1", {"source": case.sha256, **{frame.path.name: frame.sha256 for frame in frames}}, config
    )
    if version != manifest["version"]:
        raise ValueError("Video sampling producer identity differs")
    return video_frames.VideoFrames(
        directory, case.sha256, version, config["fps"], manifest["decoded_frames"], tuple(frames)
    )


def load(source: video_source.VideoDataset, directory: Path) -> tuple[PreparedVideo, ...]:
    """Verify only explicit manifests/frames and reproduce all blind proposal IDs."""
    directory = Path(directory).resolve()
    manifest = _read(directory / "manifest.json")
    identity = manifest.pop("identity")
    if (
        manifest["schema"] != "lentica-video-proposals-v1"
        or reporting.digest(manifest) != identity
        or manifest["dataset_identity"] != source.identity
        or len(manifest["videos"]) != len(source.cases)
    ):
        raise ValueError("Video proposal source/manifest identity differs")
    prepared, total = [], 0
    for row, case in zip(manifest["videos"], source.cases):
        name = f"source-{case.asset_id:03d}"
        if row["directory"] != name or (directory / name).is_symlink():
            raise ValueError("Video sampling directory identity differs")
        sampled = _sampled(directory / name, case)
        total += sum(frame.path.stat().st_size for frame in sampled.samples)
        proposals = video_windows.build(sampled, first_unit_id=case.asset_id * 1000 + 500)
        if row != _record(case, proposals):
            raise ValueError("Video proposal times/frames differ from blind construction")
        prepared.append(PreparedVideo(case, sampled, proposals))
    if total != manifest["encoded_frame_bytes"] or total > MAX_FRAME_BYTES:
        raise ValueError("Video proposal frame byte count/budget differs")
    return tuple(prepared)
