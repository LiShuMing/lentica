"""Blind fixed-time video proposals from already verified real frame samples."""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import artifacts, contracts, units, video_frames


@dataclass(frozen=True)
class VideoWindow:
    """One proposal and16 actual ordered input frames, with no relevance labels."""

    unit: units.RetrievalUnit
    frames: tuple[video_frames.FrameSample, ...]

    @property
    def middle_frame(self) -> video_frames.FrameSample:
        """Return a fixed single-frame baseline input, independent of annotations."""
        return self.frames[len(self.frames) // 2]


@dataclass(frozen=True)
class WindowSet:
    """Versioned fixed proposals; the covering time is not an action boundary."""

    version: str
    duration_seconds: float
    windows: tuple[VideoWindow, ...]


def build(
    sampled: video_frames.VideoFrames,
    *,
    first_unit_id: int,
    window_seconds: float = 4.0,
    stride_seconds: float = 2.0,
    max_windows: int = 32,
) -> WindowSet:
    """Build full windows and a final tail window without reading labels/pixels.

    Args:
        sampled: Explicit verified real-frame sampling result from one source.
        first_unit_id: Fresh integer namespace, disjoint from source-frame units.
        window_seconds: Fixed full-window support, within.5..30seconds.
        stride_seconds: Fixed start spacing, positive and no larger than a window.
        max_windows: Hard proposal budget; no silent truncation or frame repetition.

    Returns:
        Original ordered frames selected uniformly within each blind proposal.
        Input metadata/parent IDs are retained; no evidence text is invented.

    Raises:
        ValueError: Invalid timeline, identity, insufficient real frames or budgets.
    """
    contracts.asset_id(first_unit_id)
    if (
        type(window_seconds) not in (int, float)
        or not math.isfinite(window_seconds)
        or not 0.5 <= window_seconds <= 30
        or type(stride_seconds) not in (int, float)
        or not math.isfinite(stride_seconds)
        or not 0 < stride_seconds <= window_seconds
        or type(max_windows) is not int
        or not 1 <= max_windows <= 200
        or type(sampled.decoded_frames) is not int
        or sampled.decoded_frames <= 0
        or type(sampled.frames_per_second) not in (int, float)
        or not math.isfinite(sampled.frames_per_second)
        or not 0 < sampled.frames_per_second <= 120
        or not sampled.samples
    ):
        raise ValueError("Video proposal timeline/window budgets are invalid")
    duration = sampled.decoded_frames / sampled.frames_per_second
    if not window_seconds <= duration <= 120:
        raise ValueError("Video duration cannot support bounded full proposals")
    first = sampled.samples[0].unit
    previous = -1
    frame_ids = set()
    for frame in sampled.samples:
        if (
            frame.unit.asset_id != first.asset_id
            or frame.unit.kind != "video"
            or frame.unit.metadata != first.metadata
            or type(frame.frame_index) is not int
            or not previous < frame.frame_index < sampled.decoded_frames
            or frame.unit.unit_id in frame_ids
            or frame.unit.locator.start_seconds is None
            or abs(frame.unit.locator.start_seconds - frame.frame_index / sampled.frames_per_second) > 1e-7
            or abs(frame.unit.locator.end_seconds - (frame.frame_index + 1) / sampled.frames_per_second) > 1e-7
        ):
            raise ValueError("Video proposal frame identities or real timeline differ")
        previous = frame.frame_index
        frame_ids.add(frame.unit.unit_id)
    last_start = duration - window_seconds
    quotient = last_start / stride_seconds
    if not math.isfinite(quotient) or quotient >= max_windows:
        raise ValueError("Video proposal window budget exceeded")
    count = math.floor(quotient + 1e-9) + 1
    if count > max_windows:
        raise ValueError("Video proposal window budget exceeded")
    starts = [index * stride_seconds for index in range(count)]
    if not math.isclose(starts[-1], last_start, abs_tol=1e-7):
        starts.append(last_start)
    if len(starts) > max_windows:
        raise ValueError("Video proposal window budget exceeded")
    contracts.asset_id(first_unit_id + len(starts) - 1)
    if frame_ids & set(range(first_unit_id, first_unit_id + len(starts))):
        raise ValueError("Video proposal IDs collide with input-frame units")
    windows = []
    for index, start in enumerate(starts):
        end = start + window_seconds
        available = [
            frame
            for frame in sampled.samples
            if frame.unit.locator.start_seconds >= start - 1e-8 and frame.unit.locator.end_seconds <= end + 1e-8
        ]
        if len(available) < 16:
            raise ValueError("Video proposal needs16 distinct real frames; never repeat missing frames")
        frames = tuple(available[position * (len(available) - 1) // 15] for position in range(16))
        unit = units.RetrievalUnit(
            first_unit_id + index,
            first.asset_id,
            "video",
            units.Locator(start_seconds=start, end_seconds=end),
            metadata=first.metadata,
        )
        windows.append(VideoWindow(unit, frames))
    version = artifacts.space_id(
        "video-window-proposals-v1",
        {"source": sampled.source_sha256},
        {
            "sampling_version": sampled.version,
            "window_seconds": window_seconds,
            "stride_seconds": stride_seconds,
            "first_unit_id": first_unit_id,
            "max_windows": max_windows,
            "frames_per_window": 16,
            "windows": [
                {
                    "id": w.unit.unit_id,
                    "start": w.unit.locator.start_seconds,
                    "end": w.unit.locator.end_seconds,
                    "frames": [f.sha256 for f in w.frames],
                }
                for w in windows
            ],
        },
    )
    return WindowSet(version, duration, tuple(windows))
