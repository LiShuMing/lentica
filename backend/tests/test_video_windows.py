"""Blind real-frame proposals and bounded output; no semantic quality assertions."""

import dataclasses
import hashlib
import json
import subprocess
import sys

import pytest

from backend.app.retrieval.hybrid import units, video_frames, video_windows


def sampled(tmp_path, count=170):
    frames = tuple(
        video_frames.FrameSample(
            units.RetrievalUnit(
                1000 + index,
                1,
                "video",
                units.Locator(start_seconds=index / 20, end_seconds=(index + 1) / 20),
                metadata={"tenant": "a"},
            ),
            tmp_path / f"{index}.png",
            index,
            hashlib.sha256(str(index).encode()).hexdigest(),
        )
        for index in range(0, count, 4)
    )
    return video_frames.VideoFrames(tmp_path, "a" * 64, "explicit-fixture", 20.0, count, frames)


def test_proposals_cover_fixed_grid_and_tail_preserving_distinct_original_support(tmp_path):
    source = sampled(tmp_path)
    result = video_windows.build(source, first_unit_id=2000)
    assert result.duration_seconds == 8.5
    assert [w.unit.locator.start_seconds for w in result.windows] == [0, 2, 4, 4.5]
    for window in result.windows:
        assert len(window.frames) == len({f.frame_index for f in window.frames}) == 16
        assert [f.frame_index for f in window.frames] == sorted(f.frame_index for f in window.frames)
        assert all(f in source.samples for f in window.frames)
        assert window.unit.asset_id == 1 and window.unit.metadata == {"tenant": "a"}
        assert not window.unit.evidence and window.middle_frame is window.frames[8]
    assert video_windows.build(source, first_unit_id=2000).version == result.version
    assert video_windows.build(source, first_unit_id=3000).version != result.version


def test_proposals_fail_before_unbounded_allocation_and_reject_time_or_id_drift(tmp_path):
    source = sampled(tmp_path)
    for kwargs in ({"stride_seconds": 5e-324}, {"max_windows": 1}, {"first_unit_id": 1000}):
        values = {"first_unit_id": 2000, **kwargs}
        with pytest.raises(ValueError):
            video_windows.build(source, **values)
    bad = dataclasses.replace(source.samples[1], frame_index=0)
    with pytest.raises(ValueError, match="timeline"):
        video_windows.build(dataclasses.replace(source, samples=(source.samples[0], bad)), first_unit_id=2000)
    with pytest.raises(ValueError, match="16 distinct"):
        video_windows.build(dataclasses.replace(source, samples=source.samples[::4]), first_unit_id=2000)


def test_actual_cfr_sampling_byte_budget_is_enforced_before_png_write(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    path = tmp_path / "actual.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (64, 48))
    assert writer.isOpened()
    try:
        for index in range(170):
            writer.write(np.full((48, 64, 3), index % 200, dtype=np.uint8))
    finally:
        writer.release()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="output byte budget"):
        video_frames.sample(
            path, tmp_path / "rejected", asset_id=1, first_unit_id=1000, interval_seconds=0.2, max_output_bytes=1
        )
    assert not (tmp_path / "rejected/frame-00000.png").exists()
    assert json.loads((tmp_path / "rejected/incomplete.json").read_text())["complete"] is False
    original = video_frames.sample(path, tmp_path / "original", asset_id=1, first_unit_id=1000, interval_seconds=0.2)
    bounded = video_frames.sample(
        path, tmp_path / "bounded", asset_id=1, first_unit_id=1000, interval_seconds=0.2, max_output_bytes=1024**2
    )
    assert [f.sha256 for f in original.samples] == [f.sha256 for f in bounded.samples]
    assert "max_output_bytes" not in json.loads((original.directory / "manifest.json").read_text())["config"]
    assert original.version != bounded.version
    assert len(video_windows.build(bounded, first_unit_id=2000).windows) == 4
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_video_quality_imports_and_help_need_no_optional_runtimes():
    code = """
import sys
from backend.app.evaluation import video_benchmark
from backend.app.evaluation.hybrid_eval import video_source, video_window_io, video_quality, video_metrics
from backend.app.retrieval.hybrid import temporal, video_windows
assert not any(n.startswith(('numpy','cv2','PIL','torch','transformers','onnxruntime')) for n in sys.modules)
sys.argv=['video_benchmark','--help']
video_benchmark.main()
"""
    subprocess.run([sys.executable, "-S", "-c", code], check=True, capture_output=True, timeout=10)


def test_low_rate_real_video_nearest_spacing_retains_sixteen_distinct_frames(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    source = tmp_path / "low.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 6, (64, 48))
    assert writer.isOpened()
    try:
        for index in range(60):
            writer.write(np.full((48, 64, 3), index * 3, dtype=np.uint8))
    finally:
        writer.release()
    default = video_frames.sample(source, tmp_path / "default", asset_id=1, first_unit_id=1000, interval_seconds=0.2)
    with pytest.raises(ValueError, match="16 distinct"):
        video_windows.build(default, first_unit_id=2000)
    nearest = video_frames.sample(
        source, tmp_path / "nearest", asset_id=1, first_unit_id=1000, interval_seconds=0.2, interval_rounding="nearest"
    )
    assert len(nearest.samples) == 60 and len(default.samples) == 30
    assert all(
        len({f.frame_index for f in w.frames}) == 16 for w in video_windows.build(nearest, first_unit_id=2000).windows
    )
    assert "interval_rounding" not in json.loads((default.directory / "manifest.json").read_text())["config"]
