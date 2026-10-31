"""Time conjunction/filter budgets and actual subtitle/video mechanism controls."""

from __future__ import annotations

import hashlib
import json

import pytest

from backend.app.retrieval.hybrid import contracts, media, subtitles, temporal, units, video_frames


def _unit(identifier, start, end, asset=1, tenant="a"):
    return units.RetrievalUnit(
        identifier, asset, "video", units.Locator(start_seconds=start, end_seconds=end), metadata={"tenant": tenant}
    )


def test_common_time_and_same_resource_are_required_preserving_original_evidence():
    first, second, third = _unit(1, 0, 2), _unit(2, 1, 3), _unit(3, 2, 4)
    assert temporal.intersect({"a": [first], "b": [second], "c": [third]}, contracts.SearchFilter()) == ()
    assert temporal.intersect({"a": [first], "b": [_unit(9, 0, 1, asset=2)]}, contracts.SearchFilter()) == ()
    result = temporal.intersect({"b": [second], "a": [first]}, contracts.SearchFilter(equals={"tenant": "a"}))
    assert len(result) == 1 and result[0].locator == units.Locator(start_seconds=1, end_seconds=2)
    assert result[0].evidence == (first, second)
    assert result == temporal.intersect({"a": [first], "b": [second]}, contracts.SearchFilter())


def test_temporal_filter_identity_and_excess_do_not_silently_degrade():
    values = {"a": [_unit(1, 0, 2)], "b": [_unit(2, 0, 1), _unit(3, 1, 2)]}
    for kwargs in ({"max_matches": 1}, {"max_combinations": 1}):
        with pytest.raises(ValueError, match="budget"):
            temporal.intersect(values, contracts.SearchFilter(), **kwargs)
    with pytest.raises(ValueError, match="filters"):
        temporal.intersect(values, contracts.SearchFilter(allowed_ids=frozenset({1, 2})))
    with pytest.raises(ValueError, match="identity"):
        temporal.intersect({"a": [_unit(1, 0, 1)], "b": [_unit(1, 0, 1, asset=2)]}, contracts.SearchFilter())
    assert temporal.intersect({"a": [], "b": [_unit(2, 0, 1)]}, contracts.SearchFilter()) == ()


def test_subtitles_are_explicit_source_bound_cues_and_reject_invalid_times(tmp_path):
    text = "1\n00:00:00,000 --> 00:00:01,000\n<b>alarm</b> &amp; voice\n\n2\n00:00:02,000 --> 00:00:03,000\nother cue\n"
    path = tmp_path / "captions.srt"
    path.write_text(text)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    values = subtitles.extract_units(path, format="srt", asset_id=9, first_unit_id=100, metadata_values={"tenant": "a"})
    assert [unit.unit_id for unit in values] == [100, 101] and values[0].text() == "alarm & voice"
    assert all(
        unit.evidence[0].producer == "provided-subtitle-srt" and unit.metadata["tenant"] == "a" for unit in values
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    vtt = "WEBVTT\n\nNOTE comment\nignored\n\ncue\n00:01.000 --> 00:02.500 align:start\n<v Alice>Hello</v>\n"
    assert subtitles.parse(vtt, format="vtt")[0].text == "Hello"
    for invalid in ("WEBVTTbad\n\n00:01.000 --> 00:02.000\nx", "WEBVTT\nX-TIMESTAMP-MAP=LOCAL:00:00.000,MPEGTS:0", ""):
        with pytest.raises(ValueError):
            subtitles.parse(invalid, format="vtt")
    for invalid in ("1\n00:60:00,000 --> 01:00:01,000\nx", "1\n00:00:02,000 --> 00:00:01,000\nx"):
        with pytest.raises(ValueError):
            subtitles.parse(invalid, format="srt")
    with pytest.raises(ValueError, match="budget"):
        subtitles.parse(text, format="srt", max_cues=1)


def test_actual_cfr_frame_decode_and_time_filter_prevent_cross_time_false_match(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    path = tmp_path / "control.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4.0, (64, 48))
    assert writer.isOpened()
    try:
        for bgr in (
            (0, 0, 255),
            (0, 0, 255),
            (255, 0, 0),
            (255, 0, 0),
            (0, 255, 0),
            (0, 255, 0),
            (0, 0, 255),
            (0, 0, 255),
        ):
            writer.write(np.full((48, 64, 3), bgr, dtype=np.uint8))
    finally:
        writer.release()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = video_frames.sample(
        path, tmp_path / "frames", asset_id=1, first_unit_id=10, interval_seconds=0.5, metadata_values={"tenant": "a"}
    )
    assert result.decoded_frames == 8 and len(result.samples) == 4
    assert [frame.frame_index for frame in result.samples] == [0, 2, 4, 6]
    assert [frame.unit.locator.start_seconds for frame in result.samples] == [0.0, 0.5, 1.0, 1.5]
    manifest = json.loads((result.directory / "manifest.json").read_text())
    assert manifest["complete"] and all(
        frame.path.exists() and hashlib.sha256(frame.path.read_bytes()).hexdigest() == frame.sha256
        for frame in result.samples
    )
    catalog = units.UnitCatalog([frame.unit for frame in result.samples])
    color = media.ColorHistogramRetriever(catalog, {frame.unit.unit_id: frame.path for frame in result.samples})
    red_ids = color.retrieve(contracts.SearchQuery("red", contracts.SearchFilter(equals={"tenant": "a"})), 10)
    assert red_ids == (10, 13)
    cue = _unit(100, 1.0, 1.25)  # Caption at green frame, although red exists elsewhere.
    assert (
        temporal.intersect(
            {"visual": [catalog.units[identifier] for identifier in red_ids], "subtitle": [cue]},
            contracts.SearchFilter(),
        )
        == ()
    )
    assert (
        len(
            temporal.intersect(
                {"visual": [catalog.units[10]], "subtitle": [_unit(101, 0.0, 0.5)]}, contracts.SearchFilter()
            )
        )
        == 1
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    with pytest.raises(ValueError, match="sample budget"):
        video_frames.sample(
            path, tmp_path / "too-many", asset_id=1, first_unit_id=10, interval_seconds=0.2, max_samples=2
        )
    assert not (tmp_path / "too-many").exists()
