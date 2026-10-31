"""Frozen source grouping/bytes boundaries; synthetic containers are not video quality."""

import copy
import hashlib
import json
import zlib

import pytest

from backend.app.evaluation.hybrid_eval import video_source


def save_selection(path, data):
    body = {key: value for key, value in data.items() if key != "identity"}
    body["identity"] = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(body))


@pytest.fixture
def source_inputs(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    cases = []
    for index in range(16):
        name = f"{index:05d}"
        payload = b"\0\0\0\x18ftyp" + name.encode() + b"explicit-fixture-not-real-video"
        (raw / (name + ".mp4")).write_bytes(payload)
        cases.append(
            {
                "source_id": name,
                "subject": f"{index:04X}",
                "partition": "dev" if index < 6 else "validation" if index < 11 else "test",
                "annotated_seconds": 8,
                "events": [{"label": "c106", "start": 0, "end": 1}, {"label": "c153", "start": 2, "end": 3}],
                "archive_entry": {
                    "name": "Charades_v1_480/" + name + ".mp4",
                    "offset": index * 100,
                    "compressed_bytes": len(payload),
                    "bytes": len(payload),
                    "method": 8,
                    "crc32": zlib.crc32(payload),
                    "flags": 0,
                },
            }
        )
    data = {
        "schema": "lentica-video-source-selection-v1",
        "labels": dict(video_source.LABELS),
        "cases": cases,
        "window_seconds": 4,
        "stride_seconds": 2,
        "frames_per_window": 16,
        "siglip_prompts": {k: "a person " + v for k, v in video_source.LABELS.items()},
        "archive_bytes": 10000,
    }
    selection = tmp_path / "frozen.json"
    save_selection(selection, data)
    return selection, raw, data


def test_video_source_anonymous_copy_reload_and_raw_mutation_fail(source_inputs, tmp_path):
    selection, raw, _ = source_inputs
    result = video_source.prepare(selection, raw, tmp_path / "prepared")
    assert len(result.cases) == 16 and result.cases[0].filename == "video-000.mp4"
    assert result.cases[0].reference.source_id == "00000"
    assert video_source.load(result.directory).identity == result.identity
    (result.directory / result.cases[0].filename).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="bytes/path budget"):
        video_source.load(result.directory)
    with pytest.raises(FileExistsError):
        video_source.prepare(selection, raw, result.directory)


def test_missing_frozen_video_does_not_select_replacement(source_inputs, tmp_path):
    selection, raw, _ = source_inputs
    (raw / "00000.mp4").unlink()
    with pytest.raises(FileNotFoundError):
        video_source.prepare(selection, raw, tmp_path / "missing")
    assert (tmp_path / "missing/selection.json").exists()
    assert not (tmp_path / "missing/manifest.json").exists()


def test_source_identity_grouping_paths_and_raw_refs_are_checked(source_inputs):
    selection, _, data = source_inputs
    identity = video_source.load_selection(selection).identity
    for mutation in ("subject", "path", "reference", "prompt"):
        changed = copy.deepcopy(data)
        if mutation == "subject":
            changed["cases"][1]["subject"] = changed["cases"][0]["subject"]
        if mutation == "path":
            changed["cases"][0]["source_id"] = "../private"
        if mutation == "reference":
            changed["cases"][0]["events"][0]["end"] = float("nan")
        if mutation == "prompt":
            changed["siglip_prompts"]["c106"] = "chosen from model output"
        save_selection(selection, changed)
        with pytest.raises(ValueError):
            video_source.load_selection(selection)
    save_selection(selection, data)
    assert video_source.load_selection(selection).identity == identity
