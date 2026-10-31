"""Frozen public multi-action references and anonymous read-only video copies."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from . import reporting

LABELS = MappingProxyType(
    {
        "c106": "drinking",
        "c121": "washing dishes",
        "c139": "washing hands",
        "c144": "fixing hair",
        "c149": "laughing",
        "c153": "sneezing",
    }
)
MAX_VIDEO_BYTES = 16 * 1024**2


@dataclass(frozen=True)
class VideoEvent:
    """Independent raw human timing, including intervals beyond observed media."""

    label: str
    start: float
    end: float


@dataclass(frozen=True)
class SourceCase:
    """Source/actor grouping and references; never model/index feature text."""

    source_id: str
    subject: str
    partition: str
    annotated_seconds: float
    events: tuple[VideoEvent, ...]
    bytes: int
    crc32: int


@dataclass(frozen=True)
class VideoSelection:
    """An annotation-only configuration frozen before any video acquisition."""

    path: Path
    identity: str
    labels: Mapping[str, str]
    cases: tuple[SourceCase, ...]


@dataclass(frozen=True)
class VideoCase:
    """One anonymous original video with independent reference identity."""

    reference: SourceCase
    asset_id: int
    filename: str
    sha256: str


@dataclass(frozen=True)
class VideoDataset:
    """An explicit verified source snapshot; no discovery or network operations."""

    directory: Path
    identity: str
    selection: VideoSelection
    cases: tuple[VideoCase, ...]


def _read(path: Path) -> reporting.JsonObject:
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("Video metadata exceeds4MiB")
    return json.loads(path.read_text(encoding="utf-8"))


def _event(row: reporting.JsonObject) -> VideoEvent:
    label, start, end = row["label"], row["start"], row["end"]
    if (
        not isinstance(label, str)
        or not re.fullmatch(r"c\d{3}", label)
        or int(label[1:]) > 156
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in (start, end))
        or not 0 <= start < end <= 120
    ):
        raise ValueError("Video reference needs finite explicit raw event timing")
    return VideoEvent(label, start, end)


def load_selection(path: Path) -> VideoSelection:
    """Verify the fixed16-subject/six-class protocol without optional imports.

    This checks the frozen record's consistency, not an upstream signature or
    proof that the independently acquired annotations are correct.
    """
    path = Path(path).resolve()
    frozen = _read(path)
    identity = frozen.pop("identity")
    if (
        frozen["schema"] != "lentica-video-source-selection-v1"
        # Acquisition's pre-pixel frozen selection uses compact JSON. Preserve
        # that identity instead of rewriting it to this repo's report format.
        or hashlib.sha256(
            json.dumps(frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        != identity
        or frozen["labels"] != LABELS
        or len(frozen["cases"]) != 16
        or frozen["window_seconds"] != 4
        or frozen["stride_seconds"] != 2
        or frozen["frames_per_window"] != 16
        or frozen["siglip_prompts"] != {key: "a person " + value for key, value in LABELS.items()}
    ):
        raise ValueError("Video frozen source/configuration identity differs")
    cases = []
    for index, row in enumerate(frozen["cases"]):
        events = tuple(_event(item) for item in row["events"])
        part = "dev" if index < 6 else "validation" if index < 11 else "test"
        entry = row["archive_entry"]
        if (
            not re.fullmatch(r"[A-Z0-9]{5}", row["source_id"])
            or not re.fullmatch(r"[A-Z0-9]{4}", row["subject"])
            or row["partition"] != part
            or type(row["annotated_seconds"]) not in (int, float)
            or not 8 <= row["annotated_seconds"] <= 40
            or not 1 <= len(events) <= 200
            or not any(
                a.label != b.label and a.label in LABELS and b.label in LABELS and a.end + 0.5 <= b.start
                for a in events
                for b in events
            )
            or entry["name"] != "Charades_v1_480/" + row["source_id"] + ".mp4"
            or entry["flags"] != 0
            or entry["method"] != 8
            or any(
                type(entry[key]) is not int
                for key in ("offset", "compressed_bytes", "bytes", "crc32", "flags", "method")
            )
            or not 0 < entry["compressed_bytes"] <= entry["bytes"] <= MAX_VIDEO_BYTES
            or not 0 <= entry["crc32"] <= 0xFFFFFFFF
            or not 0 <= entry["offset"] < frozen["archive_bytes"]
        ):
            raise ValueError("Video reference/group/partition/archive budget differs")
        cases.append(
            SourceCase(
                row["source_id"], row["subject"], part, row["annotated_seconds"], events, entry["bytes"], entry["crc32"]
            )
        )
    ids = [case.source_id for case in cases]
    if len(set(ids)) != 16 or ids != sorted(ids) or len({case.subject for case in cases}) != 16:
        raise ValueError("Video source/subject groups overlap or source order differs")
    return VideoSelection(path, identity, MappingProxyType(dict(LABELS)), tuple(cases))


def _video(path: Path, reference: SourceCase) -> str:
    if path.is_symlink() or path.stat().st_size != reference.bytes or reference.bytes > MAX_VIDEO_BYTES:
        raise ValueError("Video source file bytes/path budget differs")
    payload = path.read_bytes()
    if payload[4:8] != b"ftyp" or zlib.crc32(payload) != reference.crc32:
        raise ValueError("Video original container/CRC differs from frozen archive entry")
    return hashlib.sha256(payload).hexdigest()


def prepare(selection: Path, sources: Path, target: Path) -> VideoDataset:
    """Copy only frozen original MP4s into a fresh anonymous snapshot.

    Raw references and source IDs remain in evaluation metadata. The model gets
    anonymous explicit paths/pixels only, with no scripts/descriptions/captions.
    Missing/duplicate/mutated videos fail without replacing a selected source.
    """
    selected = load_selection(selection)
    sources, target = Path(sources).resolve(), Path(target)
    target.mkdir()
    shutil.copyfile(selected.path, target / "selection.json")
    rows, seen = [], set()
    for index, reference in enumerate(selected.cases):
        source = sources / (reference.source_id + ".mp4")
        digest = _video(source, reference)
        if digest in seen:
            raise ValueError("Duplicate frozen video bytes; do not replace selected sources")
        seen.add(digest)
        name = f"video-{index:03d}.mp4"
        shutil.copyfile(source, target / name)
        if _video(target / name, reference) != digest:
            raise ValueError("Video changed during copying")
        rows.append({"asset_id": index + 1, "source_id": reference.source_id, "filename": name, "sha256": digest})
    manifest = {"schema": "lentica-video-dataset-v1", "selection_identity": selected.identity, "cases": rows}
    manifest["identity"] = reporting.digest(manifest)
    with (target / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return load(target)


def load(directory: Path) -> VideoDataset:
    """Verify all exact selected source bytes and immutable anonymous identities."""
    directory = Path(directory).resolve()
    selected = load_selection(directory / "selection.json")
    manifest = _read(directory / "manifest.json")
    identity = manifest.pop("identity")
    if (
        manifest["schema"] != "lentica-video-dataset-v1"
        or reporting.digest(manifest) != identity
        or manifest["selection_identity"] != selected.identity
        or len(manifest["cases"]) != len(selected.cases)
    ):
        raise ValueError("Video dataset manifest/selection identity differs")
    cases, seen = [], set()
    for index, (row, reference) in enumerate(zip(manifest["cases"], selected.cases)):
        if (
            type(row["asset_id"]) is not int
            or row["asset_id"] != index + 1
            or row["source_id"] != reference.source_id
            or row["filename"] != f"video-{index:03d}.mp4"
        ):
            raise ValueError("Video anonymous path/source identity differs")
        digest = _video(directory / row["filename"], reference)
        if digest != row["sha256"] or digest in seen:
            raise ValueError("Video byte identity differs or duplicate videos overlap")
        seen.add(digest)
        cases.append(VideoCase(reference, row["asset_id"], row["filename"], digest))
    return VideoDataset(directory, identity, selected, tuple(cases))
