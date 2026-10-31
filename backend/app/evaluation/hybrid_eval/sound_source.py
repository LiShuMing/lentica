"""Source-checked preparation of the predeclared five-class ESC50 sound slice."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import wave
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

from backend.app.evaluation.hybrid_eval import reporting
from backend.app.retrieval.hybrid import snapshots

CLASSES = ("dog", "rain", "siren", "clock_alarm", "door_wood_knock")


@dataclass(frozen=True)
class SoundItem:
    """Class/source metadata stays outside anonymous model waveform inputs."""

    identifier: int
    source_id: str
    original_filename: str
    partition: str
    label: str
    filename: str
    original_sha256: str
    sha256: str


@dataclass(frozen=True)
class SoundDataset:
    """Explicit prepared clips and their fixed native-source folds."""

    directory: Path
    items: tuple[SoundItem, ...]
    identity: str
    provenance: Mapping[str, object]

    def partition(self, name: str) -> tuple[SoundItem, ...]:
        """Return only the named frozen fold group."""
        if name not in {"dev", "validation", "test"}:
            raise ValueError("Unknown sound evaluation partition")
        return tuple(item for item in self.items if item.partition == name)


def _selected(source: Path):
    acquisition = source / "download-manifest-recovered.json"
    if acquisition.stat().st_size > 2 * 1024**2 or (source / "esc50.csv").stat().st_size > 1024**2:
        raise ValueError("Sound acquisition metadata exceeds its budget")
    report = json.loads(acquisition.read_bytes())
    raw = (source / "esc50.csv").read_bytes()
    if (
        report["complete"] is not True
        or len(report["files"]) != 200
        or hashlib.sha256(raw).hexdigest() != report["metadata_sha256"]
    ):
        raise ValueError("Sound source acquisition is incomplete or differs from the CSV")
    selected = {row["filename"]: row for row in csv.DictReader(io.StringIO(raw.decode())) if row["category"] in CLASSES}
    if len(selected) != 200 or {row["filename"] for row in report["files"]} != set(selected):
        raise ValueError("Sound acquisition differs from the predeclared class slice")
    sources = {}
    ordered = sorted(report["files"], key=lambda row: row["filename"])
    for row in ordered:
        original = selected[row["filename"]]
        if any(row[key] != value for key, value in original.items()):
            raise ValueError("Sound labels/source/folds differ from original metadata")
        fold = int(row["fold"])
        if fold not in {1, 2, 3, 4, 5}:
            raise ValueError("Invalid ESC50 fold")
        partition = "dev" if fold <= 2 else "validation" if fold == 3 else "test"
        if row["src_file"] in sources and sources[row["src_file"]] != partition:
            raise ValueError("Original recording crosses sound partitions")
        sources[row["src_file"]] = partition
        path = Path(row["file"])
        if path.is_absolute() or ".." in path.parts or len(path.parts) != 2:
            raise ValueError("Acquired waveform must stay in the explicitly selected source directory")
        yield row, partition, source / path


def _convert(path: Path, destination: Path, expected: Mapping[str, object]) -> int:
    import numpy as np
    from scipy import signal

    if (
        path.stat().st_size != expected["bytes"]
        or path.stat().st_size > 1024**2
        or snapshots.sha256_file(path) != expected["sha256"]
    ):
        raise ValueError("Original sound file size/hash differs from acquisition")
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getnframes()) != (
            1,
            2,
            44100,
            220500,
        ):
            raise ValueError("Expected original5s mono PCM16/44100 ESC50 waveform")
        data = source.readframes(220500)
        if len(data) != 441000:
            raise ValueError("Original sound PCM payload is incomplete")
    samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    resampled = signal.resample_poly(samples, 160, 147)
    if len(resampled) != 240000 or not np.isfinite(resampled).all():
        raise ValueError("Invalid resampled ESC50 waveform")
    clipped = int(((resampled < -1) | (resampled > 32767 / 32768.0)).sum())
    encoded = np.round(np.clip(resampled, -1, 32767 / 32768.0) * 32768).astype("<i2").tobytes()
    with wave.open(str(destination), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(48000)
        target.writeframes(encoded)
    return clipped


def prepare(source: Path, directory: Path) -> SoundDataset:
    """Verify the exact selected200 originals and explicitly resample into new files.

    Source files/labels are not modified. The manifest binds raw/derived bytes,
    original source groups, folds and conversion versions. Inference receives
    anonymous filenames; no filename/category text is a model audio input.
    """
    source, directory = Path(source), Path(directory)
    if directory.exists():
        raise ValueError("Choose a fresh sound preparation directory")
    rows = list(_selected(source))
    directory.mkdir(parents=True, exist_ok=False)
    items = []
    clipped = 0
    for row, partition, path in rows:
        name = f"clip-{len(items):05d}.wav"
        clipped += _convert(path, directory / name, row)
        items.append(
            SoundItem(
                len(items),
                row["src_file"],
                row["filename"],
                partition,
                row["category"],
                name,
                row["sha256"],
                snapshots.sha256_file(directory / name),
            )
        )
    provenance = {
        "metadata_sha256": snapshots.sha256_file(source / "esc50.csv"),
        "classes": list(CLASSES),
        "folds": {"dev": [1, 2], "validation": [3], "test": [4, 5]},
        "conversion": "scipy.resample_poly160/147, explicit round/clip PCM16/48k mono",
        "scipy": metadata.version("scipy"),
        "clipped_samples": clipped,
        "scope": "Public five-class5s source-isolated ESC50 subset; local integrity hashes, unknown model pretraining overlap.",
    }
    manifest = {
        "schema": "lentica-sound-dataset-v1",
        "provenance": provenance,
        "items": [asdict(item) for item in items],
    }
    manifest["identity"] = reporting.digest(manifest)
    with (directory / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return load(directory)


def load(directory: Path) -> SoundDataset:
    """Verify the explicit completed manifest/derived files without model imports."""
    directory = Path(directory)
    path = directory / "manifest.json"
    if path.stat().st_size > 2 * 1024**2:
        raise ValueError("Prepared sound manifest exceeds2MiB")
    report = json.loads(path.read_bytes())
    identity = report.pop("identity")
    if (
        reporting.digest(report) != identity
        or report["schema"] != "lentica-sound-dataset-v1"
        or len(report["items"]) != 200
    ):
        raise ValueError("Prepared sound identity/schema/count mismatch")
    items = tuple(SoundItem(**row) for row in report["items"])
    origins = {}
    if [item.identifier for item in items] != list(range(200)):
        raise ValueError("Sound IDs must be stable and contiguous")
    for item in items:
        if (
            item.filename != f"clip-{item.identifier:05d}.wav"
            or item.partition not in {"dev", "validation", "test"}
            or item.label not in CLASSES
        ):
            raise ValueError("Unexpected prepared sound identity/partition/file")
        if item.source_id in origins and origins[item.source_id] != item.partition:
            raise ValueError("Sound source crosses prepared folds")
        origins[item.source_id] = item.partition
        if (directory / item.filename).stat().st_size > 2 * 1024**2 or snapshots.sha256_file(
            directory / item.filename
        ) != item.sha256:
            raise ValueError("Prepared sound waveform identity mismatch")
    for split, count in (("dev", 16), ("validation", 8), ("test", 16)):
        if any(sum(item.partition == split and item.label == label for item in items) != count for label in CLASSES):
            raise ValueError("Prepared sound class/fold counts differ from declared protocol")
    return SoundDataset(directory, items, identity, report["provenance"])
