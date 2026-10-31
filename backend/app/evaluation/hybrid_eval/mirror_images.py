"""Bounded, source-preserving preparation of a selected public CIFAR image mirror."""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import io
import json
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from backend.app.evaluation.hybrid_eval import reporting
from backend.app.retrieval.hybrid import snapshots

CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


@dataclass(frozen=True)
class ImageItem:
    """Original source/pixel group and label, separate from anonymous model input."""

    identifier: int
    source_id: str
    partition: str
    label: str
    filename: str
    pixel_sha256: str
    file_sha256: str


@dataclass(frozen=True)
class ImageDataset:
    """Explicit prepared public images; no directory/library discovery on access."""

    directory: Path
    items: tuple[ImageItem, ...]
    identity: str
    provenance: Mapping[str, object]

    def partition(self, name: str) -> tuple[ImageItem, ...]:
        """Select one frozen source partition, rejecting misspelled names."""
        if name not in {"dev", "validation", "test"}:
            raise ValueError("Unknown public image partition")
        return tuple(item for item in self.items if item.partition == name)


def _members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    selected = {}
    total = 0
    for index, member in enumerate(archive):
        path = PurePosixPath(member.name)
        if index >= 65000 or path.is_absolute() or ".." in path.parts or not (member.isdir() or member.isfile()):
            raise ValueError("Public archive member count/path/type violates its budget")
        if member.isdir():
            continue
        total += member.size
        if (
            member.size > 64000
            or total > 256 * 1024**2
            or len(path.parts) != 4
            or path.parts[0] != "cifar10"
            or path.parts[1] not in {"train", "test"}
            or path.parts[2] not in CLASSES
            or path.suffix != ".png"
            or member.name in selected
        ):
            raise ValueError("Unexpected public image archive member/layout")
        selected[member.name] = member
    return selected


def _selection(
    members: Mapping[str, tarfile.TarInfo], count: int, excluded: frozenset[str] = frozenset()
) -> dict[str, str]:
    chosen = {}
    for label in CLASSES:
        for split in ("train", "test"):
            names = [
                name for name in members if name not in excluded and PurePosixPath(name).parts[1:3] == (split, label)
            ]
            names.sort(key=lambda name: hashlib.sha256(("42:" + name).encode()).hexdigest())
            needed = count * 2 if split == "train" else count
            if len(names) < needed:
                raise ValueError("Public image source lacks the declared class/partition count")
            for index, name in enumerate(names[:needed]):
                chosen[name] = "test" if split == "test" else "dev" if index < count else "validation"
    return chosen


def prepare(
    archive_path: Path, directory: Path, *, per_class: int = 20, exclude: Sequence[ImageDataset] = ()
) -> ImageDataset:
    """Validate gzip/member bounds and write only selected image files, never paths.

    The expected acquisition manifest must accompany the explicitly chosen
    archive. Integrity is recorded for this mirror, not claimed equivalent to an
    independently checked official binary archive. No tar extractall or pickle.
    """
    from PIL import Image

    if type(per_class) is not int or not 1 <= per_class <= 100 or Path(directory).exists():
        raise ValueError("Choose a fresh image directory and a bounded per-class count")
    archive_path, directory = Path(archive_path), Path(directory)
    acquisition = json.loads((archive_path.parent / "acquisition.json").read_bytes())
    if len(exclude) > 10 or any(previous.provenance.get("sha256") != acquisition["sha256"] for previous in exclude):
        raise ValueError("Prior image exclusions must refer to the same explicit archive, at most ten datasets")
    excluded_sources = frozenset(item.source_id for previous in exclude for item in previous.items)
    excluded_pixels = frozenset(item.pixel_sha256 for previous in exclude for item in previous.items)
    if (
        archive_path.stat().st_size > 160 * 1024**2
        or archive_path.stat().st_size != acquisition["bytes"]
        or snapshots.sha256_file(archive_path) != acquisition["sha256"]
    ):
        raise ValueError("Public image archive acquisition digest/size mismatch")
    # Consume the gzip trailer, rather than mistaking an early tar end marker
    # for validation of the whole compressed archive's CRC and completeness.
    expanded = 0
    with gzip.open(archive_path, "rb") as stream:
        while block := stream.read(1024 * 1024):
            expanded += len(block)
            if expanded > 512 * 1024**2:
                raise ValueError("Public archive exceeds its expanded byte budget")
    records = []
    with tarfile.open(archive_path, "r:gz") as archive:
        members = _members(archive)
        chosen = _selection(members, per_class, excluded_sources)
        # Ascending archive offsets avoid repeatedly rewinding a gzip stream.
        for name in sorted(chosen, key=lambda name: members[name].offset_data):
            with archive.extractfile(members[name]) as stream:
                encoded = stream.read(64001)
            with Image.open(io.BytesIO(encoded)) as image:
                if image.size != (32, 32) or image.format != "PNG":
                    raise ValueError("Public source must contain native 32x32 PNG images")
                pixels = image.convert("RGB").tobytes()
            records.append((name, chosen[name], encoded, hashlib.sha256(pixels).hexdigest()))
    directory.mkdir(parents=True, exist_ok=False)
    items, excluded = [], []
    seen: dict[str, str] = {}
    order = {"dev": 0, "validation": 1, "test": 2}
    for name, partition, encoded, pixel_hash in sorted(records, key=lambda row: (order[row[1]], row[0])):
        label = PurePosixPath(name).parts[2]
        if pixel_hash in excluded_pixels:
            excluded.append(name)
            continue
        if pixel_hash in seen:
            if seen[pixel_hash] != label:
                raise ValueError("Identical source pixels have conflicting public class labels")
            excluded.append(name)
            continue
        seen[pixel_hash] = label
        identifier = len(items)
        filename = f"{identifier:05d}.png"
        (directory / filename).write_bytes(encoded)
        items.append(
            ImageItem(identifier, name, partition, label, filename, pixel_hash, hashlib.sha256(encoded).hexdigest())
        )
    manifest = {
        "schema": "lentica-cifar-mirror-images-v1",
        "source": acquisition,
        "per_class_target": per_class,
        "excluded_exact_duplicates": excluded,
        "excluded_prior_dataset_ids": sorted(previous.identity for previous in exclude),
        "items": [dataclasses.asdict(item) for item in items],
        "limits": "Exact decoded-pixel duplicate exclusion; near duplicates and pretraining overlap unknown.",
    }
    manifest["sha256"] = reporting.digest(manifest)
    (directory / "dataset.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return load(directory)


def load(directory: Path) -> ImageDataset:
    """Reopen checksum-bound selected images with no archive traversal or extraction."""
    directory = Path(directory)
    path = directory / "dataset.json"
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("Prepared image manifest exceeds 4 MiB")
    manifest = json.loads(path.read_bytes())
    fingerprint = manifest.pop("sha256")
    if (
        manifest["schema"] != "lentica-cifar-mirror-images-v1"
        or reporting.digest(manifest) != fingerprint
        or not 1 <= len(manifest["items"]) <= 3000
    ):
        raise ValueError("Prepared image source identity/size mismatch")
    items = tuple(ImageItem(**value) for value in manifest["items"])
    if len({item.identifier for item in items}) != len(items) or len({item.pixel_sha256 for item in items}) != len(
        items
    ):
        raise ValueError("Prepared image IDs/pixel source groups must be unique")
    for item in items:
        image = directory / item.filename
        if (
            item.label not in CLASSES
            or item.partition not in {"dev", "validation", "test"}
            or Path(item.filename).name != item.filename
            or image.is_symlink()
            or snapshots.sha256_file(image) != item.file_sha256
        ):
            raise ValueError("Prepared public image path/label/partition/digest mismatch")
    return ImageDataset(directory, items, fingerprint, manifest["source"])
