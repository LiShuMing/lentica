"""Source-grouped public relation cases, explicit pixels and immutable identities."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from backend.app.evaluation.hybrid_eval import reporting

KINDS = ("replace_rel", "swap_obj")
PARTITIONS = ("dev", "validation", "test")
MAX_FILE_BYTES = 4 * 1024**2


@dataclass(frozen=True)
class SourceCase:
    """An annotation-only choice, frozen without checking image availability."""

    case_id: str
    kind: str
    source_image: str
    partition: str
    positive: str
    negative: str
    positive_index: int


@dataclass(frozen=True)
class RelationCase(SourceCase):
    """One original image plus balanced competing captions; labels score only."""

    filename: str
    image_sha256: str
    pixel_sha256: str

    @property
    def captions(self) -> tuple[str, str]:
        """Counterbalance which input position contains the correct caption."""
        return (self.positive, self.negative) if self.positive_index == 0 else (self.negative, self.positive)


@dataclass(frozen=True)
class RelationDataset:
    """A verified explicit source snapshot; no image/model discovery occurs."""

    directory: Path
    identity: str
    cases: tuple[RelationCase, ...]


def _read(path: Path) -> reporting.JsonObject:
    if Path(path).stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Relation JSON exceeds4MiB")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Relation image exceeds4MiB")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _caption(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError("Relation captions must be explicit bounded nonempty text")
    return value


def select_cases(
    annotations: Mapping[str, Path], *, per_kind: int = 60, excluded_sources: frozenset[str] = frozenset()
) -> tuple[SourceCase, ...]:
    """Freeze earliest numeric case IDs, grouping all partitions by source image.

    Selection uses annotations only, before any image availability or prediction
    is observed. A source used by an earlier kind is excluded from later kinds.
    """
    if set(annotations) != set(KINDS) or type(per_kind) is not int or not 3 <= per_kind <= 300 or per_kind % 3:
        raise ValueError("Relation selection requires both kinds and a bounded multiple of three")
    if (
        not isinstance(excluded_sources, frozenset)
        or len(excluded_sources) > 600
        or any(not isinstance(name, str) or not re.fullmatch(r"\d{12}\.jpg", name) for name in excluded_sources)
    ):
        raise ValueError("Relation source exclusions require a bounded explicit source-ID set")
    seen, chosen = set(excluded_sources), []
    for kind in KINDS:
        raw = _read(annotations[kind])
        if not isinstance(raw, dict) or len(raw) > 10000 or any(not key.isdecimal() for key in raw):
            raise ValueError("Relation annotations require bounded numeric case IDs")
        selected = 0
        for key in sorted(raw, key=int):
            item = raw[key]
            filename = item["filename"]
            if not isinstance(filename, str) or not re.fullmatch(r"\d{12}\.jpg", filename):
                raise ValueError("Relation source image must be an explicit COCO filename")
            if filename in seen:
                continue
            positive, negative = _caption(item["caption"]), _caption(item["negative_caption"])
            if positive == negative:
                raise ValueError("Competing relation captions must differ")
            seen.add(filename)
            chosen.append(
                SourceCase(
                    f"{kind}:{key}",
                    kind,
                    filename,
                    PARTITIONS[selected // (per_kind // 3)],
                    positive,
                    negative,
                    selected % 2,
                )
            )
            selected += 1
            if selected == per_kind:
                break
        if selected != per_kind:
            raise ValueError("Insufficient distinct source-image groups; never replace cases by availability")
    return tuple(chosen)


def _source_record(case: SourceCase) -> reporting.JsonObject:
    return {field.name: getattr(case, field.name) for field in dataclasses.fields(SourceCase)}


def prepare(
    annotations: Mapping[str, Path],
    images: Path,
    target: Path,
    *,
    per_kind: int = 60,
    exclude: RelationDataset | None = None,
) -> RelationDataset:
    """Copy only fixed original pixels into a fresh anonymous dataset directory.

    Args:
        annotations: Explicit local official JSON files for both relation kinds.
        images: Explicit source directory; only frozen filenames are read.
        target: Fresh output directory; no overwrite, downloads or source changes.
        per_kind: Fixed cases per kind, split equally across three partitions.
        exclude: Explicit prior snapshot, reverified before selecting any new IDs.
            Prior sources are excluded before availability; duplicate prior pixels
            fail without choosing replacement cases. The exclusion binds identity.

    Returns:
        A source/annotation/byte/pixel-bound snapshot. Identical pixel groups
        anywhere in this selected snapshot fail rather than being replaced.

    Raises:
        ValueError: Invalid annotations, duplicate pixels, unsafe paths or budgets.
        FileExistsError: The target already exists.
    """
    exclusion = None
    if exclude is not None:
        prior = load(exclude.directory)
        if prior.identity != exclude.identity or prior.cases != exclude.cases:
            raise ValueError("Prior relation snapshot differs from explicit exclusion")
        inherited_sources, inherited_pixels = _exclusions(
            _read(prior.directory / "manifest.json").get("excluded_dataset")
        )
        exclusion = {
            "identity": prior.identity,
            "source_images": sorted(inherited_sources | {case.source_image for case in prior.cases}),
            "pixel_sha256": sorted(inherited_pixels | {case.pixel_sha256 for case in prior.cases}),
        }
    excluded_sources, excluded_pixels = _exclusions(exclusion)
    selected = select_cases(annotations, per_kind=per_kind, excluded_sources=frozenset(excluded_sources))
    target, images = Path(target), Path(images).resolve()
    target.mkdir()
    annotation_hashes = {kind: _hash(Path(path)) for kind, path in annotations.items()}
    selection = {"annotation_sha256": annotation_hashes, "cases": [_source_record(case) for case in selected]}
    if exclusion is not None:
        selection["excluded_dataset"] = exclusion
    (target / "source-selection.json").write_text(json.dumps(selection), encoding="utf-8")
    from PIL import Image

    cases, pixels = [], set(excluded_pixels)
    for index, case in enumerate(selected):
        source = images / case.source_image
        if source.is_symlink() or not source.resolve().is_relative_to(images):
            raise ValueError("Relation source path escapes its explicit directory")
        image_sha = _hash(source)
        with Image.open(source) as image:
            if image.format != "JPEG" or image.width * image.height > 16_000_000:
                raise ValueError("Relation source requires bounded original JPEG pixels")
            with image.convert("RGB") as rgb:
                pixel_sha = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        if pixel_sha in pixels:
            raise ValueError("Duplicate selected pixel group; do not replace or leak between partitions")
        pixels.add(pixel_sha)
        filename = f"image-{index:06d}.jpg"
        shutil.copyfile(source, target / filename)
        if _hash(target / filename) != image_sha:
            raise ValueError("Relation source changed during copying")
        cases.append(
            RelationCase(**dataclasses.asdict(case), filename=filename, image_sha256=image_sha, pixel_sha256=pixel_sha)
        )
    manifest = {
        "schema": "lentica-relation-dataset-v1",
        "annotation_sha256": annotation_hashes,
        "per_kind": per_kind,
        "cases": [dataclasses.asdict(case) for case in cases],
    }
    if exclusion is not None:
        manifest["excluded_dataset"] = exclusion
    manifest["identity"] = reporting.digest(manifest)
    with (target / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return load(target)


def _exclusions(exclusion: reporting.JsonObject | None) -> tuple[set[str], set[str]]:
    """Check a self-contained bounded previous-source/pixel identity record."""
    if exclusion is None:
        return set(), set()
    if set(exclusion) != {"identity", "source_images", "pixel_sha256"} or not re.fullmatch(
        r"[0-9a-f]{64}", exclusion["identity"]
    ):
        raise ValueError("Relation exclusion identity differs")
    sources, pixels = exclusion["source_images"], exclusion["pixel_sha256"]
    if (
        not 1 <= len(sources) <= 600
        or len(sources) != len(pixels)
        or len(set(sources)) != len(sources)
        or len(set(pixels)) != len(pixels)
        or any(not re.fullmatch(r"\d{12}\.jpg", name) for name in sources)
        or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in pixels)
    ):
        raise ValueError("Relation source/pixel exclusion groups differ")
    return set(sources), set(pixels)


def load(directory: Path) -> RelationDataset:
    """Verify exact selected files/source groups without optional numeric imports."""
    directory = Path(directory).resolve()
    manifest = _read(directory / "manifest.json")
    frozen = dict(manifest)
    identity = frozen.pop("identity")
    per_kind = frozen["per_kind"]
    if (
        frozen["schema"] != "lentica-relation-dataset-v1"
        or reporting.digest(frozen) != identity
        or type(per_kind) is not int
        or not 3 <= per_kind <= 300
        or per_kind % 3
        or len(frozen["cases"]) != per_kind * 2
    ):
        raise ValueError("Relation snapshot schema/identity/case budget differs")
    cases = tuple(RelationCase(**item) for item in frozen["cases"])
    exclusion = frozen.get("excluded_dataset")
    excluded_sources, excluded_pixels = _exclusions(exclusion)
    selection = {
        "annotation_sha256": frozen["annotation_sha256"],
        "cases": [_source_record(case) for case in cases],
    }
    if exclusion is not None:
        selection["excluded_dataset"] = exclusion
    if _read(directory / "source-selection.json") != selection:
        raise ValueError("Relation snapshot differs from frozen annotation-only selection")
    for case in cases:
        if (
            case.kind not in KINDS
            or case.partition not in PARTITIONS
            or not re.fullmatch(re.escape(case.kind) + r":\d+", case.case_id)
            or not re.fullmatch(r"\d{12}\.jpg", case.source_image)
            or not re.fullmatch(r"image-\d{6}\.jpg", case.filename)
            or type(case.positive_index) is not int
            or case.positive_index not in (0, 1)
            or not re.fullmatch(r"[0-9a-f]{64}", case.pixel_sha256)
            or not re.fullmatch(r"[0-9a-f]{64}", case.image_sha256)
            or _caption(case.positive) == _caption(case.negative)
            or case.source_image in excluded_sources
            or case.pixel_sha256 in excluded_pixels
            or (directory / case.filename).is_symlink()
            or _hash(directory / case.filename) != case.image_sha256
        ):
            raise ValueError("Relation case/path/byte/caption identity differs")
    if any(
        len({getattr(case, field) for case in cases}) != len(cases)
        for field in ("case_id", "source_image", "filename", "pixel_sha256")
    ) or any(
        sum(case.kind == kind and case.partition == part for case in cases) != per_kind // 3
        for kind in KINDS
        for part in PARTITIONS
    ):
        raise ValueError("Relation source/pixel groups or partition counts overlap/differ")
    for kind in KINDS:
        for index, case in enumerate(case for case in cases if case.kind == kind):
            if case.positive_index != index % 2 or case.partition != PARTITIONS[index // (per_kind // 3)]:
                raise ValueError("Relation counterbalancing or frozen partition order differs")
    return RelationDataset(directory, identity, cases)
