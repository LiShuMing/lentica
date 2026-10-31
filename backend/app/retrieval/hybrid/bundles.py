"""Self-contained immutable feature/text/vector versions with one atomic pointer."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from . import snapshots

if TYPE_CHECKING:
    from numpy import typing as npt

    from . import ann, fulltext

_ID = re.compile(r"[0-9a-f]{32}")
_SCHEMA = "lentica-index-bundle-v1"


def _copy_files(source: Path, destination: Path, names: list[str]) -> dict[str, str]:
    """Copy a bounded explicit index inventory; reject nested paths and symlinks."""
    if not names or len(names) > 512:
        raise ValueError("Index bundle requires between 1 and 512 files per component")
    paths = [source / name for name in names]
    if any(Path(name).name != name or path.is_symlink() or not path.is_file() for name, path in zip(names, paths)):
        raise ValueError("Bundle components require regular direct files")
    if sum(path.stat().st_size for path in paths) > 8 * 1024**3:
        raise ValueError("Selected bundle component exceeds 8 GiB")
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in paths:
        shutil.copyfile(path, destination / path.name)
        hashes[path.name] = snapshots.sha256_file(destination / path.name)
    return hashes


class IndexBundle:
    """Pin a fully verified local version; native runtimes load only on demand.

    Files are trusted application-produced artifacts. Hashes detect corruption,
    not a malicious producer or mutation after verification. Readers must never
    write native indexes or remove old bundles while they remain referenced.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory).resolve()
        path = self.directory / "bundle.json"
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("Bundle manifest exceeds 1 MiB")
        manifest = json.loads(path.read_bytes())
        self.bundle_id = manifest.get("bundle_id", "")
        if (
            manifest.get("schema") != _SCHEMA
            or not _ID.fullmatch(self.bundle_id)
            or self.directory.name != self.bundle_id
        ):
            raise ValueError("Invalid bundle schema or directory identity")
        self.sha256 = snapshots.sha256_file(path)
        files = manifest["files"]
        if not 1 <= len(files) <= 1536:
            raise ValueError("Invalid bundle inventory size")
        for name, digest in files.items():
            candidate = self.directory / name
            if (
                candidate.is_symlink()
                or self.directory not in candidate.resolve().parents
                or snapshots.sha256_file(candidate) != digest
            ):
                raise ValueError("Bundle component path or digest mismatch")
        snapshot_id = manifest["snapshot_id"]
        if not _ID.fullmatch(snapshot_id):
            raise ValueError("Invalid bundled snapshot identity")
        self.snapshot = snapshots.Snapshot(self.directory / "snapshot" / snapshot_id)
        if dataclasses.asdict(self.snapshot.identity) != manifest["identity"]:
            raise ValueError("Bundle and feature identity differ")
        self.has_text, self.has_vector = manifest["text"], manifest["vector"]
        self._validate_components()

    def _validate_components(self) -> None:
        metadata = {str(unit.unit_id): dict(unit.metadata) for unit in self.snapshot.all_units()}
        for enabled, name, manifest_name in (
            (self.has_text, "text", "lentica.json"),
            (self.has_vector, "vector", "manifest.json"),
        ):
            if not enabled:
                continue
            value = json.loads((self.directory / name / manifest_name).read_bytes())
            if value["source_version"] != self.snapshot.identity.source_version or value["metadata"] != metadata:
                raise ValueError("Bundle index source/unit/filter domain differs from snapshot")
            if name == "vector" and value["encoder_id"] != self.snapshot.identity.model_version:
                raise ValueError("Bundle vector encoder differs from snapshot model")

    def open_text(self) -> fulltext.TantivyRetriever:
        """Reopen the pinned full-text version, without reading CURRENT again."""
        from . import fulltext

        if not self.has_text:
            raise ValueError("Bundle has no text index")
        return fulltext.TantivyRetriever(self.directory / "text")

    def open_vector(self, encode_query: Callable[[str], npt.ArrayLike], *, encoder_id: str) -> ann.HnswRetriever:
        """Reopen the pinned HNSW version, verifying the query encoder identity."""
        from . import ann

        if not self.has_vector:
            raise ValueError("Bundle has no vector index")
        result = ann.HnswRetriever(self.directory / "vector", encode_query, query_encoder_id=encoder_id)
        if set(result.ids) != {unit.unit_id for unit in self.snapshot.all_units()}:
            raise ValueError("Bundle vector rows differ from feature unit domain")
        return result


def build(
    root: Path,
    snapshot: snapshots.Snapshot,
    *,
    text_directory: Path | None = None,
    vector_directory: Path | None = None,
) -> IndexBundle:
    """Copy explicit frozen artifacts to a new version; publish nothing on failure.

    Index source versions, complete unit/filter domains and vector model must
    match the feature snapshot. Input directories must be quiescent, built by
    this application. This operation neither reindexes sources nor updates any
    existing reference. A failed unreferenced build is retained for diagnosis.
    """
    identifier = uuid.uuid4().hex
    directory = Path(root) / "bundles" / identifier
    directory.mkdir(parents=True, exist_ok=False)
    inventories = {}
    components = [
        (snapshot.directory, directory / "snapshot" / snapshot.snapshot_id, ["manifest.json", "index.sqlite"])
    ]
    if text_directory is not None:
        text_directory = Path(text_directory)
        # Only this explicitly selected native index directory is enumerated.
        names = []
        for path in text_directory.iterdir():
            if path.name.endswith(".lock"):
                continue
            names.append(path.name)
            if len(names) > 512:
                raise ValueError("Text index file inventory exceeds 512")
        components.append((text_directory, directory / "text", sorted(names)))
    if vector_directory is not None:
        components.append(
            (Path(vector_directory), directory / "vector", ["manifest.json", "index.faiss", "ids.npy", "vectors.npy"])
        )
    for source, destination, names in components:
        inventories.update(
            {
                str((destination / name).relative_to(directory)): digest
                for name, digest in _copy_files(source, destination, names).items()
            }
        )
    manifest = {
        "schema": _SCHEMA,
        "bundle_id": identifier,
        "snapshot_id": snapshot.snapshot_id,
        "identity": dataclasses.asdict(snapshot.identity),
        "text": text_directory is not None,
        "vector": vector_directory is not None,
        "files": inventories,
    }
    (directory / "bundle.json").write_text(snapshots.canonical(manifest) + "\n", encoding="utf-8")
    return IndexBundle(directory)


def activate(root: Path, bundle_id: str) -> IndexBundle:
    """Verify the entire version before atomically switching or rolling back."""
    if not isinstance(bundle_id, str) or not _ID.fullmatch(bundle_id):
        raise ValueError("Invalid bundle identity")
    root = Path(root)
    bundle = IndexBundle(root / "bundles" / bundle_id)
    temporary = root / f".BUNDLE_CURRENT-{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(snapshots.canonical({"bundle_id": bundle_id, "sha256": bundle.sha256}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / "BUNDLE_CURRENT.json")
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return bundle


def current(root: Path) -> IndexBundle:
    """Resolve one pointer and pin the whole version, never individual indexes."""
    value = json.loads((Path(root) / "BUNDLE_CURRENT.json").read_bytes())
    identifier = value["bundle_id"]
    if not isinstance(identifier, str) or not _ID.fullmatch(identifier):
        raise ValueError("Invalid current bundle identity")
    bundle = IndexBundle(Path(root) / "bundles" / identifier)
    if bundle.sha256 != value["sha256"]:
        raise ValueError("Current bundle digest mismatch")
    return bundle


def cache_namespace(bundle: IndexBundle) -> str:
    """Include the complete native index version alongside existing request keys."""
    return hashlib.sha256(f"{bundle.bundle_id}:{bundle.sha256}".encode()).hexdigest()
