"""Immutable SQLite feature snapshots with verified atomic reference changes.

These snapshots persist units, final JSON records and dense vectors. They are
not an ANN/full-text engine. Index adapters may construct their own versioned
artifacts from a pinned snapshot without reopening original private sources.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import contracts, units

_SNAPSHOT_ID = re.compile(r"[0-9a-f]{32}")
_SCHEMA = "lentica-unit-snapshot-v1"


def canonical(value: object) -> str:
    """Encode dynamic JSON boundaries deterministically and reject NaN values."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    """Hash an explicitly selected file in bounded chunks, without directory scans."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SnapshotIdentity:
    """Explicit source, encoder and configuration revisions for a single build."""

    source_version: str
    model_version: str
    config_version: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value for value in vars(self).values()):
            raise ValueError("All snapshot versions must be explicit nonempty strings")


def _unit_payload(unit: units.RetrievalUnit) -> str:
    return canonical(
        {
            "unit_id": unit.unit_id,
            "asset_id": unit.asset_id,
            "kind": unit.kind,
            "locator": dataclasses.asdict(unit.locator),
            "evidence": [dataclasses.asdict(item) for item in unit.evidence],
            "metadata": dict(unit.metadata),
        }
    )


def _decode_unit(payload: str) -> units.RetrievalUnit:
    value = json.loads(payload)
    return units.RetrievalUnit(
        value["unit_id"],
        value["asset_id"],
        value["kind"],
        units.Locator(**value["locator"]),
        tuple(units.TextEvidence(**item) for item in value["evidence"]),
        value["metadata"],
    )


class Snapshot:
    """Pin a verified, read-only feature/record snapshot independent of CURRENT.

    No persistent connection or writable database state is retained. Callers may
    keep an older reader while a new snapshot is activated. Repository-owned
    snapshot files must remain immutable after construction; verification detects
    accidental corruption on open, not hostile replacement after verification.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory).resolve()
        manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != _SCHEMA or not _SNAPSHOT_ID.fullmatch(manifest.get("snapshot_id", "")):
            raise ValueError("Invalid snapshot schema or identity")
        self.snapshot_id = manifest["snapshot_id"]
        if self.directory.name != self.snapshot_id:
            raise ValueError("Snapshot directory and manifest identities differ")
        self.path = self.directory / "index.sqlite"
        if self.path.is_symlink() or sha256_file(self.path) != manifest["sha256"]:
            raise ValueError("Snapshot database digest mismatch")
        self.identity = SnapshotIdentity(**manifest["identity"])
        self.sha256 = manifest["sha256"]
        self.parent_id = manifest.get("parent_id")
        with self._connect() as connection:
            info = dict(connection.execute("SELECT key,value FROM info"))
            if info != {
                "schema": _SCHEMA,
                "snapshot_id": self.snapshot_id,
                "identity": canonical(manifest["identity"]),
            }:
                raise ValueError("Database and manifest versions differ")
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("Snapshot database integrity failure")

    @contextlib.contextmanager
    def _connect(self):
        with contextlib.closing(sqlite3.connect(self.path.as_uri() + "?mode=ro&immutable=1", uri=True)) as connection:
            yield connection

    def all_units(self) -> tuple[units.RetrievalUnit, ...]:
        """Read this selected snapshot's units in stable ID order."""
        with self._connect() as connection:
            return tuple(_decode_unit(row[0]) for row in connection.execute("SELECT payload FROM units ORDER BY id"))

    def _fetch(self, table: str, ids: Sequence[int]) -> dict[int, Any]:
        requested = tuple(dict.fromkeys(contracts.asset_id(identifier) for identifier in ids))
        found = {}
        with self._connect() as connection:
            # Bound placeholders for SQLite builds with the traditional 999 limit.
            for start in range(0, len(requested), 900):
                batch = requested[start : start + 900]
                rows = connection.execute(
                    f"SELECT id,payload FROM {table} WHERE id IN ({','.join('?' for _ in batch)})", batch
                )
                found.update(rows)
        if set(found) != set(requested):
            raise KeyError("Snapshot omitted a requested identity")
        return {identifier: found[identifier] for identifier in requested}

    def fetch_units(self, ids: Sequence[int]) -> Mapping[int, units.RetrievalUnit]:
        """Load only candidate features; this is separate from final asset reads."""
        return {identifier: _decode_unit(value) for identifier, value in self._fetch("units", ids).items()}

    def fetch(self, ids: Sequence[int]) -> Mapping[int, object]:
        """Batch-resolve final JSON resource records from the pinned snapshot."""
        return {identifier: json.loads(value) for identifier, value in self._fetch("assets", ids).items()}

    def vectors(self) -> dict[str, dict[int, tuple[float, ...]]]:
        """Read persisted dense spaces without importing numeric/model libraries."""
        result: dict[str, dict[int, tuple[float, ...]]] = {}
        with self._connect() as connection:
            for space, identifier, dimensions, blob in connection.execute(
                "SELECT space,id,dimensions,data FROM vectors ORDER BY space,id"
            ):
                result.setdefault(space, {})[identifier] = struct.unpack(f"<{dimensions}f", blob)
        return result


def build(
    root: Path,
    catalog: Sequence[units.RetrievalUnit],
    records: Mapping[int, object],
    identity: SnapshotIdentity,
    *,
    vectors: Mapping[str, Mapping[int, Sequence[float]]] | None = None,
    parent_id: str | None = None,
) -> Snapshot:
    """Build a new snapshot without changing CURRENT or any original sources.

    Records must cover every unit's parent. Vector rows require known unit IDs,
    finite float32 coordinates and one nonzero dimensionality per named space.
    A failed build may leave its new unreferenced directory for diagnosis; it
    never changes a reader's snapshot or removes preexisting files.
    """
    indexed = units.UnitCatalog(catalog)
    for identifier in records:
        contracts.asset_id(identifier)
    if not {unit.asset_id for unit in catalog} <= set(records):
        raise ValueError("Final records must cover every unit's parent asset")
    if parent_id is not None and not _SNAPSHOT_ID.fullmatch(parent_id):
        raise ValueError("Invalid parent snapshot identity")
    identifier = uuid.uuid4().hex
    directory = Path(root) / "snapshots" / identifier
    directory.mkdir(parents=True, exist_ok=False)
    with contextlib.closing(sqlite3.connect(directory / "index.sqlite")) as connection:
        connection.executescript("""
            PRAGMA foreign_keys=ON;
            CREATE TABLE info(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE assets(id INTEGER PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE units(id INTEGER PRIMARY KEY, parent INTEGER REFERENCES assets(id), payload TEXT NOT NULL);
            CREATE TABLE vectors(space TEXT, id INTEGER REFERENCES units(id), dimensions INTEGER, data BLOB,
                                 PRIMARY KEY(space,id));
        """)
        with connection:
            connection.executemany(
                "INSERT INTO info VALUES(?,?)",
                {
                    "schema": _SCHEMA,
                    "snapshot_id": identifier,
                    "identity": canonical(dataclasses.asdict(identity)),
                }.items(),
            )
            connection.executemany(
                "INSERT INTO assets VALUES(?,?)", ((key, canonical(value)) for key, value in records.items())
            )
            connection.executemany(
                "INSERT INTO units VALUES(?,?,?)",
                ((unit.unit_id, unit.asset_id, _unit_payload(unit)) for unit in indexed.units.values()),
            )
            for space, rows in (vectors or {}).items():
                if not isinstance(space, str) or not space or len({len(row) for row in rows.values()}) > 1:
                    raise ValueError("Vector spaces require names and consistent dimensions")
                for key, row in rows.items():
                    if key not in indexed.units or not row or any(not math.isfinite(value) for value in row):
                        raise ValueError("Vector rows require known units and finite nonempty coordinates")
                    blob = struct.pack(f"<{len(row)}f", *row)
                    if any(not math.isfinite(value) for value in struct.unpack(f"<{len(row)}f", blob)):
                        raise ValueError("Vector coordinates exceed float32 range")
                    connection.execute("INSERT INTO vectors VALUES(?,?,?,?)", (space, key, len(row), blob))
    manifest = {
        "schema": _SCHEMA,
        "snapshot_id": identifier,
        "identity": dataclasses.asdict(identity),
        "parent_id": parent_id,
        "sha256": sha256_file(directory / "index.sqlite"),
    }
    (directory / "manifest.json").write_text(canonical(manifest) + "\n", encoding="utf-8")
    return Snapshot(directory)


def derive(
    previous: Snapshot,
    root: Path,
    identity: SnapshotIdentity,
    *,
    upserts: Sequence[units.RetrievalUnit] = (),
    delete_units: frozenset[int] = frozenset(),
    record_upserts: Mapping[int, object] | None = None,
    delete_assets: frozenset[int] = frozenset(),
    vector_upserts: Mapping[str, Mapping[int, Sequence[float]]] | None = None,
) -> Snapshot:
    """Copy-on-write update; changed unit/model revisions invalidate old vectors.

    Deleting an asset removes its units. Unit upserts cannot reference deleted
    assets, and simultaneous delete/upsert of one unit is rejected. Unknown
    deletion IDs fail instead of disguising a stale caller. CURRENT is untouched.
    """
    existing = {unit.unit_id: unit for unit in previous.all_units()}
    with previous._connect() as connection:
        records = {
            identifier: json.loads(value) for identifier, value in connection.execute("SELECT id,payload FROM assets")
        }
    changed = units.UnitCatalog(upserts).units
    if not delete_units <= set(existing) or not delete_assets <= set(records):
        raise ValueError("Deletion must target existing identities")
    if set(changed) & delete_units or set(record_upserts or {}) & delete_assets:
        raise ValueError("Cannot delete and upsert the same identity")
    for key in delete_units:
        del existing[key]
    existing = {key: unit for key, unit in existing.items() if unit.asset_id not in delete_assets}
    if any(unit.asset_id in delete_assets for unit in upserts):
        raise ValueError("Unit upsert references a deleted asset")
    existing.update(changed)
    records = {key: value for key, value in records.items() if key not in delete_assets}
    records.update(record_upserts or {})
    vectors = previous.vectors() if previous.identity.model_version == identity.model_version else {}
    vectors = {
        space: {key: row for key, row in rows.items() if key in existing and key not in changed}
        for space, rows in vectors.items()
    }
    for space, rows in (vector_upserts or {}).items():
        vectors.setdefault(space, {}).update(rows)
    return build(root, tuple(existing.values()), records, identity, vectors=vectors, parent_id=previous.snapshot_id)


def activate(root: Path, snapshot_id: str) -> Snapshot:
    """Verify then atomically publish or roll back a reference; old files remain."""
    if not _SNAPSHOT_ID.fullmatch(snapshot_id):
        raise ValueError("Invalid snapshot identity")
    root = Path(root)
    snapshot = Snapshot(root / "snapshots" / snapshot_id)
    temporary = root / f".CURRENT-{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(canonical({"snapshot_id": snapshot_id, "sha256": snapshot.sha256}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / "CURRENT.json")
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return snapshot


def current(root: Path) -> Snapshot:
    """Resolve CURRENT once; later activations cannot change this pinned reader."""
    value = json.loads((Path(root) / "CURRENT.json").read_text(encoding="utf-8"))
    identifier = value["snapshot_id"]
    if not isinstance(identifier, str) or not _SNAPSHOT_ID.fullmatch(identifier):
        raise ValueError("Invalid CURRENT snapshot identity")
    snapshot = Snapshot(Path(root) / "snapshots" / identifier)
    if snapshot.sha256 != value["sha256"]:
        raise ValueError("CURRENT and snapshot digests differ")
    return snapshot


def cache_key(snapshot: Snapshot, query: contracts.SearchQuery, config: contracts.SearchConfig) -> str:
    """Include snapshot/source/model/config/filter/request identities in one key."""
    if not isinstance(query, contracts.SearchQuery):
        raise ValueError("Use the multimodal cache key for image/combined requests")
    value = {
        "snapshot_id": snapshot.snapshot_id,
        "sha256": snapshot.sha256,
        "identity": dataclasses.asdict(snapshot.identity),
        "text": query.text,
        "equals": dict(query.filters.equals),
        "allowed_ids": None if query.filters.allowed_ids is None else sorted(query.filters.allowed_ids),
        "config": dataclasses.asdict(config),
    }
    return hashlib.sha256(canonical(value).encode()).hexdigest()
