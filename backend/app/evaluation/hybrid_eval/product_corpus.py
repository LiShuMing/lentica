"""Isolated SQLite fixtures for public-corpus and synthetic product benchmarks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sqlalchemy import create_engine, event, insert
from sqlalchemy.engine import Engine

from backend.app.evaluation.hybrid_eval import dataset


def create_database(path: Path) -> Engine:
    """Create a fresh benchmark database, refusing existing state."""
    from backend.app.db import models

    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path}", connect_args={"timeout": 30, "check_same_thread": False})

    @event.listens_for(engine, "connect")
    def configure(connection, _record):
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")

    models.Base.metadata.create_all(engine)
    return engine


def seed_documents(engine: Engine, source: dataset.Dataset, directory: Path) -> None:
    """Map public IDs to positive File IDs without placing judgments in features.

    The fixture contains one full abstract per chunk. Opaque filenames and markers
    do not contain query text, source relevance labels or semantic categories.
    No original source file is read by this SQL materialization.
    """
    from backend.app.db import models

    with engine.begin() as connection:
        connection.execute(
            insert(models.Source),
            [{"id": 1, "name": "public corpus", "root_path": str(directory), "kind": "documents"}],
        )
        connection.execute(
            insert(models.File),
            [
                {
                    "id": row.asset_id + 1,
                    "source_id": 1,
                    "canonical_path": str(directory / f"item-{row.asset_id:06d}.txt"),
                    "windows_path": f"item-{row.asset_id:06d}.txt",
                    "file_type": "txt",
                    "ingestion_status": "indexed",
                }
                for row in source.documents
            ],
        )
        connection.execute(
            insert(models.Document),
            [
                {"file_id": row.asset_id + 1, "title": row.title, "parser": "public-abstract"}
                for row in source.documents
            ],
        )
        connection.execute(
            insert(models.Chunk),
            [
                {
                    "file_id": row.asset_id + 1,
                    "text": row.text,
                    "page_or_segment": "abstract",
                    "chunk_hash": hashlib.sha256(row.text.encode()).hexdigest(),
                }
                for row in source.documents
            ],
        )


def seed_vectors(engine: Engine, count: int, dimensions: int) -> str:
    """Store deterministic synthetic vectors for measuring real JSON scan costs.

    These normalized random features have no semantic meaning. The same prefix
    and seed are retained across scales, with one image and one text vector/file.
    """
    import numpy as np

    from backend.app.db import models

    identity = f"benchmark:synthetic-normalized-v1:{dimensions}"
    rng = np.random.default_rng(42)
    with engine.begin() as connection:
        connection.execute(
            insert(models.Source), [{"id": 1, "name": "synthetic scale", "root_path": "/benchmark", "kind": "images"}]
        )
        for start in range(0, count, 100):
            ids = list(range(start + 1, min(count, start + 100) + 1))
            connection.execute(
                insert(models.File),
                [
                    {
                        "id": identifier,
                        "source_id": 1,
                        "canonical_path": f"/benchmark/item-{identifier:06d}.jpg",
                        "windows_path": f"item-{identifier:06d}.jpg",
                        "file_type": "jpg",
                        "ingestion_status": "indexed",
                    }
                    for identifier in ids
                ],
            )
            connection.execute(
                insert(models.Image),
                [{"file_id": identifier, "tags": "[]", "context_json": "{}"} for identifier in ids],
            )
            vectors = rng.normal(size=(len(ids), 2, dimensions)).astype(np.float32)
            vectors /= np.linalg.norm(vectors, axis=2, keepdims=True)
            connection.execute(
                insert(models.PhotoVector),
                [
                    {
                        "file_id": identifier,
                        "namespace": namespace,
                        "dims": dimensions,
                        "encoder_id": identity,
                        "vector_json": json.dumps(vectors[index, offset].tolist()),
                    }
                    for index, identifier in enumerate(ids)
                    for offset, namespace in enumerate(("image", "text"))
                ],
            )
    return identity
