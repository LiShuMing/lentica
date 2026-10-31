"""Final SQL record hydration for already selected resources and evidence IDs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db import models
from backend.app.retrieval.hybrid import stages


@dataclass(frozen=True)
class AssetRecord:
    """Presentation data read only for one final file and its selected chunks."""

    file: models.File
    image: models.Image | None
    chunks: dict[int, models.Chunk]


class SqlAssetStore:
    """One final batch, using file/image and optional exact-chunk SQL queries.

    This reads local database records, not original media bytes. Candidate feature
    and index construction reads happen separately before this adapter is called.
    """

    def __init__(self, db: Session) -> None:
        self.db = db
        self.calls = 0
        self.file_ids: tuple[int, ...] = ()
        self.chunk_ids: tuple[int, ...] = ()

    def fetch_evidence(self, hits: Sequence[stages.AssetHit]) -> Mapping[int, AssetRecord]:
        """Resolve only selected parents/support; fail on missing or disabled data."""
        self.calls += 1
        ids = tuple(hit.asset_id for hit in hits)
        if not ids or len(ids) > 100 or len(set(ids)) != len(ids):
            raise ValueError("Final product fetch requires 1..100 unique files")
        expected_chunks = {}
        image_ids = set()
        for hit in hits:
            for support in hit.evidence:
                unit = support.unit
                if unit.asset_id != hit.asset_id:
                    raise ValueError("Final supporting unit has a different parent")
                if unit.kind == "text":
                    chunk_id = (unit.unit_id - 1) // 2
                    if unit.unit_id % 2 != 1 or chunk_id <= 0:
                        raise ValueError("Final document unit namespace differs")
                    expected_chunks[chunk_id] = hit.asset_id
                elif unit.kind == "image" and unit.unit_id == hit.asset_id * 2:
                    image_ids.add(hit.asset_id)
                else:
                    raise ValueError("Unsupported final product unit identity")
        self.file_ids, self.chunk_ids = ids, tuple(expected_chunks)
        statement = (
            select(models.File, models.Image)
            .outerjoin(models.Image, models.Image.file_id == models.File.id)
            .join(models.Source, models.Source.id == models.File.source_id)
            .where(models.File.id.in_(ids), models.File.ingestion_status == "indexed", models.Source.enabled.is_(True))
            .execution_options(populate_existing=True)
        )
        records = {file.id: AssetRecord(file, image, {}) for file, image in self.db.execute(statement)}
        if set(records) != set(ids) or any(records[identifier].image is None for identifier in image_ids):
            raise ValueError("Final file/image is missing, disabled or no longer indexed")
        chunks = (
            self.db.scalars(
                select(models.Chunk)
                .where(models.Chunk.id.in_(expected_chunks))
                .execution_options(populate_existing=True)
            ).all()
            if expected_chunks
            else []
        )
        if {chunk.id for chunk in chunks} != set(expected_chunks):
            raise ValueError("Final supporting chunk is missing")
        for chunk in chunks:
            if chunk.file_id != expected_chunks[chunk.id]:
                raise ValueError("Final supporting chunk moved to a different file")
            records[chunk.file_id].chunks[chunk.id] = chunk
        return records
