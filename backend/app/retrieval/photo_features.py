"""SQL projections for photo scoring, separate from final presentation records."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import extract, select
from sqlalchemy.orm import Session

from backend.app.db import models
from backend.app.retrieval import query_plan


@dataclass(frozen=True)
class FileFeature:
    """Only identity and filename input needed by metadata ranking."""

    id: int
    canonical_path: str


@dataclass(frozen=True)
class ImageFeature:
    """Stored text/date features needed by scoring and inherited filters."""

    caption: str | None
    context_json: str | None
    exif_time: datetime | None
    tags: str | None
    ocr_text: str | None
    location: str | None


def _columns() -> tuple:
    return (
        models.File.id,
        models.File.canonical_path,
        models.Image.caption,
        models.Image.context_json,
        models.Image.exif_time,
        models.Image.tags,
        models.Image.ocr_text,
        models.Image.location,
    )


def _statement(columns: tuple, plan: query_plan.RetrievalPlan):
    statement = (
        select(*columns)
        .select_from(models.File)
        .join(models.Image, models.Image.file_id == models.File.id)
        .join(models.Source, models.Source.id == models.File.source_id)
        .where(models.File.ingestion_status == "indexed", models.Source.enabled.is_(True))
    )
    if plan.years:
        statement = statement.where(extract("year", models.Image.exif_time).in_([int(y) for y in plan.years]))
    if plan.person_names:
        statement = statement.where(models.File.id.in_(plan.person_file_ids))
    return statement


def vector_rows(
    db: Session, plan: query_plan.RetrievalPlan, namespaces: Sequence[str]
) -> Iterator[tuple[models.PhotoVector, FileFeature, ImageFeature]]:
    """Stream vector records and explicit feature columns, without File/Image ORM loads."""
    statement = (
        _statement((models.PhotoVector, *_columns()), plan)
        .join(models.PhotoVector, models.PhotoVector.file_id == models.File.id)
        .where(models.PhotoVector.namespace.in_(namespaces))
        .order_by(models.PhotoVector.id)
    )
    for row in db.execute(statement.execution_options(yield_per=256)):
        yield row[0], FileFeature(*row[1:3]), ImageFeature(*row[3:])


def lexical_rows(db: Session, plan: query_plan.RetrievalPlan) -> Iterator[tuple[FileFeature, ImageFeature]]:
    """Stream OCR/filter features even when no compatible vector exists."""
    statement = _statement(_columns(), plan).where(models.Image.ocr_text.is_not(None)).order_by(models.File.id)
    for row in db.execute(statement.execution_options(yield_per=256)):
        yield FileFeature(*row[:2]), ImageFeature(*row[2:])
