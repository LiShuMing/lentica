from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.core.paths import to_windows_path
from backend.app.db.models import Chunk, Document, File, IngestionJob, Source
from backend.app.ingestion.parsers import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    chunk_hash,
    parse_document,
    sha256_file,
)


def ensure_default_sources(db: Session) -> None:
    settings = get_settings()
    defaults = [
        ("Documents", str(settings.documents_source), "documents"),
        ("Photos", str(settings.photos_source), "images"),
    ]
    for name, root_path, kind in defaults:
        exists = db.scalar(select(Source).where(Source.root_path == root_path))
        if not exists:
            db.add(Source(name=name, root_path=root_path, kind=kind, enabled=True))
    db.commit()


def scan_documents(
    db: Session,
    source_id: int | None = None,
    max_files: int | None = 100,
    max_bytes: int = 20 * 1024 * 1024,
) -> IngestionJob:
    ensure_default_sources(db)
    job = IngestionJob(status="running")
    db.add(job)
    db.commit()
    db.refresh(job)

    query = select(Source).where(Source.enabled.is_(True), Source.kind == "documents")
    if source_id is not None:
        query = query.where(Source.id == source_id)
    sources = list(db.scalars(query))

    seen_paths: set[str] = set()
    processed = 0
    try:
        for source in sources:
            root = Path(source.root_path)
            if not root.exists():
                continue
            for path in _iter_supported_files(root, max_bytes=max_bytes):
                if max_files is not None and processed >= max_files:
                    break
                seen_paths.add(str(path))
                job.total_files += 1
                _index_one_file(db, source, path, job)
                processed += 1
                db.commit()
            source.last_scan_at = datetime.utcnow()

        if max_files is None:
            for source in sources:
                existing = db.scalars(select(File).where(File.source_id == source.id)).all()
                for file in existing:
                    if file.canonical_path not in seen_paths and Path(file.canonical_path).suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS:
                        file.ingestion_status = "missing"
                        file.updated_at = datetime.utcnow()

        job.status = "succeeded"
    except Exception as exc:  # noqa: BLE001
        job.status = "failed"
        job.error = str(exc)
    finally:
        job.finished_at = datetime.utcnow()
        db.commit()
        db.refresh(job)
    return job


def _iter_supported_files(root: Path, max_bytes: int):
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
            continue
        if path.name.startswith(".") or path.name.startswith("._"):
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
        except OSError:
            continue
        candidates.append(path)
    yield from sorted(candidates, key=lambda item: (item.stat().st_size, str(item).lower()))


def _index_one_file(db: Session, source: Source, path: Path, job: IngestionJob) -> None:
    stat = path.stat()
    digest = sha256_file(path)
    file = db.scalar(select(File).where(File.canonical_path == str(path)))
    modified = datetime.fromtimestamp(stat.st_mtime)

    if file and file.sha256 == digest and file.ingestion_status == "indexed":
        job.skipped_files += 1
        return

    if not file:
        file = File(
            source_id=source.id,
            canonical_path=str(path),
            windows_path=to_windows_path(path),
            file_type=path.suffix.lower().lstrip("."),
        )
        db.add(file)
        db.flush()

    file.byte_size = stat.st_size
    file.modified_time = modified
    file.sha256 = digest
    file.file_type = path.suffix.lower().lstrip(".")
    file.updated_at = datetime.utcnow()
    file.error = None

    try:
        parsed = parse_document(path)
        db.query(Chunk).filter(Chunk.file_id == file.id).delete()
        if file.document:
            file.document.title = parsed.title
            file.document.page_count = parsed.page_count
            file.document.parser = parsed.parser
        else:
            db.add(Document(file_id=file.id, title=parsed.title, page_count=parsed.page_count, parser=parsed.parser))

        for parsed_chunk in parsed.chunks:
            digest_chunk = chunk_hash(parsed_chunk.text, parsed_chunk.page_or_segment)
            db.add(
                Chunk(
                    file_id=file.id,
                    text=parsed_chunk.text,
                    page_or_segment=parsed_chunk.page_or_segment,
                    chunk_hash=digest_chunk,
                    vector_record_id=f"sqlite:{file.id}:{digest_chunk[:16]}",
                )
            )
        file.ingestion_status = "indexed"
        job.indexed_files += 1
    except Exception as exc:  # noqa: BLE001
        file.ingestion_status = "failed"
        file.error = str(exc)
        job.failed_files += 1
