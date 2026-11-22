from __future__ import annotations

from datetime import datetime, timedelta
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
from backend.app.ingestion.traversal import iter_source_files, scan_cursor


def ensure_default_sources(db: Session, commit=None) -> None:
    settings = get_settings()
    defaults = [
        ("Documents", str(settings.documents_source), "documents"),
        ("Photos", str(settings.photos_source), "images"),
    ]
    for name, root_path, kind in defaults:
        exists = db.scalar(select(Source).where(Source.root_path == root_path))
        if not exists:
            if db.scalar(select(Source.id).where(Source.name == name)):
                name = f"{name} ({root_path})"
            db.add(Source(name=name, root_path=root_path, kind=kind, enabled=True))
    (commit or db.commit)()


DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_COOLDOWN_SECONDS = 300


def scan_documents(
    db: Session,
    source_id: int | None = None,
    max_files: int | None = 100,
    max_bytes: int = 20 * 1024 * 1024,
    commit=None,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_cooldown_seconds: int = DEFAULT_RETRY_COOLDOWN_SECONDS,
    now: datetime | None = None,
) -> IngestionJob:
    commit = commit or db.commit
    current_time = now or datetime.utcnow()
    ensure_default_sources(db, commit=commit)
    job = IngestionJob(status="running")
    db.add(job)
    commit()
    db.refresh(job)

    query = select(Source).where(Source.enabled.is_(True), Source.kind == "documents")
    if source_id is not None:
        query = query.where(Source.id == source_id)
    sources = list(db.scalars(query.order_by(Source.last_scan_at.asc(), Source.id.asc())))

    processed = 0
    errors: list[str] = []
    try:
        for source in sources:
            if max_files is not None and processed >= max_files:
                break
            seen_paths: set[str] = set()
            try:
                cursor = scan_cursor(db, source.id, "documents")
                after = cursor.path if max_files is not None else ""
                exhausted = True
                for path in iter_source_files(Path(source.root_path), SUPPORTED_DOCUMENT_EXTENSIONS, after=after):
                    if max_files is not None and processed >= max_files:
                        exhausted = False
                        break
                    seen_paths.add(str(path))
                    job.total_files += 1
                    try:
                        if path.stat().st_size <= max_bytes:
                            _index_one_file(
                                db,
                                source,
                                path,
                                job,
                                max_retries=max_retries,
                                retry_cooldown_seconds=retry_cooldown_seconds,
                                now=current_time,
                            )
                        else:
                            job.skipped_files += 1
                    except OSError as exc:
                        job.failed_files += 1
                        errors.append(str(exc))
                    cursor.path = str(path)
                    processed += 1
                    commit()
                if exhausted:
                    cursor.path = ""
                if max_files is None:
                    _mark_missing(db, source, seen_paths)
            except OSError as exc:
                # A missing/misbehaving source must not block other sources.
                errors.append(str(exc))
            source.last_scan_at = datetime.utcnow()
            commit()
        job.status = "failed" if job.failed_files or errors else "succeeded"
        job.error = "; ".join(errors) or (f"{job.failed_files} document(s) failed" if job.failed_files else None)
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
    finally:
        job.finished_at = datetime.utcnow()
        commit()
        db.refresh(job)
    return job


def _mark_missing(db: Session, source: Source, seen_paths: set[str]) -> None:
    existing = db.scalars(select(File).where(File.source_id == source.id)).all()
    for file in existing:
        if (
            file.canonical_path not in seen_paths
            and Path(file.canonical_path).suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
        ):
            file.ingestion_status = "missing"
            file.updated_at = datetime.utcnow()


def _iter_supported_files(root: Path, max_bytes: int):
    for path in iter_source_files(root, SUPPORTED_DOCUMENT_EXTENSIONS):
        if path.stat().st_size <= max_bytes:
            yield path


def _should_skip_failed_file(
    file: File | None,
    digest: str,
    *,
    max_retries: int,
    now: datetime,
) -> bool:
    """Skip unchanged failing files while in cooldown or after exhausting retries."""
    if not file or file.ingestion_status != "failed" or file.sha256 != digest:
        return False
    failures = int(file.failure_count or 0)
    if failures >= max_retries:
        return True
    return file.next_retry_at is not None and file.next_retry_at > now


def _index_one_file(
    db: Session,
    source: Source,
    path: Path,
    job: IngestionJob,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_cooldown_seconds: int = DEFAULT_RETRY_COOLDOWN_SECONDS,
    now: datetime | None = None,
) -> None:
    current_time = now or datetime.utcnow()
    stat = path.stat()
    digest = sha256_file(path)
    file = db.scalar(select(File).where(File.canonical_path == str(path)))
    modified = datetime.fromtimestamp(stat.st_mtime)

    if file and file.sha256 == digest and file.ingestion_status == "indexed":
        job.skipped_files += 1
        return
    if _should_skip_failed_file(file, digest, max_retries=max_retries, now=current_time):
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
    elif file.sha256 != digest:
        file.failure_count = 0
        file.next_retry_at = None

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
        file.failure_count = 0
        file.next_retry_at = None
        job.indexed_files += 1
    except Exception as exc:  # noqa: BLE001
        failures = int(file.failure_count or 0) + 1
        file.ingestion_status = "failed"
        file.error = str(exc)
        file.failure_count = failures
        backoff = max(0, retry_cooldown_seconds) * (2 ** (failures - 1))
        file.next_retry_at = current_time + timedelta(seconds=backoff) if backoff > 0 else None
        job.failed_files += 1
