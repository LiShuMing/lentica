from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.session import Base


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    root_path: Mapped[str] = mapped_column(Text, unique=True)
    kind: Mapped[str] = mapped_column(String(50), default="documents")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_scan_at: Mapped[datetime | None] = mapped_column(DateTime)

    files: Mapped[list[File]] = relationship(back_populates="source")


class File(Base):
    __tablename__ = "files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    canonical_path: Mapped[str] = mapped_column(Text, unique=True)
    windows_path: Mapped[str] = mapped_column(Text)
    file_type: Mapped[str] = mapped_column(String(40))
    byte_size: Mapped[int] = mapped_column(Integer, default=0)
    modified_time: Mapped[datetime | None] = mapped_column(DateTime)
    sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    ingestion_status: Mapped[str] = mapped_column(String(40), default="pending")
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    source: Mapped[Source] = relationship(back_populates="files")
    document: Mapped[Document | None] = relationship(back_populates="file", uselist=False)
    chunks: Mapped[list[Chunk]] = relationship(back_populates="file", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_id: Mapped[int] = mapped_column(ForeignKey("files.id"), unique=True)
    title: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column(Integer)
    parser: Mapped[str] = mapped_column(String(80))

    file: Mapped[File] = relationship(back_populates="document")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("file_id", "chunk_hash", name="uq_file_chunk_hash"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_id: Mapped[int] = mapped_column(ForeignKey("files.id"), index=True)
    text: Mapped[str] = mapped_column(Text)
    page_or_segment: Mapped[str] = mapped_column(String(80))
    chunk_hash: Mapped[str] = mapped_column(String(64), index=True)
    vector_record_id: Mapped[str | None] = mapped_column(String(120))

    file: Mapped[File] = relationship(back_populates="chunks")


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_id: Mapped[int] = mapped_column(ForeignKey("files.id"), unique=True)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    exif_time: Mapped[datetime | None] = mapped_column(DateTime)
    gps_lat: Mapped[str | None] = mapped_column(String(80))
    gps_lon: Mapped[str | None] = mapped_column(String(80))
    location: Mapped[str | None] = mapped_column(Text)
    thumb_path: Mapped[str | None] = mapped_column(Text)
    caption: Mapped[str | None] = mapped_column(Text)
    ocr_text: Mapped[str | None] = mapped_column(Text)
    tags: Mapped[str | None] = mapped_column(Text)
    context_json: Mapped[str | None] = mapped_column(Text)
    context_version: Mapped[str | None] = mapped_column(String(80))
    visual_analysis_status: Mapped[str] = mapped_column(String(40), default="unavailable")
    face_analysis_status: Mapped[str] = mapped_column(String(40), default="pending")
    face_indexed_at: Mapped[datetime | None] = mapped_column(DateTime)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime)


class PersonCluster(Base):
    __tablename__ = "person_clusters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label: Mapped[str] = mapped_column(String(80), unique=True)
    display_name: Mapped[str | None] = mapped_column(Text)
    centroid_embedding_json: Mapped[str | None] = mapped_column(Text)
    face_count: Mapped[int] = mapped_column(Integer, default=0)
    cover_file_id: Mapped[int | None] = mapped_column(ForeignKey("files.id"))
    confidence: Mapped[str | None] = mapped_column(String(80))
    ignored: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PersonAlias(Base):
    __tablename__ = "person_aliases"
    __table_args__ = (UniqueConstraint("cluster_id", "alias", name="uq_person_alias_cluster_alias"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cluster_id: Mapped[int] = mapped_column(ForeignKey("person_clusters.id"), index=True)
    alias: Mapped[str] = mapped_column(String(120), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FaceDetection(Base):
    __tablename__ = "face_detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_id: Mapped[int] = mapped_column(ForeignKey("files.id"), index=True)
    cluster_id: Mapped[int | None] = mapped_column(ForeignKey("person_clusters.id"), index=True)
    bbox_json: Mapped[str] = mapped_column(Text)
    embedding_json: Mapped[str | None] = mapped_column(Text)
    detection_score: Mapped[str | None] = mapped_column(String(80))
    detector: Mapped[str] = mapped_column(String(80), default="insightface")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PhotoMetadataCandidate(Base):
    __tablename__ = "photo_metadata_candidates"
    __table_args__ = (
        UniqueConstraint("file_id", "kind", "value", "source", name="uq_photo_metadata_candidate"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_id: Mapped[int] = mapped_column(ForeignKey("files.id"), index=True)
    kind: Mapped[str] = mapped_column(String(40), index=True)
    value: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(80))
    confidence: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class GeocodeCache(Base):
    __tablename__ = "geocode_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    lat_key: Mapped[str] = mapped_column(String(40))
    lon_key: Mapped[str] = mapped_column(String(40))
    display_name: Mapped[str | None] = mapped_column(Text)
    city: Mapped[str | None] = mapped_column(Text)
    district: Mapped[str | None] = mapped_column(Text)
    country: Mapped[str | None] = mapped_column(Text)
    provider: Mapped[str] = mapped_column(String(80), default="local")
    raw_json: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PhotoVector(Base):
    __tablename__ = "photo_vectors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_id: Mapped[int] = mapped_column(ForeignKey("files.id"), index=True)
    namespace: Mapped[str] = mapped_column(String(40), index=True)
    dims: Mapped[int] = mapped_column(Integer)
    vector_json: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str] = mapped_column(String(40), default="queued")
    total_files: Mapped[int] = mapped_column(Integer, default=0)
    indexed_files: Mapped[int] = mapped_column(Integer, default=0)
    skipped_files: Mapped[int] = mapped_column(Integer, default=0)
    failed_files: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_goal: Mapped[str] = mapped_column(Text)
    plan: Mapped[str | None] = mapped_column(Text)
    tool_calls: Mapped[str | None] = mapped_column(Text)
    result: Mapped[str | None] = mapped_column(Text)
    telemetry: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ModelProfile(Base):
    __tablename__ = "model_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[str] = mapped_column(String(200), unique=True)
    role: Mapped[str] = mapped_column(String(80), default="available")
    capabilities: Mapped[str] = mapped_column(Text, default="{}")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ProcessingTask(Base):
    __tablename__ = "processing_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(200), unique=True)
    file_id: Mapped[int | None] = mapped_column(ForeignKey("files.id"))
    content_hash: Mapped[str | None] = mapped_column(String(64))
    pipeline_stage: Mapped[str] = mapped_column(String(80))
    priority: Mapped[int] = mapped_column(Integer, default=100)
    state: Mapped[str] = mapped_column(String(40), default="queued")
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime)
    worker_id: Mapped[str | None] = mapped_column(String(120))
    rate_limit_delay: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)


class TaskPayload(Base):
    __tablename__ = "task_payloads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("processing_tasks.id"), unique=True)
    payload_json: Mapped[str] = mapped_column(Text)


class ProcessingRun(Base):
    __tablename__ = "processing_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("processing_tasks.id"))
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    model_profile: Mapped[str | None] = mapped_column(String(200))
    token_usage: Mapped[str | None] = mapped_column(Text)
    output_artifact_ids: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
