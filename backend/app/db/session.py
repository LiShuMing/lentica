from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.core.config import get_settings


class Base(DeclarativeBase):
    pass


engine = create_engine(
    get_settings().database_url,
    connect_args={"check_same_thread": False, "timeout": 30},
)


@event.listens_for(engine, "connect")
def _sqlite_options(connection, _record):
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from backend.app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _ensure_lightweight_migrations()


def _ensure_lightweight_migrations() -> None:
    from backend.app.db.migrations import migrate_consistency

    image_columns = {
        "width": "INTEGER",
        "height": "INTEGER",
        "gps_lat": "VARCHAR(80)",
        "gps_lon": "VARCHAR(80)",
        "thumb_path": "TEXT",
        "context_json": "TEXT",
        "context_version": "VARCHAR(80)",
        "face_analysis_status": "VARCHAR(40) DEFAULT 'pending'",
        "face_indexed_at": "DATETIME",
        "indexed_at": "DATETIME",
    }
    with engine.begin() as conn:
        migrate_consistency(conn)
        existing = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(images)").fetchall()}
        for name, ddl_type in image_columns.items():
            if name not in existing:
                conn.exec_driver_sql(f"ALTER TABLE images ADD COLUMN {name} {ddl_type}")
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS person_clusters (
                id INTEGER PRIMARY KEY,
                label VARCHAR(80) UNIQUE NOT NULL,
                display_name TEXT,
                centroid_embedding_json TEXT,
                face_count INTEGER DEFAULT 0 NOT NULL,
                cover_file_id INTEGER,
                confidence VARCHAR(80),
                ignored BOOLEAN DEFAULT 0 NOT NULL,
                created_at DATETIME,
                updated_at DATETIME,
                FOREIGN KEY(cover_file_id) REFERENCES files(id)
            )
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS face_detections (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL,
                cluster_id INTEGER,
                bbox_json TEXT NOT NULL,
                embedding_json TEXT,
                detection_score VARCHAR(80),
                detector VARCHAR(80) DEFAULT 'insightface' NOT NULL,
                created_at DATETIME,
                FOREIGN KEY(file_id) REFERENCES files(id),
                FOREIGN KEY(cluster_id) REFERENCES person_clusters(id)
            )
            """
        )
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_face_detections_file_id ON face_detections(file_id)")
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_face_detections_cluster_id ON face_detections(cluster_id)")
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS person_aliases (
                id INTEGER PRIMARY KEY,
                cluster_id INTEGER NOT NULL,
                alias VARCHAR(120) NOT NULL,
                created_at DATETIME,
                FOREIGN KEY(cluster_id) REFERENCES person_clusters(id),
                CONSTRAINT uq_person_alias_cluster_alias UNIQUE (cluster_id, alias)
            )
            """
        )
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_person_aliases_cluster_id ON person_aliases(cluster_id)")
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_person_aliases_alias ON person_aliases(alias)")
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS photo_metadata_candidates (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL,
                kind VARCHAR(40) NOT NULL,
                value TEXT NOT NULL,
                source VARCHAR(80) NOT NULL,
                confidence VARCHAR(40) NOT NULL,
                created_at DATETIME,
                FOREIGN KEY(file_id) REFERENCES files(id),
                CONSTRAINT uq_photo_metadata_candidate UNIQUE (file_id, kind, value, source)
            )
            """
        )
        conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_photo_metadata_candidates_file_id ON photo_metadata_candidates(file_id)"
        )
        conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_photo_metadata_candidates_kind ON photo_metadata_candidates(kind)"
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS geocode_cache (
                id INTEGER PRIMARY KEY,
                lat_key VARCHAR(40) NOT NULL,
                lon_key VARCHAR(40) NOT NULL,
                display_name TEXT,
                city TEXT,
                district TEXT,
                country TEXT,
                provider VARCHAR(80) DEFAULT 'local' NOT NULL,
                raw_json TEXT,
                updated_at DATETIME
            )
            """
        )
        conn.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_geocode_cache_lat_lon ON geocode_cache(lat_key, lon_key)"
        )
