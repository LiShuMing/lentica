from __future__ import annotations

from sqlalchemy import Column, MetaData, Table, create_engine

from backend.app.db.migrations import migrate_consistency


def test_additive_migration_is_repeatable_and_preserves_old_records(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE images (id INTEGER PRIMARY KEY, caption TEXT)")
        conn.exec_driver_sql("CREATE TABLE photo_vectors (id INTEGER PRIMARY KEY, vector_json TEXT)")
        conn.exec_driver_sql("CREATE TABLE processing_tasks (id INTEGER PRIMARY KEY, state TEXT)")
        conn.exec_driver_sql("INSERT INTO images VALUES (1, 'original')")
        conn.exec_driver_sql("INSERT INTO photo_vectors VALUES (1, '[1, 0]')")
        migrate_consistency(conn)
        migrate_consistency(conn)
        assert conn.exec_driver_sql("SELECT caption FROM images").scalar_one() == "original"
        assert conn.exec_driver_sql("SELECT vector_json, encoder_id FROM photo_vectors").one() == ("[1, 0]", None)
        assert conn.exec_driver_sql("SELECT count(*) FROM schema_migrations").scalar_one() == 1
    engine.dispose()


def test_startup_upgrades_legacy_tables_before_task_constraints(tmp_path, monkeypatch):
    from backend.app.db import session
    from backend.app.db.models import Image, PhotoVector, ProcessingTask

    engine = create_engine(f"sqlite:///{tmp_path / 'legacy-startup.db'}")
    legacy = MetaData()
    additions = {"index_profile", "encoder_id", "request_key", "claim_token", "lease_expires_at"}
    for model in (Image, PhotoVector, ProcessingTask):
        Table(
            model.__tablename__,
            legacy,
            *[
                Column(column.name, column.type, primary_key=column.primary_key, nullable=True)
                for column in model.__table__.columns
                if column.name not in additions
            ],
        )
    legacy.create_all(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("INSERT INTO images(id, caption) VALUES (1, 'retained')")
    monkeypatch.setattr(session, "engine", engine)
    session.init_db()
    session.init_db()
    with engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT caption, index_profile FROM images").one() == ("retained", None)
        triggers = conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type = 'trigger'").scalars().all()
        assert "active_task_request_insert" in triggers
        assert "active_task_request_update" in triggers
        assert "doc_revision_chunks_insert" in triggers
    engine.dispose()
