"""Small, additive, versioned migrations; no index rebuilds or source access."""

from __future__ import annotations


def install_task_constraints(_metadata, conn, **_kwargs) -> None:
    # SQLite serializes writes. Triggers provide atomic active-only uniqueness
    # even on SQLite versions predating partial indexes (3.8.0).
    columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(processing_tasks)")}
    if "request_key" not in columns:
        return  # Existing databases add this column during migration below.
    for operation in ("INSERT", "UPDATE"):
        conn.exec_driver_sql(
            f"CREATE TRIGGER IF NOT EXISTS active_task_request_{operation.lower()} "
            f"BEFORE {operation} ON processing_tasks "
            "WHEN NEW.request_key IS NOT NULL AND NEW.state IN ('queued', 'running') BEGIN "
            "SELECT RAISE(ABORT, 'duplicate active task request') WHERE EXISTS ("
            "SELECT 1 FROM processing_tasks WHERE request_key = NEW.request_key "
            "AND state IN ('queued', 'running') AND id != NEW.id); END"
        )


def migrate_consistency(conn) -> None:
    conn.exec_driver_sql("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY)")
    if conn.exec_driver_sql("SELECT 1 FROM schema_migrations WHERE version = 2").first():
        return
    additions = {
        "images": {"index_profile": "TEXT"},
        "photo_vectors": {"encoder_id": "TEXT"},
        "processing_tasks": {
            "claim_token": "VARCHAR(64)",
            "lease_expires_at": "DATETIME",
            "request_key": "VARCHAR(64)",
        },
    }
    for table, columns in additions.items():
        existing = {row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table})")}
        for name, kind in columns.items():
            if name not in existing:
                conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
    conn.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS ix_processing_tasks_lease_expires_at ON processing_tasks(lease_expires_at)"
    )
    conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_processing_tasks_request_key ON processing_tasks(request_key)")
    install_task_constraints(None, conn)
    conn.exec_driver_sql("INSERT INTO schema_migrations(version) VALUES (2)")
