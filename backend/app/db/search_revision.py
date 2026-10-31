"""Transactional corpus revision for bounded, cross-process search caching."""

from __future__ import annotations


def install_revision(_metadata, conn, **_kwargs) -> None:
    conn.exec_driver_sql(
        "CREATE TABLE IF NOT EXISTS document_search_revision (id INTEGER PRIMARY KEY, revision INTEGER NOT NULL)"
    )
    conn.exec_driver_sql("INSERT OR IGNORE INTO document_search_revision VALUES (1, 0)")
    operations = {
        "chunks": ("INSERT", "UPDATE", "DELETE"),
        "documents": ("INSERT", "UPDATE", "DELETE"),
        "files": ("UPDATE", "DELETE"),
        "sources": ("UPDATE", "DELETE"),
    }
    for table, events in operations.items():
        for operation in events:
            timing = operation
            condition = ""
            if table == "files":
                if operation == "UPDATE":
                    timing = "UPDATE OF ingestion_status, canonical_path, windows_path, source_id"
                condition = " WHEN EXISTS (SELECT 1 FROM chunks WHERE file_id = OLD.id)"
            elif table == "sources":
                if operation == "UPDATE":
                    timing = "UPDATE OF enabled, kind, root_path"
                    condition = " WHEN OLD.kind = 'documents' OR NEW.kind = 'documents'"
                else:
                    condition = " WHEN OLD.kind = 'documents'"
            conn.exec_driver_sql(
                f"CREATE TRIGGER IF NOT EXISTS doc_revision_{table}_{operation.lower()} "
                f"AFTER {timing} ON {table}{condition} BEGIN "
                "UPDATE document_search_revision SET revision = revision + 1 WHERE id = 1; END"
            )
