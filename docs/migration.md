# Migrating From Personal Knowledge Lab

Lentica is a fresh source-only repository extracted from the previous working tree,
including its then-uncommitted implementation and tests. It does not include the
old Git history, databases, photos, logs, virtualenvs, or model caches.

1. Install Lentica into an existing or chosen Python environment.
2. Set `LENTICA_DOCUMENTS_SOURCE`, `LENTICA_PHOTOS_SOURCE`, and model paths/IDs.
3. Use a fresh `LENTICA_DATA_DIR` for initial validation. Validate with a small
   synthetic library before scanning a large personal collection.
4. To reuse an index, stop the old backend and workers, back up the old data
   directory, and point `LENTICA_DATA_DIR` at the stopped instance's data directory.
5. Review the persisted `queue_settings.json`: old endpoint/model settings override
   initial environment defaults. Face indexing now requires explicit opt-in.

Only one backend should own a given data directory. The API paths and the
`backend.app.*` module namespace are retained. The schema uses the existing
lightweight migrations; no automatic collection deletion or full reindex is run
as part of repository migration.
