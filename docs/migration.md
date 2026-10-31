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

## Index/task consistency (spec 0002)

Startup performs an additive, versioned SQLite migration for scan cursors, task
leases/request identities, and photo encoder provenance. It retains existing
files, vectors and task payloads, without scanning or rebuilding a library.
Back up an existing database before switching application versions.

Historical semantic vectors do not identify the model revision/preprocessing that
created them. They remain stored but are excluded from semantic ranking. To
replace them, deliberately submit a photo indexing job with `force: true` to the
existing `/api/photos/index` endpoint and choose the intended encoder settings.
This is an explicit operation, not a step run by the migration. Hash smoke vectors
use a known deterministic format and remain compatible.

V2 search keeps `hits` for photos and adds `document_hits` with chunk/file citations.
Explicit `asset_types` values are `photo`, `screenshot`, and `document`; unknown
values are rejected. Photo person/year/location filters apply before candidate
truncation. Document years remain lexical terms because document content-time
metadata is not yet modeled. Confidence fields remain compatibility fields holding
heuristic relevance, not calibrated probabilities.
