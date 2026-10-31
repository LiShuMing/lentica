# Index, Task and Retrieval Consistency

## Problem and Scope

Bounded scans repeatedly visit a prefix, workers can strand running tasks, model
fallbacks share vector namespaces, and search plans/metrics overstate behavior.
Fix these contracts within the local product; preserve original source files.

## Requirements

- SCAN-1: Persist bounded scan progress per source/stage; wrap only at a completed
  traversal. Recheck content hashes before skipping photos. An unavailable source
  must not imply deletion. Partial scans never infer missing files.
- TASK-1: Atomic claims have owner tokens and renewable leases. Recover expired
  ownership during normal controller operation. Pause/concurrency reduction drains
  current work. Stale owners cannot complete a reassigned attempt. Stable active
  request identities prevent duplicate submissions. Partial file failure must
  produce a failed task with preserved output statistics.
  Database-enforced active request uniqueness must also work on SQLite versions
  without partial indexes.
- VECTOR-1: Persist actual encoder identity, revision/configuration and dimensions;
  compare only compatible spaces. Unknown legacy semantic vectors remain readable
  as records but are excluded from semantic ranking until explicitly reindexed.
  Updating a photo replaces its previous vector set atomically.
- SEARCH-1: Execute query plan branches and explicit asset types. Apply person,
  year and location restrictions before per-retriever top-k. Document results have
  citations. Reuse a bounded revision-keyed document index cache; batch photo
  people reads and expose a vector scoring boundary for future indexes.
  OCR exact-text retrieval remains available without compatible semantic vectors.
- EVAL-1: Standard Recall@K divides retrieved relevant items by all relevant items;
  report Hit@K separately. Confidence remains a heuristic, not a probability.
- COMPAT-1: Keep existing routes and fields. V2 adds document hits. Additive,
  versioned database migrations never rebuild indexes or alter original sources.
  Queue settings updates are atomic. Configuration/help checks do not invoke models.

## Ownership and Design

Backend owns scan cursors, task leases, vector contracts, retrieval and migrations.
Frontend owns accurate labels and additive V2 response types. See ADR 0002 and
workspace.toml. Retain the modular monolith and separate model runtime; no new
product dependency or remote service is introduced.

## Compatibility and Failure Behavior

Old task payloads remain valid; new leases and cursors use additive columns/tables.
Old photo vectors have unknown encoder provenance and cannot silently match a
different model. Synthetic hash vectors have a deterministic known format.
Source failures are reported, not treated as an empty successful scan. Retry keeps
completed file outputs and uses stable input identities; execution is at-least-once.
Shutdown has bounded grace, then termination; unfinished attempts recover by lease.

## Acceptance

Synthetic temporary libraries exceed 100 documents and 1000 photos; bounded calls
eventually visit the suffix and survive a new scanner instance. Test source loss,
same-path changes, active enqueue deduplication, pause/drain, lease expiry and stale
completion, partial failures, encoder revision/dimension mismatch, pre-top-k filters,
document V2 routing/citations, cache invalidation, standard recall and hit rate.
Run root pytest, workspace/spec checks, Ruff and frontend build after UI changes.
Record actual results in verification.md.

## Non-goals

No user-library scan/reindex, GPU/cloud execution, Context Pack, distributed queue,
ANN service, authentication redesign, or claimed retrieval/performance improvement.
