# Product hybrid search integration

Status: implemented. This scoped integration follows the user's request to unify the
actual search flow; the broader TODO and real-video inference remain paused.
Backend owns ranked retrieval/SQL adapters/API; frontend owns unified rendering.

## Problem and requirements

The existing photo route uses only the shared fusion primitive. V2 executes a
separate document search and returns two independently limited result arrays.
The independent coordinator, unit aggregation and final store are not used.

- MAIN-1: Actual `/api/v2/search`, `/api/photos/search` and document chat use a
  product service backed by StagedSearch -> HybridSearch, rather than fusing
  already-final photo/document payloads. Photo branches reuse existing encoder
  identity checks, intent/metadata/OCR rules and pre-window constraints. Document
  TF-IDF/exact boosts remain a ranked-ID route using the revision-aware cache.
  SQL/model specifics stay outside the standard-library hybrid core.
- MAIN-2: Stable disjoint unit IDs: image=file_id*2, document chunk=chunk_id*2+1;
  parent asset IDs are File.id. No database ID/index rewrite. Resource order is
  its best fused unit; at most 2 supporting chunks per file. Shared ranked fusion
  uses existing photo weights and document weight 1, constant 60, candidate branch
  windows max(50, limit*2), fused unit window bounded 1000. Default final limit is
  one total resource budget, not a separate allowance for each modality. Positive
  branch contributions are rank-only; no cross-model native score comparison.
  Deterministic unit-ID ties are intentional for the new unified ranking.
- MAIN-3: Candidate features (including document index text) and final records
  are separate. Photo SQL projections read scoring/filter fields rather than full
  File/Image entities. An optional evidence-aware final store receives selected
  AssetHits (parents plus exact supporting unit IDs), alongside the unchanged
  DocumentStore.fetch(ids) interface. These store choices are mutually exclusive.
  One final store batch hydrates only selected files,
  image presentation fields and selected supporting chunks, in bounded SQL
  batches. Empty results cause no final fetch; missing/stale/ineligible records
  fail closed. This is SQL record hydration, not original media-file reading.
- MAIN-4: Preserve existing routes/request defaults and old response fields.
  V2 adds `results` with global rank, raw RRF score, asset type and typed payloads;
  old photo/document arrays project those selected resources in global order.
  Their display scores retain route-specific semantics; global results.score is
  authoritative for cross-modality order and is not confidence. Legacy photo and
  chat use the same product service in explicit modality scope; chat retains
  grounded excerpts/native document scores for the existing extractive answerer.
  Document-only search must not construct an image embedder.
- MAIN-5: Frontend image search calls V2, offers all/photo/screenshot/document
  scope and renders `results` in authoritative order with citations/preview data.
  Keep pending/error/empty states, no new model/serving controls or cloud path.
- MAIN-6: Temporary-database acceptance includes mixed requests/global limits,
  file/chunk ID collisions, repeated chunks, stable repeat ordering, source/type/
  year/person/location filters before windows, stale encoder/OCR behavior,
  document cache invalidation/rollback, final-fetch IDs/counts/empty/failure, and
  actual ASGI requests for legacy/v2/chat. Check frontend handlers and rendering
  with mocked hooks/HTTP; browser acceptance is separate. Compare photo-only adapter rankings to
  the old implementation where deterministic-tie policy permits. Run full
  workspace/pytest/Ruff and frontend build, recording actual evidence separately
  from retrieval accuracy, serving latency and private-library acceptance.

## Compatibility, failures and non-goals

No schema migration, private-library scan/reindex, dependencies/model changes,
Tantivy/FAISS activation, cache service, CPU-video inference, GPU/cloud operation,
background task or publication. The route/store interfaces permit future native
indexes and rerankers but their presence is not product activation. Existing
public helpers/imports remain. Failure is strict; no silent branch fallbacks.
The existing embedder selection policy is unchanged. Cross-modality ranking,
file-level deduplication and V2's total resource limit intentionally replace the
old two independently limited arrays; clients needing global order use results.
`limit` is now a file budget, including for chat (default 8, validated range
1..100); supporting citations can reach twice that file budget. Out-of-range chat
limits return HTTP 422. Fixed candidate windows may return fewer unique files
when one file dominates the window; no unbounded backfill is promised. Date,
person and location constraints retain their existing photo-specific meaning;
this integration does not invent document metadata to apply those filters.

Photo-only scores and payload policy are retained, while exact score ties now
use unit ID order rather than the legacy secondary confidence ordering. Photo
compatibility scores and native document scores remain in separate score spaces;
only `results.score`/`rank` define cross-modality ordering. Branch timings cover
cached ranked-ID routing; `candidate_preparation_ms` separately includes SQL,
index and encoding work, and `pipeline_elapsed_ms` covers the shared stages.

Acceptance means actual source/API integration and local correctness checks,
not measured recall gains or deployment to a running user service.
