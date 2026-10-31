# Architecture and Current Limits

The API and background controller are a modular monolith. Subprocess workers keep
inference models warm; a separate vLLM service owns Qwen-VL's GPU runtime.

## Data Flow

- Document scanner: configured roots -> SHA-256 -> native parsers -> page/segment
  chunks -> SQLite -> local TF-IDF/extractive answer with citations.
- Photo worker: configured root -> hash/EXIF/thumbnail -> Qwen context -> CLIP image
  embedding plus BGE text embeddings -> SQLite photo vectors.
- Optional faces: InsightFace observations -> incremental centroid clusters ->
  user names/aliases -> person context refresh jobs.
- Search: rule-based query plan -> unfused photo namespace/document TF-IDF
  rankings -> StagedSearch/HybridSearch weighted RRF -> file aggregation and one
  total limit -> selected file/evidence SQL reads -> unified V2 results. Legacy
  photo and document-chat routes use the same service in explicit scopes. Raw RRF
  and the photo heuristic confidence are not calibrated probabilities.

The `observations`, `inferences`, and `metadata` sections distinguish visible
facts, uncertain model output, and deterministic/user-confirmed data. Legacy
payload readers remain available during schema rollout.

## Durable Work

`ProcessingTask` stores stage, idempotency key, state, priority, and attempts.
`TaskPayload` stores inputs; `ProcessingRun` records each attempt and output stats.
Workers claim queued rows using conditional updates, a unique ownership token and
a renewable 120-second lease. The controller recovers expired leases every loop;
output commits check the ownership token in the same write transaction. Active
identical requests deduplicate through database triggers, including on older
SQLite versions without partial indexes; retries retain separate run records. Run one
API/controller process; multiple Uvicorn controllers remain unsupported.

Cancellation targets queued tasks. Pause and concurrency reduction drain workers:
the current task finishes, then the process exits without claiming another task.
Shutdown grants bounded grace, then kills only owned workers; unfinished work
recovers on lease expiry. This is not a distributed or exactly-once queue.

Source/stage cursors resume bounded deterministic scans and wrap at the end.
Unavailable sources fail rather than implying deletion; only successful full
document traversals infer missing files. Photo hash checks precede indexed skips.
Mixed-success batches report failed with per-file totals in the run record.

## Boundaries

Original files are read-only. Application state stays in the configured data
directory. APIs are for one trusted local user and are not authenticated.

SQLite vectors retain explicit encoder/revision/preprocessing identities. Unknown
legacy semantic vectors are excluded until an explicit reindex; no migration
rebuilds the library. Query filters precede per-retriever top-k, and V2 routes
document requests to lexical retrieval with citations. Vector decoding/scoring is
batched behind a scorer protocol; people metadata is fetched in batches.
OCR exact-text retrieval works independently of missing/incompatible vectors.
Character n-gram TF-IDF is cached by a transactional database revision, independent
of the query. Cache invalidation covers content, source and indexing-state changes.
JSON vectors still require a linear scan; the cache still builds a corpus matrix.
Product activation of optional ANN indexing, document layout/OCR and rerankers,
calibrated confidence, and large-scale product quality evaluation remain future
work. Recall@K and Hit@K are separate metrics.
See the [roadmap](roadmap/README.md) for the separate Context Plane proposal.

## Independent Hybrid Retrieval

`backend/app/retrieval/hybrid` separates query/filter contracts, ranked-ID
retrievers, a replaceable fusion strategy, and final record fetching. Its core
uses only the standard library; local BM25 and exact-vector adapters are separate.
The coordinator runs bounded branches, reports explicit partial failures when
selected, and fetches final top-k records once. The old photo fusion facade now
delegates to the same rank-only primitive, preserving its external response shape.
Spec0009 connects the product's `asset_search` service to the coordinator. Photo
scoring uses explicit feature projections; document ranking reuses its existing
corpus index. Disjoint image/chunk unit IDs map to parent File IDs; the staged
pipeline caps supporting units, selects files and invokes one evidence-aware SQL
store call for final presentation. V2 adds globally ordered `results`, with old
arrays as projections. Candidate/index reads, including a cold TF-IDF corpus
build, remain separate costs from final selected-file/chunk hydration. Optional
native indexes, model rerankers, caches and cancellation are not enabled here.

Spec 0003 evaluates the layer with public labeled queries and real CPU BM25 plus
corpus-fitted latent semantic vectors. This does not validate pretrained model
quality or replace the product's default retrievers. RRF constants and windows can
change rankings; increasing the window does not guarantee better retrieval.

## Opt-in multimodal quality stages

Spec 0006 adds resource-linked units and versioned text evidence, plus separate
candidate/rerank/resource budgets around the existing hybrid coordinator. The
initial pipeline accepts text queries over image/page/text units, preserves
precise locators, caps repeated units and aggregates by the best unit. Candidate
features and the one final asset-store batch are distinct costs. A UnitStore
protocol and Reranker protocol keep feature storage/model execution replaceable.

New diagnostics attribute misses to route windows, fused candidates and final
ranking. Existing SciFact artifacts remain fixed historical controls; generated
PNG/PDF experiments actually read pixels/native text while OCR/captions are supplied
synthetic annotations. The six-color pixel route and identifier rules are limited
reference adapters. No pretrained semantic/visual quality, actual OCR, image-query
API or general audio/video quality is implied by these initial controls. Product
integration is separately recorded in spec0009 and is not an accuracy gain.
