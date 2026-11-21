# Reliability and Retrieval Backlog

Derived from the [2026-10-28 review](../reviews/2026-10-28.md) and reconciled
against `main` through spec `0011`. Each remaining item needs a numbered spec
and verification record before behavior changes. Finish correctness/recovery
before adding models or a new storage service. No private-library reindex is
implied.

| ID | Status / owner | Next Slice | Acceptance |
| --- | --- | --- | --- |
| REL-1 | Implemented (`0002`) backend/workers | Worker ownership, heartbeats/leases, graceful `SIGUSR1` drain, expiry recovery, active request deduplication | Covered in `specs/0002-index-task-consistency`; bounded retry backoff remains under REL-6 |
| REL-2 | Implemented (`0002`) embedding/storage | Persist effective `encoder_id` provenance and dimensions; exclude incompatible/legacy vectors from semantic ranking; atomic vector replacement | Covered in `specs/0002-index-task-consistency`; reversible multi-version reindex remains an operational follow-up |
| REL-3 | Implemented (`0002`, `0009`) retrieval/API | Honor query plan branches, asset types, document routing/citations, pre-top-k person/year/location filters, and unified hybrid search | Covered in `specs/0002-index-task-consistency` and `specs/0009-product-hybrid-integration` |
| REL-4 | P2 planned ingestion | EXIF nested IFD extraction and orientation-aware thumbnails/faces | Synthetic JPEGs containing Exif/GPS IFDs preserve DateTimeOriginal and signed coordinates; corrupt/missing metadata is safe; rotated face crops agree with displayed images; no inferred date/location replaces confirmed metadata |
| REL-5 | P2 planned backend/frontend | Count actual model calls, namespace writes, stage timings, tokens/pixels where reported; show degraded mode and query failures | Counts match persisted rows in multi-namespace and fallback fixtures; no invented FLOPs; UI handles API/config errors and stale polling; tests cover loading/error/empty and task transitions |
| REL-6 | P2 planned ingestion/workers | Retry budget, cooldown, fairness on repeatedly failing files, batched hashing/embedding | An early corrupt file cannot starve healthy later files; repeated model failure cannot spin indefinitely; bounded batches and crash recovery remain idempotent; benchmark hashing and encoder time separately |
| REL-7 | P2 planned retrieval/evaluation | Label a private representative evaluation set, freeze baseline/model fingerprint, add category reports and persistent lexical/ANN candidate indices only after measurement | Recall@1/5/10, MRR, nDCG@10, filter accuracy and cold/warm P50/P95 measured for visual/OCR/person/time/location/doc queries; compare old/new on identical inputs; report footprint, failures and sample counts, not hash-mode quality (public CPU baselines live in `0003`–`0010`) |
| REL-8 | P3 planned workspace/CI | Reproducible formatter/dependency versions, optional runtime contract jobs, stronger lint/types incrementally | Base Python 3.11/3.12 and frontend builds remain green; selected vLLM/GPU environments have opt-in smoke evidence; notebook/spec checks cannot be bypassed by executable cells (`0011`); no cloud/GUI actions or downloads occur in default CI |

## Order and Boundaries

REL-1 through REL-3 landed in specs `0002` and `0009`, and strict queue settings
validation, explicit face-job failure, and notebook governance landed in `0011`.
Next, REL-4 establishes reliable EXIF/orientation evidence and REL-5/REL-6
expose and bound runtime work. Build REL-7's private baseline before deciding
whether Qdrant, dedicated OCR, text reranking, or VLM reranking is justified
for default product search. Evaluate a small slice at a time rather than keeping
all models resident on GPU.

The product, service, cloud media CLI, GUI learning example, and GPU lessons share
Git/spec conventions, not a runtime or import graph. Keep environments independent
and reuse selected interpreters. The separate Context Plane proposal is not a
prerequisite for these reliability changes and does not supersede root AGENTS.md.
