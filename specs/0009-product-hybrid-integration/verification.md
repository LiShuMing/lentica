# Verification

Implemented and locally verified on 2026-10-31. Existing unrelated working-tree
changes are preserved. Broad TODO/video work remains paused. No commit/push,
private-library scan/reindex, model download, running-service restart or real
video-corpus evaluation was performed.

## Source and acceptance coverage

- MAIN-1/2/4: `asset_search` prepares unfused photo branch rankings and native
  document TF-IDF rankings, then calls `StagedSearch -> HybridSearch`. Real ASGI
  requests for V2, legacy photo and chat spy on the actual HybridSearch method.
  Mixed SQL fixtures cover a global file limit, genuine File/Chunk ID collisions,
  stable repeat output, per-file supporting-chunk caps and compatibility arrays.
  The photo-only payload matches the legacy policy on the non-tie fixture; the
  new route still works when legacy `weighted_rrf` is forbidden.
- MAIN-3: photo candidate projection loads no complete File/Image entities.
  With seven candidates and limit 1, entity-load listeners observe only the final
  File/Image. Warm document-index listeners observe exactly three selected Chunk
  entities from six candidates, not all chunks of the winning files. Cold TF-IDF
  construction still reads corpus text/metadata; that is a separate feature cost.
  Empty results invoke no final store; deleted supporting chunks and newly
  disabled sources raise HydrationError. Old ID-only stores remain supported;
  choosing both store interfaces or omitting required records fails explicitly.
- MAIN-1/6: acceptance covers more-than-window year/person/location distractors,
  screenshot scope, disabled sources, encoder-revision drift, OCR without vectors,
  malformed/non-dictionary legacy context, committed document replacement and
  uncommitted changes followed by rollback. Document-only scope rejects any
  attempt to construct an image encoder. A real single-candidate chat request
  exposed and now covers the unit-cap/window boundary.
- MAIN-5: frontend uses V2 and renders its global order. A focused Node handler/
  React static-render check exercises blank/trimmed submission, all/document
  scope, intentionally empty legacy arrays with valid mixed results, citations,
  thumbnails, pending controls, scope clearing, empty results and HTTP errors.
  It mocks hooks and HTTP; this is not a browser or real-network acceptance run.

## Executed checks

Use the existing `.venv/bin/python`; no interpreter or dependency changes.
ASGI/AnyIO stalled under the filesystem sandbox during an initial check (timeout
exit 124). Subsequent pytest commands ran with normal local execution permissions
and completed; the timeout is not counted as a passed check.

| Check | Actual result |
| --- | --- |
| Focused product/consistency/stages pytest | 51 passed in 4.13 s |
| Full `.venv/bin/python -m pytest -q` | 359 passed in 46.08 s; 5 SWIG deprecation warnings |
| Final product pytest after adding exact warm Chunk-load assertions, typing and diagnostics | 15 passed in 3.04 s; same 5 warnings |
| `npm --prefix frontend run build` | TypeScript and Vite passed; 1619 modules |
| `node frontend/scripts/check-search.mjs` | Passed; mocked hooks/HTTP, actual client/transpilation/React static rendering |
| `.venv/bin/python scripts/check_workspace.py --base HEAD` | Passed; 415 source files |
| `ruff check backend scripts tests apps examples services labs learning` | Passed |
| `ruff format --check scripts tests apps examples services labs learning` | Passed; 64 files |
| Touched backend modules/tests: Ruff E9/F/I and format check | Passed; 8 files formatted |
| `.venv/bin/python -m pip check` | No broken requirements |
| `git diff --check` | Passed |

Full pytest used `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`OPENBLAS_NUM_THREADS=2`, `OMP_NUM_THREADS=2`, an existing benchmark temporary
directory and a 180-second process deadline. It ran CPU/mock regression controls,
not a native video-model quality/corpus benchmark. The final product-only check
followed the last assertion/diagnostic changes instead of repeating unchanged
full-workspace tests.

## Limits

This establishes source integration, API compatibility within the documented
changes, deterministic fixture behavior and buildability. It provides no new
Recall/nDCG result, cross-modality weight calibration, private-library acceptance,
browser session, production latency, deployment or running-service proof.
RRF weights remain the explicitly documented initial integration policy. Fixed
candidate windows can reduce the unique-file count; original media are not read
by final SQL hydration. Model reranking, Tantivy/FAISS, snapshot/cache activation
and process timeout/cancellation remain separately selectable capabilities and
are not enabled on this product path.
