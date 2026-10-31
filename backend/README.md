# Lentica Backend

FastAPI, SQLite metadata/queue/vector baseline, document parsing, and local model
adapters. See the [root quick start](../README.md), [configuration](../docs/configuration.md),
and [workspace spec](../specs/0001-agentic-workspace/README.md).

Index/task consistency and compatible retrieval are specified in
[0002](../specs/0002-index-task-consistency/README.md). Scan cursors, task ownership
and vector scoring have separate adapters; the product remains a local monolith.

The independent [hybrid search layer](app/retrieval/hybrid/README.md) adds typed
retriever/fusion/store interfaces and an explicit CPU evaluation command. See
[spec 0003](../specs/0003-hybrid-search-layer/README.md) for contracts and evidence.
The opt-in [0004 optimization workflow](../specs/0004-hybrid-search-optimization/README.md)
adds sparse adapters, isolated development/validation and locked test evaluation.
The [0005 module refactor](../specs/0005-hybrid-module-refactor/README.md) applies
the root Google-derived Python conventions and separates reusable
[evaluation modules](app/evaluation/hybrid_eval/README.md) from command-line IO.
The opt-in [0006 quality/multimodal stages](../specs/0006-multimodal-retrieval-quality/README.md)
add resource-linked units, bounded reranking and candidate-loss diagnostics.
The first slice uses actual local image pixels/native PDF text and explicit
synthetic annotations; pretrained multimodal models and product integration
have separate acceptance evidence. The opt-in
[0007 execution layer](../specs/0007-retrieval-execution/README.md) now adds
capability/family routing, image input, immutable SQLite snapshots, Tantivy/FAISS
adapters, native CPU BGE/CLIP/SigLIP, mixed PDF OCR/native ruled-table cells and
bounded graph-verified Whisper transcription, CLAP sound semantics and explicit
subtitle/frame temporal conjunction. Three disjoint-source image trials
and real multimodal reranking show both gains and failed cost/validation gates;
fixed five-class sound gates pass twice; reference-scored speech remains a small
English read-speech baseline. VideoMAE now runs actual16-frame public-clip controls
and exact-class unit retrieval; broader action/language/layout quality remains
on the [execution checklist](../specs/0007-retrieval-execution/todo.md).

[0009 product integration](../specs/0009-product-hybrid-integration/README.md)
connects `/api/v2/search`, legacy photo search and document chat to the same
`asset_search` service and `StagedSearch -> HybridSearch`. Unfused photo branches
and the existing revision-cached document TF-IDF route share rank fusion, file
aggregation, one total file limit and final SQL hydration. V2 `results` defines
global order; `hits` and `document_hits` project that final selection. Photo-only
and chat calls keep explicit modality scopes. Candidate/index reads remain
separate from final selected-file/chunk reads; a cold document index still reads
its corpus. Native indexes, model rerankers, snapshots, candidate caches and
process cancellation are not activated by this integration. No reindex is needed.

From the root: `python -m pytest backend/tests`. Start with the selected Python:
`python -m uvicorn backend.app.main:app --env-file .env --host 127.0.0.1 --port 8000`.


Spec0007 adds optional source-isolated fast/slow image-gate and Chinese-query
adaptation experiments. Use `python -m backend.app.evaluation.cascade_benchmark
--help` and `python -m backend.app.evaluation.translation_benchmark --help` for
explicit local paths. Imports/help load no models. Native Marian translation
requires the optional `retrieval-query-adaptation` extra; it preserves original
queries/provenance and fails on exhausted budgets. These are explicit experiments,
not a new default search route. See0007 evaluation/verification for actual results.


Optional page-region ordering is evaluated by
`python -m backend.app.evaluation.layout_benchmark --help`. It verifies an explicit
public PDF and manual anchor annotations; the old extraction APIs/defaults remain.
See0007 LAYOUT-2/3 for original failures, visual annotation corrections and the
limited same-document heading-order check. No semantic layout model is implied.


Optional native unruled table regions use
`python -m backend.app.evaluation.unruled_benchmark --help`. Explicit regions,
cell/glyph provenance and budget failures preserve existing PDF defaults. The
survey control records real text/structure/location plus final-read behavior;
independent-document and image/merged-table quality remain on0007 TODO.

CASCADE-4 adds explicit `--policy calibrated-rescue`: fixed development grid,
fresh per-candidate queries and one validation pair. Ninth-source development
found a feasible pair, but validation FPR5/80 exceeded the.05 budget; it remains
rejected even though the frozen test passed its own gates. See0007 for original
failures and separate selection/test results; no product default changes.

Optional explicit numeric-table row contexts use
`python -m backend.app.evaluation.table_benchmark --help`: caller field schema,
header count, native baseline subrows and marked geometric merged conditions.
The independent OTSL paper control scores64 fields and8 typed conditional
answers with locations/real final PDF reads. It does not infer free-language
questions or automatic header meaning; old PDF extraction defaults remain.

Optional genuine joint image/text matching uses
`python -m backend.app.evaluation.relation_benchmark --help`. The separate
`retrieval-relation-models` extra supports verified local native BLIP ITM with
restricted weights-only loading; the bounded reranker preserves the existing
candidate/filter/final-read contracts. Initial384 SugarCrepe controls improve
relation preferences but fail CPU cost. A development-only native-resolution grid
freezes256 before fresh120-source acceptance: all cost gates pass, validation gain
is.025 below the.05 requirement, so promotion stays rejected. Both batches repeat
exactly. See0007 RELATION-1–4 for identities, original failures and metric limits.

The opt-in [0008 video quality protocol](../specs/0008-video-temporal-quality/README.md)
adds frozen public multi-action sources, blind16-frame proposals and explicit
same-video ordered evidence. `python -m backend.app.evaluation.video_benchmark
--help` separates source preparation, window decoding, native VideoMAE/single-frame
SigLIP evaluation and fixed-gate comparison. Labels only score outputs; model
features are reused explicitly, and final selected originals are actually read.
This fixed-class control does not certify free-text temporal reasoning or general
video quality. See0008 verification/evaluation for the current evidence stage.
