# Hybrid Evaluation Library

Reusable local evaluation code extracted from the original command modules by
[spec 0005](../../../../specs/0005-hybrid-module-refactor/README.md). Follow the
root Google-derived Python engineering rules; optional numeric imports remain
lazy. Importing these modules does not read data, construct models or import CLI
entrypoints. Experiments use explicitly selected local fixtures/public data.

| Module | Responsibility |
| --- | --- |
| `dataset.py` | Bounded BEIR/fixture loading, validation and stable data identities |
| `encoders.py` | Explicit corpus-fitted LSA construction and query transformation |
| `reporting.py` | Canonical hashes, environment/timing summaries, fetch counts and paired statistics |
| `execution.py` | Named route builders, candidate caches, ranking and live-pipeline checks |
| `workflow.py` | Leakage-aware partitions, development/validation selection, lock checks and frozen evaluation |
| `benchmark.py` | Original eleven-configuration benchmark and example formatting |
| `diagnostics.py` | Per-route/union coverage, exclusive relevance and loss attribution |
| `quality.py` | Locked public-corpus stage/rule controls and actual-pipeline parity |
| `media_fixture.py` | Fresh generated images/PDFs with explicit synthetic annotations |
| `multimodal.py` | Fixed multimodal/resource-grouping controls and location metrics |

`workflow` composes `dataset`, `execution` and `reporting`; execution constructs
retrievers behind the standard-library-only `retrieval.hybrid` contracts. New
retriever kinds add focused builders; the existing experiment matrix stays fixed.
Document/store, lexical and dense-vector implementations have separate modules.

The existing `backend.app.evaluation.hybrid_benchmark` and `hybrid_optimize` modules
parse arguments and write files, with explicit compatibility re-exports. Their
CLI flags, report shapes, selection schema/checksums, encoder IDs and rankings
remain unchanged. Existing locked selection files can be evaluated directly.

Run `python -m pytest backend/tests` and the root workspace/Ruff checks. This
refactor's acceptance compares deterministic outputs, not elapsed times; current
checks do not constitute full Pylint or static type-checker certification.

The opt-in `hybrid_diagnostics` and `multimodal_benchmark` CLIs follow spec 0006.
They do not modify the old search space or selection. Public-corpus diagnostics
are fixed historical comparisons; generated image/PDF scores validate mechanisms,
not unseen or pretrained model quality. Timings explicitly distinguish cached
ranks, actual route execution, candidate features and final asset reads.

Spec 0007 adds `partitions` (source-connected isolation), `localization` (separate
positive/no-answer and spatial/temporal metrics), `parquet_io` (checksummed public
data round trips), `routed_controls` (fixed generated capability/family controls)
and `native_quality` (fixed real local BGE controls with a verified corpus cache).
The new `routed_benchmark` and `native_benchmark` modules only own CLI/IO. Native
quality computes actual queries and checks real baseline/three-route pipelines;
the corpus cache is not counted as fresh corpus encoding on every run.

`index_quality` and `index_benchmark` compare persistent HNSW with exact retrieval
inside the same eligible ID domain, measure post-filter candidate loss, and
compare four independent native text indexes with whole-corpus statistics.
Reference BM25 also verifies injected aggregate N/df/length against a whole
reference index; this does not claim native Tantivy global-statistics injection.

`image_quality` and `mirror_images` prepare bounded real public pixels, exclude
prior source/exact-pixel groups and freeze development/validation selections
before test. `vision_quality` compares fixed CLIP/SigLIP controls and separately
measures actual bounded text/image reranking versus reused-feature diagnostics.
The source/selection/model identities and failed quality/cost gates are retained.
`weight_slices` checks original Top10/qrels and diagnoses fixed historical weights
on label-dependent post-hoc groups; it is not a deployable routing classifier.

`sound_source` verifies the exact predeclared ESC50 source manifest and writes
anonymous explicitly resampled clips. `sound_quality` owns fixed sound/text
retrieval, development-only margin selection and the frozen test quality/cost
gates. `sound_benchmark` only parses paths and writes fresh artifacts; importing
or requesting help never downloads data or loads a model. Native CLAP extraction
and unpadded single-query text inference live in the optional `clap_onnx` adapter.

`speech_source` reads only embedded audio bytes from an explicit bounded Parquet
and prepares the fixed ID-sorted slice. `speech_quality` runs waveform-only
transcription and computes reference-only word error, retaining failed samples.
The `speech_benchmark` CLI keeps decoding/preparation separate from evaluation.
Its optional `retrieval-speech-eval` dependencies do not enter default installs.

`vision_runtime` profiles only the declared development samples and freezes a
runtime identity before model-quality selection. `language_quality` measures all
30 fixed English/original/paraphrase/Chinese class prompts without choosing a
test winner; its class qrels do not imply arbitrary multilingual understanding.
`action_quality` owns public source-named single-clip temporal-model controls and
unit/resource composition. `action_benchmark` only constructs explicit adapters
and handles CLI/output. Reversed/static inputs are explicitly diagnostic, not
real accuracy samples.


`cascade_quality` records actual selective slow-model calls, frozen development/
validation decisions and a full SigLIP reference. `translation_quality` compares
all ten fixed Chinese queries, actual native Marian translations and English
controls on sixth-source test images. Translation errors keep empty rankings in
the denominator. Each command constructs explicit local adapters and writes a
fresh report; optional numeric/model imports remain lazy.


`rescue_quality` adds the predeclared CASCADE-2 trial: uncertainty rescue, separately
built SigLIP gallery, actual slow-space query ranking and full-model quality/cost
comparison. Selection checks nDCG loss as well as hit/FPR/cost; validation tests one
candidate only and cannot repair development failure. The cascade CLI exposes
`--policy rescue-slow`; its default remains the original strict-fast experiment.

The same cascade CLI accepts explicit `--fast-negative-margin 0.005` only with
`--policy rescue-slow`. This fixed eighth-source CASCADE-3 protocol keeps all
quality/cost thresholds; it does not select a tolerance on held-out data. Frozen
selection identity rejects mismatched tolerance before test inference.


`layout_quality` compares geometric y/x with explicit XY-cut ordering against
frozen manual anchors. Missing anchors remain in all-pair denominators. The
`layout_benchmark` CLI verifies public source SHA before native extraction and
records source/annotation/config identities. Original failed18-anchor labels are
retained; two visual transcription corrections and a narrower column-gap trial
are reported separately. Six independently authored headings on another page of
the same paper provide a narrow check, not independent-document generalization.


`unruled_quality` compares native line/text controls with the explicit-region
adapter, scores every frozen reference slot and separately reports nonempty text,
opaque glyphs and localization. It composes actual cell retrieval with inherited
filters and one real final PDF read per query; wrong-filter/glyph queries read
nothing. The `unruled_benchmark` CLI verifies source SHA and reads only explicit
bounded references, keeping reporting IO outside the reusable library. A known
survey region is development evidence, not unseen/general table accuracy.


`calibrated_rescue` exposes the explicit `--policy calibrated-rescue` experiment.
Four fast tolerances are measured on development with fresh query inference for
each. `PreparedGallery` shares only a read-only, source/partition/model/ID-bound
gallery and a separate full-reference ranking report. Validation checks one pair
and test cannot repair failed selection. A dedicated selection schema locks the
original quality/cost gates; the old strict/fixed policies remain compatible.

`table_quality` scores all frozen expanded fields and explicit typed query/answer
bindings on a public numeric table. Answers/qrels never enter extraction; caller
field schema/header count is explicit. It records native coarse values, marked
geometric spans, answer bbox IoU and actual final source reads. `pdf_controls`
provides the shared16MiB/SHA-verified final read used by both table controls.
The `table_benchmark` CLI owns bounded reference/report IO and local source paths.
A single independent development document does not certify general table QA.

`relation_source` freezes numeric-ID/source-grouped SugarCrepe annotations before
image availability, copies only explicit bounded original JPEGs, rejects duplicate
pixels and verifies source/byte/selection/counterbalancing identities. Captions
are competing text inputs and reference labels, never image features or a caption
index. `relation_quality` runs fresh dual or joint inference for every case; ties
and provider failures stay in the denominator. Comparison rechecks per-case scores
and recomputes summaries, then applies the predeclared per-partition quality/cost
gates and paired diagnostics. The CLI owns optional model construction and bounded
report IO. These fixed-source caption preferences do not measure cross-image Recall,
open-world rejection or serving latency; failed cost gates remain failed.
Fresh iteration re-verifies an explicit prior snapshot before source selection;
source/pixel exclusions are stored in both frozen selection and data manifest.
Reload checks exclusions without reopening other directories. Smaller native BLIP
inputs are explicit, separately identified numerical profiles; development grids
choose at most one candidate before any new-source acceptance or held-out use.

`video_source` verifies a pre-acquisition public16-subject reference selection and
anonymous original videos. `video_window_io` decodes only those sources into
bounded verified frames and reproducible blind proposals. `video_quality` composes
caller-owned native VideoMAE or SigLIP inference with pre-window filtered Top40,
same-resource ordered evidence and actual one-batch final original reads.
`video_metrics` owns independent union-coverage, unit/resource ranking, raw/clipped
IoU, per-step localization and no-answer denominators; human-interval oracle joins
remain separately marked. Provider failures never remove difficult windows or
queries. The `video_benchmark` CLI alone constructs models and owns report IO;
all modules/help remain safe without numerical/video/model packages. Fixed-class
external references are a bounded quality control, not open-world/serving proof.
