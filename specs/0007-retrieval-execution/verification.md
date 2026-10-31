# Verification

Date: 2026-10-31. Spec, ownership/decision and checklist preceded implementation.
Status remains **active**; the checklist explicitly retains incomplete work.

Latest checkpoint:267 tests pass, workspace361 sources, full required/scoped
Ruff and format/pip/diff checks pass. Build7 and extracted-wheel acceptance
cover the current action_quality separation. Sound five-class gates pass twice;
reference speech WER11.6883% is a12-clip one-speaker baseline. Four image trials
retain failed validation/cost; Chinese prompt coverage is measured but weak.
VideoMAE actual temporal inference/unit retrieval works on one source-named
public example; independent action/time-order quality remains unverified.
Detailed historical checkpoints below keep their original scope and failures.

## Implemented and actually checked

- QUERY-1/FAMILY-1: Additive MultimodalQuery, capability/index-state planning,
  fair bounded branch windows, text/image/combined binding and family-capped RRF.
  Disabled/unsupported/missing-modality/not-indexed/budget states stay distinct.
  Image vectors require an aligned encoder identity. Character/pixel/byte bounds
  are checked before provider execution. Original RRF is used for empty families.
- EVAL-2: Source-connected partitions and duplicate-query grouping; independent
  resource/unit recall, page/bbox/time overlap and no-answer denominators. Two
  84-query generated-media runs have identical per-query rankings, metrics and
  partition digests. The 12 difficult no-answer queries still all false-positive;
  no result rejection threshold or generalization is claimed.
- SNAPSHOT-1 baseline: Immutable SQLite units/final JSON records/dense vectors,
  checked manifests, pinned readers, copy-on-write additions/deletions, vector
  invalidation, atomic CURRENT activation and rollback. Tests inject corruption
  and pointer-replace failure and retain old readers/references. Text cache keys
  reject multimodal requests; multimodal keys include image bytes, filters,
  stage/routing/budget semantics and all snapshot/source/model/config versions.
- SCALE-1 partial: Actual Tantivy 0.22.0 and FAISS CPU 1.8.0.post1 artifacts build
  and reopen. Tantivy uses an I64 field and zero-score ID conjunction; a targeted
  probe caught this binding's unsigned/I64 mismatch before implementation. HNSW
  uses real IDSelectorBatch row IDs plus exact selective/tie fallbacks. Four
  acceptance tests exercise 100 text documents and 512 24-dimensional vectors,
  filtering/reload/zero queries/ties/identity/corruption. This is not public-scale
  ANN quality or a million-vector performance result.
- MODEL-2 partial: Verified explicit local BGE safetensors and CLIP quantized ONNX
  artifact bundles from selected public mirrors. Every selected file matched its
  published size/SHA-256. Mirrors provide identity, not an upstream signature.
  Native BGE CLS/prefix semantics were checked against pooling config/model card;
  old product mean-pooling space stays unchanged. CPU BGE text and CPU CLIP text/
  image inference ran. Actual packaged RapidOCR recognized generated text with
  regions/confidence/provenance; native rotated PDF blocks preserve page bounds.
  CLIP shape checks are not a real-image retrieval score. Complex page OCR/tables,
  text/multimodal model rerankers and audio/video extraction remain pending.
- Native BGE corpus encoding: all 5183 SciFact documents, 384 dimensions, about
  945.08 seconds CPU. Two fixed five-control runs reuse a checksum-bound corpus
  cache, probe endpoints against the actual encoder and actually encode queries.
  All five 300-query rankings/metrics/paired statistics match across runs. Old
  BM25/selected rankings also match spec-0005 for all 300 queries. Actual baseline
  and three-route pipelines match cached ranks and each fetch 300 batches/3000
  final records, at most 10 per call. Numbers, regressions, timing and exposure
  limitations are in evaluation.md; no test-based winner/default selection.
- Public Parquet round trip: 5183 documents, 300 queries, 339 binary positive
  pairs. Corpus/query IDs, source identity, text/metadata and judgments preserved.
  Checksums and compressed/expanded byte/row bounds are enforced on reload.
- EXEC-1 partial: Nine actual spawn/deadline/reaping/failure checks, bounded
  parallelism, cancellation of other workers on strict failure, queued request
  deadlines and fail-closed filter/ID violations even in partial mode. Factories
  initialize inside POSIX workers; cold startup counts toward their deadline.
  This currently bounds retriever processes, not arbitrary final store/fusion
  callbacks; there is no persistent warm pool or serving performance claim.

## Executed final checks

- Complete repository CPU/mock pytest: **209 passed**, 5 preexisting SWIG warnings,
  27.07 seconds, no skipped tests in this selected environment. Earlier checkpoint
  had 208 passed; the added multimodal-cache case is included in the final 209.
  Tests ran outside the sandbox for the previously confirmed AnyIO wakeup issue,
  with model network disabled and OPENBLAS/OMP threads limited to two.
- Full configured Ruff check, extra E9/F/I for touched hybrid/evaluation/tests,
  required workspace format check plus scoped backend formatting: passed.
- Workspace checker `--base HEAD`: passed, 303 source files at this checkpoint.
  `git diff --check`: passed. `pip check`: no broken requirements.
- Final sdist/wheel build completed in ignored build-0007-run-2. The wheel contains
  all new modules including cache_keys and no generated data/weights. Extracted
  wheel `python -S` executes both new CLI helps and actual snapshot build/open/
  activate -> BM25/family stages -> one final asset fetch -> cache-key smoke.
  No NumPy/model/image/database-application dependency was imported there.
  This is extracted-wheel validation, not a clean dependency installation.
  The first final-wheel pipeline subprocess exceeded its 10-second harness
  timeout. A bounded direct rerun in the extracted-wheel directory, with the
  imported module path explicitly checked, passed all stages. The cause of the
  isolated timeout is not established; a source-tree retry is not counted as
  wheel evidence.

The selected CPU dependency closure is version-recorded in cpu-constraints.txt
(44 distributions), not a wheel-hash or operating-system lock. Actual runtime:
Python 3.11.7, NumPy 1.26.4, SciPy 1.13.1, sklearn 1.5.2, Torch 2.6.0+cpu,
Transformers 4.57.6, ONNX Runtime 1.16.3, RapidOCR 1.3.24, Pillow 10.4.0,
PyMuPDF 1.24.14, Tantivy 0.22.0, FAISS package 1.8.0.post1, PyArrow 15.0.2.
Initial module discovery found no model/index/OCR packages; later live checks
established the above runtime. Historical absence is no longer a blocker.

## Execution/accounting and limits

Early sandbox network probes failed DNS. Outside the sandbox, system CA roots
fixed PyPI TLS; a selected public mirror provided compatible CPU wheels. The slow
owned pip dry-run was explicitly stopped using verified command/start identity;
its session exited 143. Installation used the selected .venv only, no system
packages. An attempted pidfd cancellation API was unavailable in this Python;
the verified PID/start fallback succeeded. No shared process was terminated.
All investigation/model-build/evaluation/test sessions from this checkpoint have
completed or were explicitly accounted for; no background recursive search left.

Model/corpus/media/reports/wheels remain under ignored data/hybrid-benchmark/.
Original public archive, old selection lock and original TODO.md stay read-only;
unrelated dirty files remain intact. No private-library reindex, application
default switch, frontend change, GPU/cloud run, service operation, commit or push.
No independent real multimodal holdout, pretrained reranking, audio/video, public
ANN scale, full-request cancellation or full Google/Pylint/type-checker gate is
claimed. Continue the active todo.md rather than treating this checkpoint as
completion of the user's continuous task.

## Continuation: whole-request cancellation, index scale, bundles and cache

This checkpoint supersedes the earlier limitations about final store deadlines,
public-corpus ANN measurements, scanned-page OCR and index bundles. It does not
close the remaining multimodal/holdout/audio/video/warm-pool work.

- EXEC-1: **15 actual process tests passed**. The final plugin factory, fusion
  and hydration now run in a disposable worker sharing the request deadline.
  Three separate 10-second stalls (construction/fusion/fetch) actually terminate
  and reap the final worker under a 1.5-second request budget. Partial retrieval
  never masks final failures, omitted records or oversized final responses.
  Final records cross the boundary as JSON, with an 8 MiB bound. ProviderSpec
  factories replace parent plugin instances only in the new ProcessSearch API;
  established synchronous HybridSearch remains unchanged. Coordination/cleanup
  add overhead and do not constitute a hard realtime guarantee.
- SCALE-1: two independent native index builds and actual CPU query encodings on
  5183 SciFact documents/300 queries. All ranks and quality metrics agree; digest
  `a738daf54d9e8c3b261a89fbbbdfe23c68a27962408bfacb5af31ed12a6fff35`.
  Same-domain HNSW index recall@10: .999333 (5183), .997 (1296), 1 (52, exact
  fallback). Zero filter violations, all reopened rankings agree. Relevant-query
  denominators are 300/71/5; excluded/no-positive cases are reported separately.
  Post-filtering ten whole-domain candidates loses recall in both narrowed
  domains. Build/reload/bytes and measured retrieval latency are in evaluation.md.
- Four real native text shards vs whole-corpus Tantivy: 295/300 text Top10s and
  253/300 fixed hybrid Top10s differ. RRF is not immune to the rank perturbation.
  Separate reference BM25 aggregation of N/df/length matches the unsharded
  reference Top50 for all 300 queries. It is not native Tantivy stats injection.
  Existing old BM25 implementation tests still pass after adding opt-in shared
  statistics; 34 retrieval/optimization/persistence checks passed.
- Native/scan PDF: **6 extraction checks passed**, including real OCR on a
  rotated image-only PDF page, native-page preservation, inherited metadata,
  source SHA-256 unchanged and pre-raster/combined-region budget rejection.
  Rendering uses a temporary directory cleaned by context managers. Mixed-page
  image text, tables and reading-order semantics remain outstanding.
- Bundles: **3 checks passed**, using actual Tantivy/HNSW files. Self-contained
  feature/text/vector versions validate source/model/ID/filter domains and every
  inventoried file hash. Atomic switch/rollback, old pinned readers, corrupted
  segment refusal, mixed-source/model refusal, injected pointer replace failure
  and temporary-pointer cleanup pass. Wrapper reopen works under python -S
  without importing native/numeric dependencies.
- Candidate cache: **4 checks passed** for persisted reopen, non-sliding TTL,
  transactional LRU/count/payload-byte eviction, oversized-entry behavior,
  namespace/filter/window isolation, cached-domain revalidation and noncaching
  of provider failures/filter violations. Only candidate IDs are cached.
- Complete regression after these changes: **225 passed**, 5 existing SWIG
  warnings, 40.31s, no skip. Full Ruff, additional scoped E9/F/I, required and
  scoped formatting checks and diff check passed. This count precedes the pair
  model additions below; their subsequent checks are recorded separately.

Real pair model preparation: selected ModelScope Xenova/ms-marco-MiniLM-L-6-v2
quantized ONNX (23,143,499 bytes) and native tokenizer/config, every file verified
against published size/SHA-256. Actual local CPU graph uses input_ids,
attention_mask and token_type_ids; two explicit sentence pairs returned finite
logits 8.448820/-11.035024. Version
`native-pair-onnx-v1:b7df7823c39d4a5434937628c3fddc84c1409baba952913d1862815c8605e0b7`.
This is inference/contract evidence, not corpus quality; the fixed 300-query
experiment is still running at this checkpoint. Logits are not refusal probabilities.

Public image preparation: the explicit official CIFAR-10 binary archive download
hit its 180-second deadline (exit124); the partial file is unverified and unused,
with a separate incomplete-download record. Selected raw-GitHub/HF mirror probes
failed, and the selected ModelScope CIFAR dataset URL was absent. No image quality
claim or independent multimodal holdout result is inferred from these probes.

Dual-route native filtering control completed: actual Tantivy/HNSW, all300
queries encoded once, both retrieval/fusion repeats equal. Positive-query Recall
for mod4 (71 positive) is .882629 prefilter / .849765 route50-postfilter /
.772300 fused10-postfilter; mod100 (5 positive) is 1/1/.8. Full results and output
lengths are in evaluation.md; no production-category or unseen-generalization claim.

The first pair-quality run was explicitly stopped after50 queries to add durable
progress saving; verified evaluator/timeout PID identity before signalling only
its owned timeout parent, session exited143. No quality report was produced and
these discarded inferences are not counted in the resumed run. New opt-in pair
checkpoint requires corpus/cases/model/encoder/config identities and per-row
score digests, append/fsync, bounded rows/bytes, no overwrite; corruption and
identity/count mismatch test passed. The replacement has a 2400-second job
deadline and per-query checkpoints; it remains an actual running experiment,
not a completed quality result at this checkpoint.

## Continuation: completed pair, warm workers and preserved image failures

The pair run described as running above has completed: 7972 unique actual pairs,
300 queries, 798.973 seconds. BM25 nDCG .662213 -> .680439; three-route nDCG
.709467 -> .703905. Both paired 95% intervals include zero. Replay reused all7972
verified checkpoint pairs, did zero new inference and exactly matched rankings.
No default switch or stable improvement claim. Details are in evaluation.md.

19 cold/warm execution checks passed. Actual BGE/Tantivy/HNSW warm control reused
route PIDs and matched all10 cold rankings; first request6562.114ms, subsequent
nine median176.272ms. Context exit reaped workers. No serving/throughput claim.

Full CPU/mock checkpoint:237 passed,5 existing SWIG warnings,39.41s,no skips.
This precedes the later margin-v2/exclusion changes; subsequent focused source/
image protocol checks:5 passed. Workspace327 sources; Ruff/format/diff passed.
Build3 and extracted-wheel python -S CLI/snapshot/bundle/cache/warm-spawn smoke
passed before margin-v2, so they are not final packaging proof for later changes.

Real image trial1 tested200 sources twice: exact repeated rankings/metrics but
refusal FPR9/80=.1125 failed the .1 limit. Trial2 uses600 source/pixel-disjoint
images; dev found no acceptable margin, validation failed, frozen fallback test
FPR8/80=.1 but positive hit119/120 ->105/120 loses11.667pp and fails. Actual image
encoding10.231s passes the predeclared .2s/image batch-only budget. Prior failures,
locks and reports remain unchanged; neither trial demonstrates general refusal.
The trial2 evaluator session46825 exited0; a successful evaluator exit does not
mean the quality gate passed. Its ranking digest and all metrics are recorded
in evaluation.md. The progress message initially quoted98.3% raw hit; the actual
artifact is119/120=99.167%, corrected here and in the evaluation record.

Third trial design is recorded before development/test: separate SigLIP dual
encoder, new source/pixel-excluded600 images, same margin protocol, fixed CLIP
controls and bounded text-image reranking. New model9 selected files verified
against mirror size/SHA (separate text111475220/vision99499129bytes); download
session46774 completed,7.05s. No combined graph download or new pip dependency.
New source preparation session56546 completed200/200/200; inference and quality
are separately pending, not inferred from downloads or preparation.

## Continuation: actual SigLIP, multimodal reranking and weight diagnostics

The above pending SigLIP inference/quality stages are now complete. Native
SentencePiece0.2.0 installed in the existing .venv after AutoTokenizer correctly
reported its absence; vision extra remains optional. CPU native tokenizer uses
EOS/max-length64,224x224 processor,768-dimensional pooler outputs; selected graph
input/output contracts verified. Model version and fresh source identity are in
evaluation.md. No generic tokenizer substitute, joint cross-attention or
calibrated probability claim.

Third dev/validation rejected margin .005, lock recorded before test. Two full
actual test executions (sessions10359/80521, both exit0) match all image/text/
reranked and refusal rankings and metrics. Image Recall .346842 CLIP -> .390526
SigLIP; ten text queries nDCG .965922 ->1 with actual bounded Top20 reranking.
Refusal test FPR1/80, positive hit114/120 passes quality-only bounds but batch
encoding .215034/.209944s per image fails .2s limit. Overall gate remains false,
and validation remains rejected. No default switch. Each test execution actually
encoded600 SigLIP images/30 texts including refusal/gallery/live reranking, plus
200 CLIP images/10 texts; the repeat is fresh inference, not checkpoint replay.
Batch-reused and live candidate ranks differ for all10 prompts including Top10,
although their class metrics match. Live ranks are authoritative; no assertion
of batch/cache numeric equivalence. Full costs/limitations are in evaluation.md.

16 focused checks passed (7.44s) for image selection identity/no-test selection,
model-specific naming, image reranker path snapshot/filter/budget/tie/full order,
historical-weight evidence consistency, existing actual OCR and safe imports.
The new vision CLI help also passed under python -S. Full regression/packaging
after these and the subsequent mixed-layout implementation is still pending.

P4 weight observation now explicitly covered: metrics recalculated from original
Top10/qrels, fixed controls verified, both existing historical reports yield
identical statistics.164 text-advantage/24 vector-advantage/112 other queries;
no new inference. Grouping is label-dependent and post-hoc, never online routing.

Mixed PDF layout: actual two-page native+scanned-image+2x2 ruled-table control,
one90-degree rotated page. Initial test exposed native table row/column reversal;
in-memory logical-orientation detection now restores page rotation and maps
bounds back. OCR is also performed in logical orientation. The original source
hash is unchanged. Retained table slots/row-major cell IDs, native/OCR provenance,
no duplicated exact header, inherited filters, raster/unit/cell budgets pass.
Actual OCR joins words to SCANNEDERROR429; word BM25 does not match SCANNED.
The existing character-fragment adapter correctly retrieves both page units with
the same filters; source/expected text were not rewritten to hide this failure.
This generated PDF uses actual OCR and native geometry, not independent complex
layout accuracy. Multicolumn/unruled-table semantics remain outstanding.

Whisper model: two selected native-weight mirrors returnedHTTPError; exact
Xenova/whisper-tiny.en mirror provides selected native configs/tokenizer and
quantized encoder10124913/decoder30459718bytes, all13 downloaded files verified.
CPU graph/native prefix/suppression/no-timestamp contracts checked; unused
attention outputs are not requested. The fast tokenizer does not map timestamp
special strings here, so the timestamp range comes from native no_timestamps ID,
not an unknown-token lookup. No new speech dependency beyond selected runtime.

Public PyTorch sample returned403, HF sample failedURL access. The explicit
Whisper micro-machines example was accessible but exceeded initial4MiB raw
budget; that partial remains unverified and unused. HEAD established5272390bytes;
a new complete download within8MiB succeeded, sourceSHA
37de21902b32aa2fc147ccbfdcc0566cc7061fffb2c0b10874f05147c0b9de0f.
Before inference, explicitly selected first8seconds, stereo mean and SciPy
resample_poly160/441 -> mono16k PCM16, derivativeSHA
1910b07726f844f84713e8df355900732b42fed6d3e49702749474fcec95f89f.
Source and derivative/acquisition record remain separate and unchanged. These
local hashes do not prove independent upstream source authenticity.

Actual public8s inference twice:3.080682/2.955233s, identical transcripts, model
load4.080563s. Version
whisper-onnx-v1:624196a030ea8b4d101850dba51e248d5a6e665ab27badce36903356a77bc5f2.
Report audio-0007-inference-1.json. Third fresh actual inference3.137124s fed
unit200/asset100, locator[0,8], real transcript lexical retrieval and one final
parent fetch. Wrong tenant returns0 and no extra fetch; source bytes unchanged.
Report audio-0007-pipeline-1.json. No reference transcript/WER, word timing,
speaker, acoustic-event, or broad speech retrieval quality claim.2 WAV/greedy
budget/suppression/EOS checks pass; truncation returns no partial transcript.

Final checkpoint for this slice: **247 passed**,5 existing SWIG warnings,
40.67s,no skips. Includes readonly executor factory-map protection so a changed
same-name configuration cannot leave a warm worker silently serving an old model.
Workspace checker with --base HEAD:339 source files. Full required Ruff,
additional touched E9/F/I, required format and touched backend format passed;
pip check reports no broken requirements; diff whitespace check passed. Existing
frontend work was untouched in this slice, so no new frontend-build claim.
sdist/wheel build4 completed after these changes. Extracted-wheel python -S
acceptance passed:127 wheel files, all new modules present, no test/model/vector
artifacts;7 CLI helps; actual snapshot/bundle/atomic reference/candidate SQLite
cache/warm spawn/reused PID/one final store/context reap. Imported module paths
were checked inside the extracted wheel. Session59176 exited0. This is not a
clean dependency installation. No remaining owned model/test/build/investigation
session at this checkpoint; subsequent work starts a new recorded slice.

## Next slice: subtitle/video temporal evidence and sound-model preparation

TEMPORAL-1 code added after the247-test/build4 checkpoint.4 focused checks pass:
actual local MJPG AVI8-frame decode/4-frame sampling with observed timestamps,
source/file digests, inherited filters, budget rejection before output creation;
SRT/WebVTT cue identities/markup/time errors/unsupported timestamp-map rejection;
same-resource/common-time joins, touching endpoints, missing branches, stable
group order, identity/filter and combination/result failures. Same-file red and
caption evidence at different times no longer form a conjunction. This is an
actual decoder plus generated-color reference, not an action model or real VFR
certification. New global regression/wheel verification is not yet claimed.

SOUND-1 design frozen before data/model quality: ESC50 five declared classes,
fold12/3/45, original-source cross-partition checks, CLAP native48kHz/10s CPU and
separate test false-positive/hit-loss/cost gates. Exact Xenova CLAP mirror files
downloaded/size-SHA verified,9 artifacts, audio34301667/text126603263bytes.
Native features(1,1,1001,64), separate audio/text graph contracts and actual512-d
finite outputs verified on an explicit zero-waveform control and one text.
That probe is graph/inference evidence, not sound accuracy.

ESC50 metadata explicit jsDelivr URL returned93742bytes/2000 rows, SHA
ca660da60191a97de289983a05821c9382d852a38a2ba8428980816b68cf6246.
200 chosen5s clips, source folds checked. The initial concurrent selected-file
acquisition hitHTTP403 and session55857 exited1 after its thread pool drained;
incomplete-download.json is retained. Partial files are not a completed dataset
or quality result; a bounded audit of exactly the200 declared paths follows.

Recovery session75052 exited124 at its explicit300s deadline. The next bounded
exact-file audit found147 complete5s/44100 mono PCM16 clips and53 missing/partial,
recorded in recovery-audit-2.json. Retrying only these53 into a fresh directory;
old failures and partials remain. No reduced/easier class/fold subset is used.

Five new sound controls pass (0.31s): python -S CLI help, WAV format/window and
truncated payload, independent unpadded text graph calls, no-test selection and
identity checks, and failed-validation promotion protection. Synthetic score
fixtures prove protocol only, not sound quality. Actual preparation/selection/
test quality remains pending until all200 declared sources are complete.

The exact53-file recovery finished: session31046 exit0,200 complete original
WAVs, no errors, old partials retained. source.prepare/load verified all raw
sizes/hashes/metadata/folds, resampled into fresh anonymous inputs and retained
conversion saturation counts.80/40/80 source-isolated partition counts and no
cross-partition exact derived-waveform duplicate. Dataset identity
 e48a10f7674bcdadb42e4aa7f6353b0abcb78e2bf4bc6be924f9de5f9814620d.
Selection session86551 exited0, accepted category-gate-only margin-1; lock saved
before test. Test23836 and fresh repeat43306 exited0. All rankings/quality match,
SHA64381b4ed392c2fec7da78d5230252cada43ef384cc2023fc90cb0caabddf153.
Test FPR0/32, positive hit48/48, batch cost.172804/.174651seconds per5s clip;
predeclared validation/test quality/cost gates pass in this fixed five-class task.
No default switch. Full scores, restricted vocabulary and unknown pretraining
source overlap are recorded in evaluation.md. No broad sound accuracy claim.

Checkpoint after temporal/CLAP/sound additions:256 tests passed,5 existing SWIG
warnings,43.15s,no skips. Workspace348 sources; required Ruff/format, touched
E9/F/I+backend format, pip check and diff whitespace checks pass. This precedes
subsequent SPEECH-1 additions; final new-code regression/build still required.

SPEECH-1 design fixed before inference: first12 ID-sorted public dummy LibriSpeech
clean validation records, no reference model input, word-edit micro-WER and all
failures retained. PyAV14.2.0 already installed; explicitly versioned as optional
speech-eval dependency, not a new default product dependency. Four focused
controls passed (0.37s): word edits/normalization, actual embedded WAV decode and
first12 ordering, waveform mutation rejection, provider failures retained in WER,
wrong sample-rate rejection and python -S help. Initial test exposed valid WAV
one-channel layout named '1 channels'; count actual channels instead of requiring
a layout label 'mono'. Stereo/rate constraints are retained, no test relaxation.
Public source HEAD/GET handshake/address failures retained; Range mirror confirms
9192059bytes but full acquisition/quality is not yet claimed.

SPEECH-1 source session19721 exit0: exact mirror Range GET obtained9192059bytes,
SHA4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603.
Prior source1 handshake failure and its incomplete record remain; it was not used.
Actual prepare exit0:73 original rows, fixed first12 source IDs, native mono16k
PCM16,130.47seconds; no resampling, reference model input or embedded-path access.
Dataset identity a3b09465d44fa93d4f705c12a49a6a8ef9c2095be9637c49c92254e29be9593b.
The first evaluate command used a nonexistent model directory and failed before
inference; no report/score was produced. Correct verified model directory then
used: sessions45817/84241 exit0, two fresh full inference runs,12/12 successful,
308 reference words/36 edit errors/WER.116883116883. Actual inference costs
23.378462/24.208605seconds. All per-item transcripts/error counts match,
SHA c86381492e84f9a0595fe9d79a935d3c6d1ba8aed98cf3e99ff8949ff172908b.
This is one-speaker read speech, not general ASR/Chinese/noisy/word-timing proof.

Complete post-SPEECH-1 regression:260 passed,5 existing SWIG warnings,43.72s,
no skips (session36448 exit0). Workspace352 source files, required and scoped
Ruff/format checks, pip check and diff whitespace checks passed. Build5 session
90811 exit0: sdist and wheel created;137 wheel files with new temporal/sound/
speech modules and no tests/weights/vector caches. Extracted-wheel runtime smoke
is still pending at this record; prior wheel4 cannot certify these new modules.

Extracted build5 wheel python -S acceptance now passed (session97455 exit0):
9 CLI helps, new optional modules import without NumPy/Torch/Transformers/ORT/
PyAV/PyArrow, actual subtitle parse and word edit, snapshot/bundle/atomic reference/
candidate SQLite cache/warm spawn and same-PID reuse/one final store/context reap.
Imported paths resolve to the extracted wheel. The first smoke launch used the
wrong working directory and exited127 before Python; rerun from the extracted
wheel completed. This is extracted-wheel behavior, not a clean dependency install.
All owned data/inference/test/build/smoke sessions in this slice have exited;
no background recursive investigation remains. Spec remains active for listed
image validation/cost, language/layout and temporal model quality gaps.

## Fourth-source image runtime and language slice (in progress)

VISION-4 design/metadata preceded runtime/quality execution. A new600-image
source/pixel-excluded dataset is prepared, identity
 af7811d436334fe1ee9e0bc28ae22205fe492dc3859fec7792e1c16c0f1a4ed2,
200 each development/validation/test; previous1800 sources/pixels excluded.
Runtime profile actually encoded the fixed20 development images twice per each
of3 declared configurations. Default threads2/batch8 remained fastest (8.573464s
total for40 encodings), versus threads1/batch1 (15.236542s) and threads2/batch4
(8.784773s). Within-config repeated vectors were bitwise consistent; different
batch configurations differ and retain distinct encoder IDs. No faster-runtime
claim. Runtime lock SHA396c35bdff16157c9845bd9d3600ca55407a98a678dc2ae48b2d83380db065a6.
New selection uses only this fresh dev/validation and retains the original margin
and cost protocol; not yet a completed new quality result.

Seven focused iteration/image checks pass (0.50s), including runtime dev-only
inference/fastest selection and lock-before-model identity, all30 fixed bilingual/
paraphrase prompts without test tuning, no failed-validation reversal, and
python -S CLI help. Fixtures only certify protocol. Added image selection_accepted
and eligible_for_promotion fields prevent a good test from reversing failed
validation, without changing the existing heldout_gate_passed field semantics.
Workspace356 sources and required Ruff/format/diff checks pass at this slice;
post-iteration full regression/build has not yet run.

Temporal model artifact discovery only: exact Xenova VideoMAE ModelScope listing
returned404, bounded same-model HF search returned403. No temporal model weights,
actual action inference or quality result is claimed from these failures.

Fourth image selection78779 exit0: no feasible development margin, accepted=false,
lock saved before test10065 exit0. Test FPR7/80/positive hit114/120, cost.209200s
per image exceeds.2, total gate and promotion eligibility false. Language37373
exit0 actually encoded200 images/30 frozen prompts: original/paraphrase/Chinese
Recall.480/.465/.290, failed Chinese deer/horse prompts retained. No test tuning,
no faster-runtime/multilingual success/default claim. Full results in evaluation.md.

ACTION-1: exact native MCG-NJU mirror exists after other two candidate names404;
3 declared files acquired by session11563 exit0, all published mirror sizes/SHA
verified. Safetensors346161616bytes SHA
065cfde56f97671da4d196e1a3a6d3cf9304dcad92eb671e3632a6c6cdd04f73.
Native CPU VideoMAE16-frame/224/400-class contracts match. Selected public
source1 clip1013655bytes, SHA252f63d13748f08acf56765c295506bfdb8bb73b822e93a33a57d73988814a71;
only source filename provides whole-clip class, no independent event boundaries.
First two runs87677/84279 exit0: new frame extraction and three actual temporal
model inferences each. Original/reverse/static expected class all rank1, exactly
same per-control logits across repeats. Original inference4.192991/4.133446s.
The unchanged Top1 under reversed/static input prevents an order-understanding
claim; changed logits alone do not establish correct temporal reasoning.

Third actual action run42114 exit0: original/reverse/static logits again match
first two. Added exact native-class ranked-ID adapter captures immutable aligned
original-only scores, validates timed unit/model/vocabulary identity, rejects
controls and unsupported queries, and filters before limiting. Actual post-model
unit1000/asset1 StagedSearch read1 final batch/row; wrong source filter returns0
with no extra fetch. Only just-computed class features were reused at query time,
not new per-query model inference. Parent source SHA unchanged.11 focused checks
pass (0.58s): original/time/control provenance, stable class ties, immutable scores,
finite values, filtered windows/allowed IDs and existing temporal/vision protocol.
New full regression and build6 were started after these changes; results pending.

Post-action/library checkpoint before final packaging:267 tests passed,5 existing
SWIG warnings,67.30s,no skips (74977 exit0). Workspace360 sources and required
Ruff/format/scoped checks, pip check, diff check passed. Build6 exit0,143 wheel
files/no model or test artifacts, extracted-wheel11 CLI helps and actual snapshot/
bundle/cache/warm/final-store lifecycle plus exact-action filtered-ID controls
passed (17879 exit0). This checkpoint preceded the following module extraction.

For the root Google-derived responsibility rule, reusable action experiment and
pipeline logic now lives in hybrid_eval.action_quality; CLI only constructs
explicit source/model adapters and owns reporting IO. Fourth fresh real run31489
exit0 after extraction: every per-control logit digest/rank and the final pipeline
record match run3 exactly. New library/CLI scoped Ruff/import format and workspace
361-source checks passed. Final post-extraction full regression/build7 follow;
previous package results are not reused to certify the extracted new module.

Final post-extraction checkpoint: full267 tests passed,5 existing SWIG warnings,
45.17s,no skips (33482 exit0). Workspace361 source files; full configured Ruff,
required formatting, touched backend formatting/E9/F/I, pip check and diff checks
passed. Build7 (39949 exit0) produced sdist/wheel144 files with action_quality and
all current modules, no tests/model/vector artifacts. Extracted-wheel7 python -S
smoke (73203 exit0) passed actual snapshot/bundle/atomic reference/candidate cache/
warm worker PID reuse/final store/context reap, action ranked-ID/filter and safe
optional imports,11 CLI helps. Paths resolve inside the wheel; this is not a
clean dependency install. All owned download/inference/test/build/smoke processes
for this slice have exited. No commit/push/default switch/private-library or
GPU/cloud/service operation occurred. Active checklist gaps remain real work.


CASCADE-1 fifth dataset preparation and selection39210 exited0. Actual held-out
runs84032/16016 exited0; rankings and actual slow-call IDs match exactly, each
200 fast/116 slow. Test cost .174606/.170300seconds/image, hit.9, FPR1/80;
selection rejected because development hit loss13.33pp exceeds10pp. Test-only
pass never overrides accepted=false. Full SigLIP comparison separately computed;
its better ranking and extra cost are retained in evaluation.md. No default switch.

LINGUAL-1 design precedes implementation; actual native graph/tokenizer probe
37285 exited0. Corrected checkpoint default beams from4 to6 before decoding.
Selected verified encoder/uncached decoder expose native attention-mask inputs;
only logits are requested, no past ABI assumed. Installed sacremoses0.1.1 in the
selected existing environment without upgrading existing dependencies. Optional
extra/CPU constraint updated. Native CPU smoke42480 exited0: 鹿 query generated
“A picture of a deer.” with natural EOS, no label/reference in input. This is a
single smoke, not accuracy. Sixth dataset93156 exited0,600 images excluding all
previous3000 source/pixel groups, identity
`d3e9c069b99c845c59e58c7cb3980c48a5481f25d5debee6517e8c77176de386`.


LINGUAL-1 held-out actual CPU runs38613/69069 exited0. Both reloaded local models,
generated all ten translations and encoded all200 images anew; translations,
rankings and quality match exactly. Original Chinese Recall.27/nDCG.557353;
actual translations Recall.495/nDCG.993375; English Recall.495/nDCG.993638.
No failed query; deer/horse recover0->.5 recall. Initial targeted15 checks passed
(.60s), covering EOS/pad/start, budgets/no partial output, provider failure,
provenance, denominator, test-only/shared image pass and stdlib help. Two initial
failures were test fixture/assertion mistakes (empty decode and nested metrics),
corrected without changing behavior/acceptance. Workspace369/full Ruff/formats/
pip/diff checks passed. Full CPU/mock282 passed,5 existing warnings,44.10s
(75913 exited0). This checkpoint precedes CASCADE-2; final post-change checks
are still required and will not reuse this result as certification.


CASCADE-2 design precedes implementation. Seventh600-image preparation59228
exited0, excluding all previous3600 source/exact-pixel identities; dataset hash
`3fc4cfc4ad5bd2a9a8ca16898c30b3729db92e8b49d5426431ab7afa177c12e8`.
Seventeen targeted checks4531 passed(.65s), including actual uncertain-negative
rescue calls, aligned slow features/zero-mask rejection, independent gallery/query/
reference passes, source lock before inference and no test-based acceptance reversal.
Actual development/validation selection55323 is running; no quality pass claimed.

Targeted recheck3221 passed17(.64s) after frozen rescue tolerance/property and
slow-feature dtype handling. Workspace370/full configured Ruff/formats pass.
Public PDF acquisition99890 ended404 for the exact old Docling example URL;
no PDF/manifest was created and no quality claim follows from this failed fetch.
A bounded exact arXiv report URL is the next source attempt.

CASCADE-2 selection55323 exited0: dev/validation quality at-1 satisfies predeclared
FPR/hit/nDCG loss gates, but actual query costs .206811/.202207 exceed.2s. Slow
calls150/147 of200; selection rejected, locked fallback-1. Test68986 is running
with that unchanged artifact. Latest18 focused checks84084 passed(.67s), including
nonfinite-score failure rather than an arbitrary plausible ranking.

Second exact public PDF attempt18024 hit its55s outer deadline(exit124), no
successful PDF/manifest. These two source failures do not block local image
experiments or certify complex layout. Acquisition remains pending.


CASCADE-2 actual held-out68986/41782 exited0. Both produced exactly matching
rankings, full-reference rankings and actual slow IDs(144/200). Test quality
passes, costs.203059/.203396 exceed.2 and eligible_for_promotion=false. CASCADE-3
.005 hypothesis/spec/code and eighth-source preparation were fixed before
seventh test output was read; it is motivated by dev/validation cost, not test
winner selection. Eighth600 source preparation88842 exited0, identity
`caff853c0badd0c1f79ef83bea0721f51781512bee12f0f39c7850889578d44d`.
Nineteen focused checks41814 passed(.73s), both explicit rescue tolerances and
parameter-mismatch rejection before pixel reads. Workspace370/Ruff/formats/
pip/diff checks passed. Final full regression/packaging still pending.

Bounded GitHub directory metadata found the previously moved example under
pdf/sources. Actual162131-byte public PDF acquired via verified blob API4086
exited0; SHA `a07f5c34601ba2c234d898cbfaa9e29a7045996ccd82ccab3012516220a1f3a4`.
Rendered/visually inspected: this is a ruled table with merged headers, not an
unruled-table or multi-column acceptance sample. A raw-file connection-reset
attempt created no source artifact; API retrieval verified the same expected blob.


CASCADE-3 development/validation17919 exited0. Actual cost.187690/.186657 passes
but development hit/nDCG losses exceed bounds, so accepted=false. Frozen test
27288 exited0, test-only quality/cost pass without reversing rejected selection;
second actual test57625 is running. No alternate source/tolerance was substituted.

LAYOUT-2/3 added opt-in stdlib column_ordering and reusable layout_quality; CLI
only verifies explicit source/annotations and owns reporting IO. Five focused
checks79927 passed(.19s) after one real header/footer control exposed middle-gap
interleaving. Added explicit wide-edge-band isolation without changing assertions.
First public page2 run retained as failure:16/18 anchors,81/153 pairs. Exact native
bbox and200dpi visual crops disproved preliminary cross-column-merge diagnosis;
actual problems were.03 gap above actual.024 and two mis-transcribed anchors.
Both corrected labels were frozen in a new file, original reference/report retained.
New.015 hypothesis uses page2 as development; same-paper page7 six-heading labels
were authored before candidate inference. Runs79927/53193 exited0; corrected-label
page2 baseline110/153 vs explicit.015 153/153, with annotation-only29-pair change
reported separately from algorithm43-pair gain. Page7 two real extractions/orderings
6/6,15/15 vs baseline12/15; complete rankings match.39 units, ~.00057s ordering;
same-document scope is explicit. Not unruled-table or general semantic accuracy.

Acquisition84773 exited0 also verified45KB public survey PDF and4.31MB paper via
expected GitHub blob/SHA; these are only acquired/inspected artifacts. No quality
claim follows for the survey or the source-named table sample. Static workspace
374-source/full configured Ruff/required formats/touched backend E9/F/I and diff
checks pass. Final full regression/build8/smoke remain pending after these changes.

CASCADE-3 repeat57625 exited0: query cost.183515, complete rankings/reference and
actual slow-call domain match first run exactly. All download/inference/preparation
processes owned so far have exited. Final full CPU/mock regression now running.

Final post-change full CPU/mock regression14155 exited0:291 tests passed,5 existing
SWIG warnings,45.62s,no skips. Workspace374 sources/full configured Ruff/required
format checks/touched backend E9/F/I, pip and diff checks passed. A check concurrent
with packaging counted transient sdist files(531); after build exit the same check
returned374, without changing source or ignoring a failure. Build8 session66603
exited0, produced sdist/wheel154 files with all new modules and no tests/model/data
payloads. Extracted-wheel8 python -S actual bundle/atomic ref/candidate cache/warm
worker PID reuse/final-store/context reap plus optional import/action/column smoke
70367 exited0. Paths resolve inside extracted wheel; this is not a clean dependency
installation. All20 extracted-wheel CLI helps17283 exited0 under python -S. All three acquired public PDFs remain byte-identical to acquisition SHA256. No commit/push/
private-library/default switch/GPU/cloud/service operation occurred. Remaining
checklist items and goal stay active; no claim that all TODOs are complete.


UNRULED-1 spec/reference frozen before implementation. Actual selected-source
runs1/2/3/4 exited0, with full table/glyph payload SHA stable. Nine real native
text cells and15 empty slots score24/24; mean cell IoU.702728, table IoU.922170;
21 opaque font glyphs never become searchable fake yy/* text. Final2 runs match
all seven query results, one actual parent PDF read per query, wrong-filter/glyph
controls0 extra reads. Unit hit@1=1, evidence bbox IoU.733560, source SHA unchanged.
This is an inspected development region, not independent-document accuracy.
Focused83832 passed10(5.20s),5 existing SWIG warnings: actual four-rotation
native cells/glyph isolation, budgets/empty/page failures, source rejection, safe
help and existing PDF/column checks. Additional composition check76998 passed11 total(5.12s),5 existing warnings; final suite/package checks required after following changes.


CASCADE-4 focused session56666 exited0:11 calibration/existing-cascade tests
passed(.66s). Ninth-source preparation21696 exited0:600 images,200 each partition,
excluding all4800 previous source IDs/exact pixels. Source identity8614062e...
Actual selection90845 exited0; four candidates execute fresh query inference,
partition/model/ID-bound read-only gallery/full-reference sharing only. Added
preparation-time model identity drift rejection; its new regression is pending.
Development chooses.01/-1; validation FPR5/80 exceeds.05 despite quality/cost
otherwise passing, so accepted=false. No validation replacement. Selection SHA
abe71c6d6154e89670a713a339ef16c0bc1c6e84b32626b7e439db1a5409cc61.
Frozen test15 now running; cannot reverse failed validation. Workspace380/full
configured Ruff/required format/diff static checks pass. Full regression/build9/
extracted-wheel9 remain pending; build8 evidence is historical for this code.

Frozen CASCADE-4 actual tests56433/15350 both exited0. Query cost.184878/.188058s;
130 actual slow queries out of200, FPR3/80, hit113/120, nDCG.735413. Full reference
nDCG.765328. Exact complete candidate/reference rankings and per-batch slow IDs
match; SHA7641f89f3367d1af6b1e1690b20b86946f7b2e8e2cf288644d57d0584e763702.
Test-only gates pass twice; rejected validation remains ineligible twice. No model
inference processes remain. Full regression98295 running; all static380-source/
configured Ruff/64 required formats/8 touched backend E9/F/I+format/pip/diff pass.

Final post-change full regression98295 exited0:300 tests passed in46.01s,5 existing
SWIG deprecation warnings,no skips; this includes preparation model-drift rejection
and actual unruled-table final-read composition. Build9 session82298 exited0:
sdist166 files and wheel158 files include all four new implementation modules;
no model/data/compiled-cache payloads, and wheel contains no tests. Extracted-wheel9
stdlib lifecycle smoke32944 exited0: bundle/ref/cache, repeated warm worker PID,
final store reads, context reaping, new unruled/calibrated imports without optional
runtimes. All21 extracted-wheel CLI helps45366 exited0 under python -S. This is an
extracted-distribution check, not a fresh dependency installation. Workspace380
sources, full configured Ruff, required64-file formatting, touched8 backend E9/F/I
and formatting, pip requirements and diff checks pass. All three selected public
PDF source hashes unchanged. All owned preparations/inference/tests/build/smoke
processes have exited. Current branch remains main; no commit/push/private-library
or default-model change, GPU/cloud/service operation. Remaining acceptance items
stay open and the continuous goal remains active.


TABLE-2 spec and visual8x8/8-query reference frozen before implementation/native
table detection. Explicit public OTSL source remains byte-identical. Actual native
6 coarse rows including2 headers become8 data rows;64/64 values,8 geometric-span
source cells,8/8 typed Top1 conditional row/answer bindings. Mean answer bbox
IoU.934450; one actual final source read/query, wrong-source/missing-field0 extra.
Two real extractions/evaluations exit0 and full structure/retrieval match; SHA
7f3fcc1f37bd292dbb5e94fda3ab6213478ba71abf5e9d19a83f478d3437c311.
Extraction.163779/.161521s excludes scoring/retrieval/final reads. This is one
independent-document development control, not unseen/general table quality.

Focused37528 exited0:11 tests passed in6.00s,5 existing SWIG warnings. Actual
four-rotation PDF controls verify merged condition provenance, exact filters before
window1, sparse-cell ambiguity and opaque-symbol rejection, word/cell/row budgets,
reserved metadata namespace, reference-only scoring and real source-read mutation
failure. Imports/CLI help pass under python -S. Shared final-read adapter also
retains complete real survey structure/retrieval compared with previous report;
actual survey5 exits0 and reads7 PDFs, glyph/wrong-filter0 extra. Final full suite,
workspace checks and packaging still required after these changes.

Final TABLE-2 full regression6237 exited0:305 tests passed in52.92s,5 existing
SWIG warnings,no skips. Workspace385 sources/full configured Ruff/required64-file
formatting/touched6 backend E9/F/I+format/pip/diff checks pass. Build10 session17868
exited0, sdist170 files and wheel162 files include all four new implementation
modules; no source data/models/cache payloads and wheel contains no tests.
Extracted-wheel10 stdlib smoke52084 exited0: new table/schema/evaluation imports
load no optional runtimes; actual bounded public-resource final read, original
bundle/ref/cache lifecycle, repeated warm worker PID, final store and context
reaping pass. All22 extracted-wheel CLI helps99673 exited0 under python -S. This
checks the extracted distribution, not a clean dependency installation. All three
explicit public PDFs retain acquisition SHA256. All owned investigation/test/
extraction/build/smoke processes have exited. No commit/push/default-model/private
library/GPU/cloud/service change. TABLE-2 scoped control is implemented/verified;
remaining image gates, independent-scale complex/image/unruled numeric tables,
relations/language and multi-action/ordering-video quality stay open, goal active.

RELATION-1 source IDs and reference captions were frozen before downloading120
unique COCO images or running models. Official SugarCrepe Git blobs verified;
actual bounded S3 acquisitions retained per-image bytes/SHA. Source selection,
anonymous original JPEG bytes and unique RGB-pixel groups bind dataset identity
c436e5b203c9152cc2eb7c233acfdb972ff425a6aebf94234b2db93b5d3c01fa.
ARO Drive and HF access failures remain recorded, not presented as executed data.
First development image transport/annotation inspection is disclosed; no selection
based on predictions. Captions are competing inputs/reference only, not image index
features. COCO-finetuned BLIP/pretraining overlap remains unknown.

Actual first CLIP55582/SigLIP31925/BLIP39849 runs exited0:76/81/102 correct out
of120, no ties/failures. BLIP per-partition accuracies.875/.85/.825 satisfy quality
gates, but costs.9283/.9366/.9324s exceed predeclared.5s; comparison-1 exits0 with
eligible_for_promotion=false. Native equivalence80183 and repeated evaluate probe
both max absolute error0 against full-pair native forward. Publisher895139697-byte
BLIP bin matches manifest SHA and is loaded only with explicit weights_only=True,
Torch2.6/Transformers4.57, local CPU native classes, no remote code/downloads.
Repeated BLIP78611 exits0:102/120,.949158s/case; every score and ranking exactly
matches first run. No cached/dual-encoder output passed off as joint scoring.

New model/evaluation/domain tests initially exposed a test-construction error:
SearchQuery rejects blank input before reranker invocation. Replaced that fixture
with a valid oversized query to exercise the intended reranker budget boundary.
Focused14668 exits0:17 tests pass in.47s, including original dual-reranker controls.
Anonymous input order, source grouping, selection-before-availability, duplicate
pixels, symlink/bytes, manifest/frozen-selection binding, strict ties/failure
denominators, model drift, score/summary validation and test-cannot-repair-dev are
checked. Joint batching/complete stable ranking/filters/final-parent-once pass.
Fake fixtures prove mechanisms only; native quality evidence is the runs above.
Imports/CLI help work under python -S with no optional model/numeric/image imports.
Current workspace391-source check passes. Remaining dual repeats, actual joint
stage control, final full regression/static checks and packaging remain pending.

CLIP4521/SigLIP81419 repeats exited0: every per-case score and ranking exactly
matches first run; repeat costs.082782/.310091s/case. Comparison-2-repeat also
rejects promotion on original cost gates. Actual native stage90275 exits0: two
explicit development images pass one fresh BLIP call, complete order(1,2), one
actual parent JPEG read/SHA verification; wrong tenant gives0 additional model
or final-store calls. Candidate pool is fixed IDs, with no caption index/qrels;
this proves actual composition only, not cross-image retrieval accuracy.
Full configured Ruff/required64 formats/touched7 backend E9,F,I and formatting/
pip/diff checks pass. Full regression47071 now running; build11 and wheel smoke11
remain required, prior305-test/build10 evidence is historical for current changes.

Final RELATION-1 full regression47071 exited0:320 passed in46.41s,5 existing
SWIG warnings,no skips. Build11 session63888 exited0:sdist174 files/wheel166 files,
all four new implementation modules present, no data/models/cache payloads or
wheel tests. Extracted-wheel11 stdlib lifecycle75774 exits0: new relation/joint
imports, bundle/cache/reference lifecycle, actual final source read, repeated warm
worker PID and verified context reaping. Configured Ruff/64 required formats/
touched7 backend E9,F,I+format/pip/diff checks pass. This is extracted distribution
verification, not a clean dependency installation. Relation preference control is
implemented and real quality gain recorded; original cost gates remain failed.

All23 extracted-wheel11 CLI helps5395 exited0 under python -S, including new
relation prepare/evaluate/compare entry point. Workspace391 sources still pass.
All RELATION-1 inference/test/build/smoke processes exited. RELATION-2 spec now
predeclares old-development-only four CPU/numerical profiles, two real repeats
each, and original gates on fresh sources only if a feasible profile exists.
No product configuration/model defaults changed. No commit/push or private data,
service/GPU/cloud operation. Continuous goal remains active; unpassed quality,
broader language/relations/tables/video coverage are still outstanding.

RELATION-2 profile15270 exited0 after all four fixed configurations/two fresh
repeats on six original-development images. Float32(threads1)1.8945/1.8939s,
float32(threads2)1.0410/1.0267s, both6/6 correct. Dynamic-int8 quantizes171 Linear
modules on explicitx86 backend:threads1=1.6664/1.6754s,threads2=.9446/.9502s,
both5/6 correct. No profile passes original<=.5s and<=.025 loss; chosen_config=null.
No new-source data acquired or unsupported quality claim based on this failure.
Native BLIP _supports_sdpa=False and public forwards expose positional interpolation;
RELATION-3 now predeclares one224-native float32 profile, all40 old-development
cases twice before any fresh held-out evaluation. New model option still requires
fresh tests/regression/packaging; build11 evidence predates that option.

RELATION-3 actual profile7745 exited0:224 native position interpolation on all40
original-development images twice,30/40 correct in each, costs.369493/.368512s.
All scores exactly repeat; native-equivalence max absolute error3.8147e-6<1e-4. Original384
development35/40, so.125 accuracy loss exceeds.025; eligible=false despite passing
cost. No old held-out access/new-source acquisition or test-based config repair.
Explicit224 is a distinct producer identity; original384 encoding identity stays
compatible. CLI only permits this option for BLIP; unchanged old invocations keep
384. New tests/checks/build are still required for the added option.

RELATION-4 fixed256/288/320 development grid47320 exited0, two fresh full40-case
runs each. 256=.875/.875 accuracy,.455703/.463193s;288=.8/.8,.559955/.559157s;
320=.825/.825,.680303/.696386s. Only256 meets>=.85 and<=.5 in both runs; fixed
model_id blip-native-itm-v1:4f1698284cbe40270cd6fafc694e92f6b6b3d02198f343b793fa92fe58f9179e.
Grid result SHA55c17b681497b1c2991d8143138d9d56dec29d447ce12708cc1585e4647b374d
is frozen before any new-source image acquisition. New annotations select120
distinct sources excluding all first120, first case replace_rel:62; source overlap0.
Frozen IDs/runtime stored in frozen-cases-2.json before downloader43627 starts.
Fresh-source dev/validation/test quality/cost acceptance remains pending.

Focused63590 exits0:20 tests passed in.58s, covering explicit resolution, one
unique decode per call and image closing, wrong-model CLI option rejection before
IO, prior snapshot revalidation/source exclusion and old-pixel failure without
replacement. Old dataset still loads with original identity. Newly added options
and source-exclusion support require final full checks/build12, not build11 proof.

Full regression2158 exits0:323 passed in73.69s,5 existing SWIG warnings,no skips.
Build12 session5692 exits0:sdist174/wheel166 files, four new modules present,
no runtime model/data/cache payloads. Extracted-wheel12 lifecycle50824 and all23
stdlib CLI helps15473 exit0. Source391/full Ruff/required64 formats/touched7 extra
checks/pip/diff all pass. These precede the final inherited-exclusion correction:
third-round source iteration now carries all previous source/pixel exclusions,
bounded600 groups, preventing accidental first-round resampling. Focused21151
passes20 tests in.67s including that regression. Final full suite/build13 required.

Fresh acquisition43627 exits0 with all120 selected original public JPEGs; prepare
44556 exits0 and verifies no first-round source/pixel duplicates. New identity
46899899f3fa3d0461e0d8ecbf67a20ed9da1bddcf143657b8d07ad713632bcb.
Frozen pre-download case sequence exactly equals prepared cases; acquisition SHA
matches every original copied image. CLIP54310 exits0:78/120=.65, no ties/failures,
.084977s/case. Sequential fixed SigLIP/native256-BLIP and all-model repeats17485
are running; no concurrent benchmarks/tests. Fresh-source gates are not yet known.

New-source first SigLIP/BLIP256 processes under17485 completed:80/97 correct
out of120, no ties/failures. BLIP dev/validation/test=31/31/35 correct out of40,
query costs.444068/.476139/.475040s (overall.465082). All original cost/absolute
accuracy gates pass, but validation gain over SigLIP=.025<.05, so comparison-1
correctly eligible=false. Test.875 cannot reverse rejected validation. Fixed
runtime identity matches pre-acquisition selection. Repeats still running.

Sequential17485 exits0 after all remaining fixed first/repeat runs. All3 new-source
models reproduce every score/ranking exactly, no ties/failures. Repeat BLIP256
cost.457105s/case, partition.466563/.450304/.454448, all<=.5. Two comparisons
reject on validation gain.025<.05; successful test cannot reverse it. BLIP ranking
SHAeb03c572cc465338a0b322d290faa178a5a33f493fea3e1ded55fe3224a64e70.
First native256 equivalence actual error8.6427e-7<1e-4; model loading9.068s and
data verification.078s are separate from first query cost. No configuration retuned
from new validation/test. Latest boundary fix also rejects empty bytes/None as
pair sequences instead of silently accepting them; valid tuples/lists and model
encoding/scoring are unchanged. Final checks/build13 now required.

Final RELATION-4 full regression19387 exited0:323 tests passed in45.41s,5 existing
SWIG warnings,no skips. Build13 session20926 exited0:sdist174 files/wheel166 files,
all four new implementation modules and optional relation-model extra present;
no runtime models/data/cache payloads or wheel tests. Extracted-wheel13 stdlib
lifecycle49803 exits0: new resolution/exclusion signatures, no optional runtime
imports, original bundle/cache/reference handling, actual temporary final-file
read, repeated warm PID and verified context reaping. All23 CLI helps16226 exit0
under python -S. This is an extracted-distribution check, not a clean dependency
installation. Workspace391/full configured Ruff/required64 formatting/touched7
backend extra E9,F,I+format/pip/diff checks pass.

Actual256 stage91248 exits0 on new dev cases62/63:one real joint call processes
two explicit images, full order(1,2), exactly one actual final parent JPEG read
with SHA verification; wrong tenant0 extra model/store calls. No cross-image
quality claimed from this fixed candidate pool. Both public raw image batches
retain all240 acquisition SHA256 values. All owned download/inference/profile/
test/build/smoke processes exited. No commit/push/private-library/default-model/
GPU/cloud/service changes. RELATION-1–4 controls are implemented and evidenced;
validation promotion still fails, image rejection/full quality and broader
cross-image relations/language/tables/independently labelled videos remain open.
The continuous goal stays active; this is not all-TODO completion.
