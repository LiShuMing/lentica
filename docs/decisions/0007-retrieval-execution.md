# 0007: Keep query capabilities, evidence families and snapshot identity explicit

Status: accepted design, implementation in progress; see spec 0007.

Checkpoint: capabilities/families, feature snapshots, persistent adapters and
provider-process cancellation are implemented. Real CPU BGE/CLIP/OCR inference
has executed; fixed SciFact controls and generated routing controls repeat.
Public-scale ANN and fixed text-pair reranking are measured. Three source-disjoint
image trials and actual bounded multimodal dual-encoder reranking are measured,
with validation/cost failures preserved. Broader quality, sound/video and temporal
models remain active. A managed warm executor
reuses per-route models, enforces capacity and replaces cancelled workers; final
workers remain cold. Final store/fusion factories run in a cancellable process
sharing the request deadline. ProcessSearch only
accepts factory specs for these plugins and bounded JSON records; the established
synchronous HybridSearch instance-based API is unchanged.

Extend the optional unit pipeline with explicit multimodal inputs and capability
routing rather than changing the old text-only SearchQuery/HybridSearch API.
Combine correlated route contributions by maximum within each declared family,
then sum independent families. Route decisions never read relevance judgments.

Use a new immutable SQLite snapshot as the standard-library persistent baseline.
Pin readers to one version; incremental updates produce new snapshots. Publish a
verified reference atomically and preserve previous snapshots for rollback.
Index/model/source/config/filter identities belong in cache keys.

Evaluate source-grouped data and localization separately from ranking. Use real
subprocess termination for execution deadlines. Optional model/index/OCR tools
require explicit adapters and actual runtime acceptance, with missing artifacts
reported as incomplete work, not mock accuracy. Product defaults stay unchanged.

After two preserved CLIP refusal failures, trial a separate native SigLIP dual
encoder on sources/pixels excluded from both previous trials. Keep development,
validation and frozen test separate. Use the same encoder for explicitly bounded
text-image candidate reranking; this is independent-encoder scoring, not joint
cross-attention or a calibrated relevance probability. Include fixed CLIP controls
on the same new test and report both gains and regressions without retuning there.

Mixed PDF layout is a separate opt-in entrypoint: in-memory logical-orientation
OCR/table extraction, displayed coordinates, native-cell replacement and narrow
spatial/text duplicate suppression. Preserve existing native/scanned-only APIs.
This is ruled-table/geometric ordering, not a semantic layout model. Local speech
uses verified Whisper tiny English ONNX and explicit PCM16 WAV; native greedy
tokens and coarse windows are versioned. Public-example inference remains
distinct from reference-scored ASR and acoustic/temporal model quality.

Keep subtitle/frame/ASR source timestamps explicit. Temporal conjunction joins
evidence only within one parent resource and a nonempty common time interval,
under shared filters and bounded combinations. Coarse ASR chunks retain their
coarse meaning. Actual video decoding and generated timeline controls do not
certify action understanding, variable-frame-rate timing or sound-event models.

The optional CLAP path keeps native mel features and independently encoded
unpadded text queries: its selected export lacks an attention mask. Five-class
source-separated ESC50 selection/test gates passed twice, with vocabulary and
pretraining-overlap limits documented; no default was changed. Speech quality
now adds a separate fixed public reference slice and micro word-error rate.
References never enter model input; failed predictions remain in the denominator.
PyAV/Parquet decoding is isolated in an optional evaluation extra and does not
change the product audio runtime or introduce implicit media conversion.

The fourth isolated image trial selects runtime configuration on development
inputs only and adds fixed Chinese/paraphrase coverage. Failed validation/cost
and weak language prompts remain visible. Test success cannot reverse failed
validation: promotion eligibility is reported separately from test-only gates.
Native safetensors VideoMAE adds actual temporal inference, with reversed/static
controls kept distinct from original sequence features. Its exact-class adapter
returns filtered IDs and composes with the existing unit/resource stages; class
logits do not enter fusion. A one-clip source-named label cannot certify general
action understanding, event timing or time-order sensitivity.


The fifth image trial adds explicit fast/slow category gating. Only actual eligible
pixels reach the slow model; its features never mix with CLIP vectors. Compare
full SigLIP quality/cost separately. Development hit loss rejects this candidate
although held-out cost/rejection gates pass; defaults remain unchanged.

Chinese adaptation uses separately verified local Marian ONNX with native source/
target SentencePiece and bounded greedy decoding. Preserve original query,
generated text and producer identity; require natural EOS and keep failed queries
in metric denominators. The checkpoint defaults to six beams, so greedy is an
explicit alternative. Fixed sixth-source Chinese/translation/English controls
measure this narrow query slice; translation is neither automatic nor general
multilingual quality certification. Dependencies remain an optional extra.


CASCADE-2 explicitly rescues uncertain fast-negative queries and uses actual slow
features against a separately built slow-space gallery. Fixed fast tolerance.02
precedes all seventh-source test access. Add a nDCG-loss gate against full SigLIP
so lower cost cannot hide the ranking regression found in CASCADE-1. Index-build,
prompt, loading and complete-reference costs stay separate from query encoding/
ranking costs. Source separation excludes all previous3600 images/pixels.

CASCADE-3 keeps the same quality/cost gates and narrows the explicit rescue
tolerance to.005 after seventh-source development costs exceeded budget. Its
eighth-source protocol excludes4200 earlier sources/pixels and is fixed before
new test access. Selection files bind the exact tolerance; a mismatch fails
before gallery construction or held-out query inference. No test-guided retuning.


Column ordering is a separate opt-in stdlib geometric adapter over immutable page
units. Preserve all IDs/locations/evidence and keep existing extraction defaults.
Use bounded recursive XY-cut with full-width edge-band isolation and stable
fallback. Native block geometry and rendered visual references are separate
sources of evidence: retain failed annotations/reports and disclose visually
confirmed transcription corrections instead of crediting them to the algorithm.
The explicit.015 trial is development-motivated; another page of the same paper
is a narrow heading-order check, not a new independent-document test corpus.


Native unruled tables get a separate explicit-region adapter. Project word boxes
onto axes and cut only sufficiently large whitespace; thresholds scale with median
word height. Keep old ruled-table/OCR APIs unchanged. Symbol-font overlap marks the
whole native word undecidable, even when mixed with regular spans; preserve raw
encoding/font/location in opaque glyph records while excluding fake letters from
search evidence. Empty/glyph-only cells remain structural slots without invented
text or checkbox state. A previously inspected public survey verifies24 slots,
localization and actual one-final-PDF-read composition; broad table discovery,
merged/image tables and independent-document quality remain separate work.


CASCADE-4 replaces successive fixed-tolerance guesses with one predeclared joint
calibration on ninth-source development images. All four tolerances execute real
fast queries and eligible slow queries; only the partition-bound gallery and
full slow reference report are shared. Query vectors never cross candidates.
Choose the smallest feasible fast tolerance then slow margin, validate one pair
without replacement, and retain all original quality/cost gates. A separate
selection schema binds both grids, constraints, source and model identities;
failed development/validation cannot be reversed by test. Existing strict/fixed
rescue policies and product defaults retain their original contracts.


TABLE-2 keeps typed numeric-table row contexts separate from existing PdfTable/
TableCell extraction. Native rulings supply coarse geometry; repeated dense word
baselines split data subrows. Centered shared values carry marked geometric-span
provenance, while ambiguous sparse text fails instead of forward-filling. Caller
field keys/labels and header count are explicit configuration, included in producer
identity; expected values never enter extraction. Inherited source filters and
reserved table.field. conditions are applied before candidate windows. One shared
bounded PublicPdfStore verifies actual final reads for both real-table controls.
This is explicit structural retrieval, not automatic header semantics or free QA.

RELATION-1 distinguishes independent image/text embedding similarity from genuine
pair-conditioned matching. Add an explicit ImageTextPairScorer consumer contract
and JointTextImageReranker; reuse candidate-domain validation and retain stable
full permutations and one final parent read. BLIP native cross attention shares
vision only for identical explicit paths within one bounded call. Publisher bin
artifacts require verified size/SHA and Torch>=2.6 explicit weights_only loading;
no unrestricted deserialization, remote code, default model change or core model
dependency. Model identity includes native processing, runtime and numerical profile.
Public source-grouped, counterbalanced SugarCrepe caption preferences complement
class-level retrieval controls. They do not substitute for cross-image Recall or
open-world confidence. Each partition must pass both fixed quality and actual CPU
cost gates; a better test cannot repair failed development/validation. Unknown
pretraining/COCO fine-tuning overlap and the selected subset remain explicit limits.

CPU/numerical and resolution optimization uses only original development sources.
Retain failed dynamic-int8 and224 controls. BLIP's public position interpolation
permits explicit smaller processor inputs with distinct identities, while original
384 keeps its exact configuration/identity. Fixed grids execute all candidates
twice before freezing one for new data. Reverify prior snapshots, remove prior
source IDs before availability, and fail on old pixel duplicates without sample
replacement. Self-contained exclusions bind selection/manifest identities; old
manifests remain readable. This iteration does not lower quality/cost gates or
convert a development speed gain into a completed retrieval-quality claim.
