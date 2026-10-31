# 0006: Separate multimodal units, candidate ranking and resource presentation

Status: accepted; M1 implemented and verified, M2/M3 planned. See spec 0006.

Add an opt-in standard-library pipeline around the existing HybridSearch instead
of changing its public contracts. Existing integer candidate IDs identify units
within this pipeline; a catalog maps them to resource IDs and precise locators.
Representations remain adapter-owned, with explicit text-evidence provenance.

Use separate branch/candidate/rerank/final budgets. Apply pre-window filters,
limit units per resource, rerank a bounded prefix, then aggregate by the best
unit rather than summing page/frame counts. Preserve supporting unit locations.
Read candidate features separately from the one final resource-store batch;
never extend the old one-fetch claim to all candidate text or the photo business.

Ship measurable CPU baselines first: locked public-corpus candidate diagnostics,
a fixed identifier reranker and generated image/PDF controls. Pretrained visual/
semantic models, actual OCR, image-only input, audio/video, persistent indexes
and service integration require their own later acceptance evidence. Keep all
existing import/CLI/API/data/model defaults compatible and private sources intact.
