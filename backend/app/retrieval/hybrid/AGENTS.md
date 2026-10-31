# Hybrid Search

Follow root/backend rules and spec 0003. Core contracts, fusion and engine must
use only the standard library and relative imports. Keep database entities,
settings, model loading and native scores outside fusion. Adapters apply filters
before candidate windows. Tests use temporary or explicitly selected public data.
Sparse adapters and selection/evaluation follow spec 0004; keep numeric imports
lazy and preserve the locked development/validation/test boundary.
Module ownership and compatibility follow spec 0005 and the root Google-derived
Python engineering rules. Implement documents/store, BM25 and dense vectors in
documents.py, lexical.py and vector.py; adapters.py preserves the public facade.
Evaluation libraries must not depend on their CLI entrypoints.

Spec 0006 adds opt-in units/stages/reranking/media adapters. Integer retrieval
IDs refer to units there; final store IDs refer to parent assets. Never confuse
their domains or count candidate-feature access as final resource hydration.
Keep core units/stages/reranking standard-library-only and model/label independent.
The color adapter is an explicitly limited pixel reference, not semantic vision.

Spec 0007 adds multimodal request planning, evidence families, immutable snapshots,
local model/OCR adapters and process execution. Keep native dependencies lazy and
model paths explicit/local-only. ProcessSearch constructs retriever and final
store/fusion factories inside cancellable workers; final records are bounded JSON.
Keep the synchronous HybridSearch object API compatible. WarmProcessSearch needs
an explicit context owner; confirm cleanup, preserve capacity and refuse shared
concurrent requests. Reused scores/embeddings are not fresh inference evidence.
Use cache_keys.multimodal_key for image/combined requests; text-only cache keys
reject those requests. Never treat fixture or shape checks as model quality.
