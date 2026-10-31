# One product coordinator for photo and document evidence

Spec0009 unifies real API/UI calls through StagedSearch/HybridSearch. Keep SQL,
encoder construction and presentation in product adapters. Expose native branch
ranked IDs, not pre-fused/presented per-modality final lists. File.id identifies a
resource; parity-separated image/chunk IDs prevent namespace collisions. Cap
supporting units, select one global resource budget, then invoke one final SQL
store batch. Feature/index reads remain explicit and do not count as final reads.

Preserve old response arrays as projections; add global results as authoritative
cross-modality order. Existing photo weights and document weight1 are explicit
initial integration policy, not an evaluated quality improvement. Strict missing
record behavior prevents incomplete citations. Optional native indexes, caches,
rerankers and process cancellation remain separate activation decisions; no
private index migration or service restart follows from source integration.
