# 0005: Separate retrieval adapters and evaluation libraries from command IO

Status: accepted; backend ownership, see spec 0005.

Apply selected Google Python Style Guide rules through the root AGENTS.md, with
explicit local formatting/import adaptations. Keep document/store, lexical and
dense retrieval implementations cohesive while preserving the adapter facade.
Break search coordination into branch execution, fusion-output validation and
final batch fetch. Broad exception translation remains confined to the two
documented retriever/store extension boundaries, with causes retained.

Share evaluation data, encoders, statistics, execution and workflows in
evaluation/hybrid_eval. CLI modules import the library and retain old helper
names as explicit compatibility exports. Keep optional model/numeric imports
lazy; do not add another environment or mandatory style dependency. Report data
is explicitly dynamic JSON; retrieval contracts keep concrete domain annotations.

This is a structural change. Preserve parameters, accumulation order, tie rules,
selection checksum payloads and output keys. Validate against the exact prior
source snapshot and existing public-corpus rankings; do not regenerate expected
outputs or reinterpret the change as a new quality/performance improvement.
