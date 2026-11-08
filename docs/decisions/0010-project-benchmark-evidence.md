# Isolated product and optional-component benchmark evidence

Spec: 0010-project-benchmark. Status: accepted measurement design.

Measure the unchanged product SQL/scoring/coordinator path separately from
optional native retrievers and model-quality controls. Use explicit public data,
fresh isolated work directories and fixed configurations. Shared-host latency,
synthetic scale, CPU/mock regression and real pretrained inference have distinct
evidence scopes; none substitutes for GPU/cloud/GUI/browser or private-library
acceptance. Existing model-quality failures remain failures on reproduction.

The benchmark command sets isolated product configuration before any product
imports. It never selects a user's database or changes serving defaults. Public
corpus features exclude query/relevance labels; failed queries remain in metric
denominators. New numerical artifacts are local ignored outputs, not source.
