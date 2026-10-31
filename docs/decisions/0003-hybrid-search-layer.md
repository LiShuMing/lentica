# ADR 0003: Independent Hybrid Search Contracts

Status: accepted. Spec: 0003-hybrid-search-layer.

Keep a dependency-free retrieval core inside the backend distribution, with no
imports of application configuration, database entities or model implementations.
Separate ranked-ID retrieval, fusion, and final record hydration. An adapter owns
native scoring and pre-window filter support. The coordinator owns validated
branch budgets, explicit failures and one final batch fetch. RRF consumes ranks
only; its constant and per-route windows/weights are explicit experiment inputs.

Keep the existing photo fusion function as a compatibility facade. This first
slice does not change photo candidate materialization or API response fields.
CPU BM25 and exact vector adapters provide a working baseline. Statistical
latent semantic encoding can evaluate the pipeline without requiring a model
download; its results do not establish pretrained semantic model quality.

Use measured labeled-query outcomes to choose later adapters. Remote vector
services, approximate indexes and learned rerankers remain optional future work.
