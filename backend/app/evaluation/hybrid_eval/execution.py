"""Explicit route construction, candidate caching and deterministic evaluation."""

from __future__ import annotations

import dataclasses
import sys
import time
from collections.abc import Iterable, Mapping, Sequence

from backend.app.evaluation import dataset as evaluation_dataset
from backend.app.evaluation import metrics
from backend.app.evaluation.hybrid_eval import dataset as datasets
from backend.app.evaluation.hybrid_eval import encoders, reporting
from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import adapters, lexical, sparse, vector
from backend.app.retrieval.hybrid import documents as corpus

BASELINE = "bm25-1.2-0.75"
# The fixed workflow stops rankings at 10, so its historical 'mrr' key means MRR@10.
METRICS = ("ndcg_at_10", "recall_at_10", "mrr")


def route_definitions() -> dict[str, reporting.JsonObject]:
    """Return the fixed, JSON-compatible route space used by selection locks."""
    return {
        **{f"bm25-{k1}-{b}": {"kind": "bm25", "k1": k1, "b": b} for k1 in (0.6, 1.2, 2.0) for b in (0.25, 0.5, 0.75)},
        "word": {"kind": "sparse", "analyzer": "word", "ngram_range": [1, 2], "max_features": 60000},
        "char": {"kind": "sparse", "analyzer": "char_wb", "ngram_range": [3, 5], "max_features": 60000},
        "lsa": {"kind": "lsa", "dimensions": 128, "seed": 42},
    }


def _build_bm25(documents: Sequence[corpus.CorpusDocument], definition: reporting.JsonObject) -> hybrid.Retriever:
    return lexical.Bm25Retriever(documents, k1=definition["k1"], b=definition["b"])


def _build_sparse(
    documents: Sequence[corpus.CorpusDocument], definition: reporting.JsonObject
) -> sparse.SparseTfidfRetriever:
    return sparse.SparseTfidfRetriever(
        documents,
        analyzer=definition["analyzer"],
        ngram_range=tuple(definition["ngram_range"]),
        max_features=definition["max_features"],
    )


def _build_lsa(documents: Sequence[corpus.CorpusDocument], definition: reporting.JsonObject) -> hybrid.Retriever:
    encoder = encoders.LatentSemanticEncoder(documents, dimensions=definition["dimensions"], seed=definition["seed"])
    return vector.ExactVectorRetriever(
        [doc.asset_id for doc in documents],
        encoder.vectors,
        encoder.encode_query,
        encoder_id=encoder.encoder_id,
        query_encoder_id=encoder.encoder_id,
        metadata={doc.asset_id: doc.metadata for doc in documents},
    )


_ROUTE_BUILDERS = {"bm25": _build_bm25, "sparse": _build_sparse, "lsa": _build_lsa}


def build_routes(
    documents: Sequence[corpus.CorpusDocument], names: Iterable[str], *, progress: bool = False
) -> tuple[dict[str, hybrid.Retriever], reporting.JsonObject]:
    """Construct only explicitly named corpus-fitted routes and record builds.

    Args:
        documents: Corpus used for all route representations.
        names: Names from the frozen route definitions.
        progress: Emit build observations to stderr when enabled.

    Returns:
        Named retrievers and measured build/resource details.

    Raises:
        KeyError: A route name is outside the frozen experiment space.
    """
    routes: dict[str, hybrid.Retriever] = {}
    measurements: reporting.JsonObject = {}
    definitions = route_definitions()
    for name in sorted(names):
        definition = definitions[name]
        started = time.perf_counter()
        retriever = _ROUTE_BUILDERS[definition["kind"]](documents, definition)
        routes[name] = retriever
        measurements[name] = {"seconds": time.perf_counter() - started, "definition": definition}
        if isinstance(retriever, sparse.SparseTfidfRetriever):
            matrix = retriever.matrix
            measurements[name].update(
                features=matrix.shape[1],
                nonzero_entries=matrix.nnz,
                csr_bytes=matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes,
            )
        if progress:
            print(f"Built {name} in {measurements[name]['seconds']:.2f}s", file=sys.stderr, flush=True)
    return routes, measurements


def config_for(
    primary: str, *, secondary: str | None = None, weight: float = 1.0, constant: float = 60, window: int = 200
) -> hybrid.SearchConfig:
    """Create a single route or rank-fusion configuration with explicit budgets."""
    branches = (hybrid.Branch(primary, window=window),)
    if secondary:
        branches += (hybrid.Branch(secondary, window=window, weight=weight),)
    return hybrid.SearchConfig(branches, rrf_constant=constant)


def cache_pools(
    routes: Mapping[str, hybrid.Retriever],
    cases: Sequence[evaluation_dataset.EvaluationCase],
    *,
    progress: bool = False,
) -> tuple[dict[str, dict[str, Sequence[int]]], reporting.JsonObject]:
    """Cache each route's first 200 IDs once per query and time actual retrieval.

    This unfiltered pool covers the largest window in the fixed selection space.
    Its latency includes query encoding and route retrieval, not fitting or fusion.
    Subsequent candidate evaluations reuse ranks and cannot measure retrieval cost.
    """
    pools, timings = {}, {name: [] for name in routes}
    for case_index, case in enumerate(cases, start=1):
        pools[case.query_id] = {}
        for name, retriever in routes.items():
            started = time.perf_counter()
            pools[case.query_id][name] = retriever.retrieve(hybrid.SearchQuery(case.query), 200)
            timings[name].append(time.perf_counter() - started)
        if progress and (case_index % 100 == 0 or case_index == len(cases)):
            print(f"Retrieved {case_index}/{len(cases)} queries", file=sys.stderr, flush=True)
    return pools, {
        name: {"p50": reporting.percentile(values, 0.5), "p95": reporting.percentile(values, 0.95)}
        for name, values in timings.items()
    }


def evaluate_config(
    config: hybrid.SearchConfig,
    cases: Sequence[evaluation_dataset.EvaluationCase],
    pools: Mapping[str, Mapping[str, Sequence[int]]],
) -> tuple[reporting.JsonObject, reporting.Rankings]:
    """Evaluate a configuration from cached ranks without refitting a model.

    cases must be nonempty; each query needs a pool for every configured branch,
    covering that branch's window. Metrics average queries equally (macro average),
    rather than pooling all relevant documents into a single denominator.
    """
    rankings, query_metrics = {}, []
    for case in cases:
        engine = hybrid.HybridSearch(
            {branch.name: adapters.RankedListRetriever(pools[case.query_id][branch.name]) for branch in config.branches}
        )
        rankings[case.query_id] = [hit.asset_id for hit in engine.search(hybrid.SearchQuery(case.query), config).hits]
        query_metrics.append(metrics.metrics_for_case(case, rankings[case.query_id]))
    return {
        "config": dataclasses.asdict(config),
        "metrics": {key: sum(row[key] for row in query_metrics) / len(query_metrics) for key in METRICS},
        "ranking_sha256": reporting.digest(rankings),
    }, rankings


def read_config(value: reporting.JsonObject) -> hybrid.SearchConfig:
    """Reconstruct a validated public search config from the stored JSON shape."""
    return hybrid.SearchConfig(
        tuple(hybrid.Branch(**branch) for branch in value["branches"]),
        top_k=value["top_k"],
        rrf_constant=value["rrf_constant"],
        failure_policy=value["failure_policy"],
    )


def live_pipeline(
    config: hybrid.SearchConfig,
    dataset: datasets.Dataset,
    routes: Mapping[str, hybrid.Retriever],
    expected: Mapping[str, list[int]],
) -> reporting.JsonObject:
    """Time actual retrieval/fusion/fetch and verify cached-rank equivalence.

    'Live' means serial local execution of already built routes and an in-memory
    store, not a deployed service or disk-backed index. The observation excludes
    index construction, network traffic and concurrent request scheduling.

    Raises:
        ValueError: Actual top-k differs from the precomputed comparison.
    """
    store = reporting.CountingStore(dataset.documents)
    engine = hybrid.HybridSearch({branch.name: routes[branch.name] for branch in config.branches}, store=store)
    latencies = []
    for case in dataset.cases:
        started = time.perf_counter()
        result = engine.search(hybrid.SearchQuery(case.query), config)
        latencies.append(time.perf_counter() - started)
        if [hit.asset_id for hit in result.hits] != expected[case.query_id]:
            raise ValueError("Live pipeline differs from cached-rank evaluation")
    return {
        "latency_ms": {"p50": reporting.percentile(latencies, 0.5), "p95": reporting.percentile(latencies, 0.95)},
        "queries": len(dataset.cases),
        "fetch_calls": store.calls,
        "records_fetched": store.rows,
        "max_records_per_fetch": store.max_rows,
        "scope": "Serial in-process query encoding, retrieval, fusion and in-memory final fetch; excludes index build.",
    }
