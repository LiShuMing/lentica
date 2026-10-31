"""Fixed pretrained-text controls with verified corpus cache and actual queries."""

from __future__ import annotations

import json
import time
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

from backend.app.evaluation import metrics
from backend.app.evaluation.hybrid_eval import dataset, execution, reporting
from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import lexical, semantic, snapshots, sparse, vector

if TYPE_CHECKING:
    from numpy import typing as npt


def fixed_configs() -> dict[str, hybrid.SearchConfig]:
    """Return the predeclared controls; never choose weights on test labels."""
    return {
        "bm25": hybrid.SearchConfig((hybrid.Branch("bm25", 50),)),
        "original-hybrid": hybrid.SearchConfig((hybrid.Branch("bm25", 50), hybrid.Branch("char", 50)), rrf_constant=1),
        "bge": hybrid.SearchConfig((hybrid.Branch("bge", 50),)),
        "bm25-bge": hybrid.SearchConfig((hybrid.Branch("bm25", 50), hybrid.Branch("bge", 50))),
        "three-routes": hybrid.SearchConfig(
            (hybrid.Branch("bm25", 50), hybrid.Branch("char", 50), hybrid.Branch("bge", 50))
        ),
    }


def load_vectors(
    directory: Path, source: dataset.Dataset, encoder: semantic.NativeTextEncoder
) -> tuple[npt.NDArray, reporting.JsonObject]:
    """Load only a checksum-bound, source/model/ID-aligned local corpus cache."""
    import numpy as np

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["encoder_id"] != encoder.encoder_id
        or manifest["corpus_sha256"] != dataset.dataset_identity(source)
        or manifest["ids"] != [document.asset_id for document in source.documents]
    ):
        raise ValueError("Corpus cache has a different source, model, or ID ordering")
    if snapshots.sha256_file(directory / "vectors.npy") != manifest["vectors_sha256"]:
        raise ValueError("Corpus vector cache digest mismatch")
    values = np.load(directory / "vectors.npy", allow_pickle=False)
    if values.shape != (len(source.documents), encoder.dimensions) or not np.isfinite(values).all():
        raise ValueError("Corpus cache shape/finite-value contract violated")
    # Check actual encoder/cache alignment on fixed endpoints as an additional
    # provenance probe; this does not replace the complete artifact checksum.
    probes = [0, len(source.documents) - 1]
    texts = [source.documents[index].title + " " + source.documents[index].text for index in probes]
    if not np.allclose(values[probes], encoder.encode_documents(texts), rtol=1e-4, atol=1e-5):
        raise ValueError("Corpus cache does not match the selected native encoder")
    return values, manifest


def run(
    source: dataset.Dataset, model_directory: Path, corpus_cache: Path, *, progress: bool = False
) -> reporting.JsonObject:
    """Measure all fixed controls, recording regressions without choosing a winner.

    Corpus vectors are reused only after identity checks. Each measured route
    computes actual query embeddings. Real baseline/three-route pipelines are
    checked against cached rankings and include one final in-memory asset fetch.
    Historical-test exposure remains explicit; no default model/index is changed.
    """
    encoder = semantic.NativeTextEncoder(model_directory)
    values, cache_manifest = load_vectors(corpus_cache, source, encoder)
    started = time.perf_counter()
    routes = {
        "bm25": lexical.Bm25Retriever(source.documents),
        "char": sparse.SparseTfidfRetriever(source.documents, analyzer="char_wb", ngram_range=(3, 5)),
        "bge": vector.ExactVectorRetriever(
            [document.asset_id for document in source.documents],
            values,
            encoder.encode_query,
            encoder_id=encoder.encoder_id,
            query_encoder_id=encoder.encoder_id,
        ),
    }
    build_seconds = time.perf_counter() - started
    pools, route_timings = execution.cache_pools(routes, source.cases, progress=progress)
    configs = fixed_configs()
    reports, rankings = {}, {}
    for name, config in configs.items():
        reports[name], rankings[name] = execution.evaluate_config(config, source.cases, pools)
    live = {
        name: execution.live_pipeline(configs[name], source, routes, rankings[name])
        for name in ("bm25", "three-routes")
    }
    return {
        "schema": "lentica-native-text-fixed-v1",
        "encoder_id": encoder.encoder_id,
        "encoding": encoder.encoding,
        "scope": "Fixed pretrained CPU controls; corpus encoding cached after identity/probe checks; actual query encoding. "
        "Historical test, no parameter selection or unseen-generalization claim.",
        "dataset": {**source.provenance, "documents": len(source.documents), "queries": len(source.cases)},
        "environment": {
            **reporting.environment(),
            "torch": metadata.version("torch"),
            "transformers": metadata.version("transformers"),
        },
        "corpus_encoding": cache_manifest,
        "route_build_seconds_excluding_corpus_encoding": build_seconds,
        "comparisons": reports,
        "route_latency_ms": route_timings,
        "actual_pipeline": live,
        "paired_vs_original_hybrid": {
            name: reporting.paired_comparison(source.cases, rankings["original-hybrid"], rankings[name])
            for name in ("bge", "bm25-bge", "three-routes")
        },
        "per_query": [
            {
                "query_id": case.query_id,
                "query": case.query,
                "rankings": {name: ranking[case.query_id] for name, ranking in rankings.items()},
                "metrics": {
                    name: metrics.metrics_for_case(case, ranking[case.query_id]) for name, ranking in rankings.items()
                },
            }
            for case in source.cases
        ],
        "ranking_sha256": reporting.digest(rankings),
        "decision": "Report every predeclared control; no default switch or test-based winner selection.",
    }
