"""Run predeclared persistent-index controls on explicitly selected public data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from backend.app.evaluation.hybrid_eval import dataset, index_quality, native_quality, reporting
from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import semantic, vector


def main() -> None:
    """Own model construction and fresh artifact/report paths; never select defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--corpus-cache", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.artifact_dir.exists():
        parser.error("Choose fresh artifact and report paths")
    import torch

    torch.set_num_threads(2)
    source = dataset.load_dataset(args.dataset)
    encoder = semantic.NativeTextEncoder(args.model_dir)
    values, manifest = native_quality.load_vectors(args.corpus_cache, source, encoder)
    started = time.perf_counter()
    queries = encoder.encode_queries([case.query for case in source.cases])
    encoding_seconds = time.perf_counter() - started
    print(f"Encoded {len(queries)} queries in {encoding_seconds:.2f}s", flush=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=False)
    ann_report = index_quality.compare_ann(source, values, queries, encoder.encoder_id, args.artifact_dir / "ann")
    print("ANN comparisons complete", flush=True)
    # Four shards share precisely the same native query vectors/encoder space.
    encoded = {case.query: queries[index] for index, case in enumerate(source.cases)}
    dense = vector.ExactVectorRetriever(
        [doc.asset_id for doc in source.documents],
        values,
        encoded.__getitem__,
        encoder_id=encoder.encoder_id,
        query_encoder_id=encoder.encoder_id,
    )
    pools = {case.query_id: dense.retrieve(hybrid.SearchQuery(case.query), 50) for case in source.cases}
    shard_report = index_quality.compare_shards(source, pools, args.artifact_dir / "text")
    report = {
        "schema": "lentica-persistent-index-fixed-v1",
        "environment": reporting.environment(),
        "source": source.provenance,
        "corpus": dataset.dataset_identity(source),
        "encoder_id": encoder.encoder_id,
        "corpus_cache": manifest,
        "query_encoding_seconds": encoding_seconds,
        "ann": ann_report,
        "shards": shard_report,
        "scope": "Fixed historical public test controls, serial CPU; no tuning, default switch or serving claim.",
    }
    report["ranking_sha256"] = reporting.digest(
        {
            "ann": {key: value["rankings"] for key, value in ann_report["domains"].items()},
            "shards": shard_report["rankings"],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "ranking_sha256": report["ranking_sha256"]}))


if __name__ == "__main__":
    main()
