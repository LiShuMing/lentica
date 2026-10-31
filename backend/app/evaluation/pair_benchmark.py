"""Run fixed local native pair reranking without changing a selected default."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.evaluation.hybrid_eval import dataset, native_quality, pair_checkpoints, pair_quality
from backend.app.retrieval.hybrid import pair_reranking, semantic, vector


def main() -> None:
    """Own explicit model/cache paths and fresh report IO; network is not used."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--bge-model", required=True, type=Path)
    parser.add_argument("--pair-model", required=True, type=Path)
    parser.add_argument("--corpus-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--score-checkpoint", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh report path")
    import torch

    torch.set_num_threads(2)
    source = dataset.load_dataset(args.dataset)
    encoder = semantic.NativeTextEncoder(args.bge_model)
    values, _ = native_quality.load_vectors(args.corpus_cache, source, encoder)
    queries = encoder.encode_queries([case.query for case in source.cases])
    query_vectors = {case.query: queries[index] for index, case in enumerate(source.cases)}
    dense = vector.ExactVectorRetriever(
        [doc.asset_id for doc in source.documents],
        values,
        query_vectors.__getitem__,
        encoder_id=encoder.encoder_id,
        query_encoder_id=encoder.encoder_id,
    )
    model = pair_reranking.PairReranker(args.pair_model)
    checkpoint = (
        pair_checkpoints.PairScoreCache(
            args.score_checkpoint,
            {
                "corpus": dataset.dataset_identity(source),
                "cases": dataset.case_identity(source.cases),
                "model": model.version,
                "encoder": encoder.encoder_id,
                "controls": "bm25/three-routes:candidates20:window50:c60",
            },
        )
        if args.score_checkpoint is not None
        else None
    )
    report = pair_quality.run(source, dense, model, progress=True, score_cache=checkpoint)
    report["encoder_id"] = encoder.encoder_id
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {"output": str(args.output), "quality": report["quality"], "ranking_sha256": report["ranking_sha256"]}
        )
    )


if __name__ == "__main__":
    main()
