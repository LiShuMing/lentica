"""Run fixed native text/vector filtering controls on explicit local indexes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.evaluation.hybrid_eval import dataset, filter_quality, reporting
from backend.app.retrieval.hybrid import ann, fulltext, semantic


def main() -> None:
    """Reopen matching immutable indexes; share actual encoded queries across repeats."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--text-index", required=True, type=Path)
    parser.add_argument("--vector-index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh report path")
    import faiss
    import torch

    faiss.omp_set_num_threads(1)
    torch.set_num_threads(2)
    source = dataset.load_dataset(args.dataset)
    encoder = semantic.NativeTextEncoder(args.model_dir)
    values = encoder.encode_queries([case.query for case in source.cases])
    query_vectors = {case.query: values[index] for index, case in enumerate(source.cases)}
    text = fulltext.TantivyRetriever(args.text_index)
    dense = ann.HnswRetriever(args.vector_index, query_vectors.__getitem__, query_encoder_id=encoder.encoder_id)
    first, second = filter_quality.run(source, text, dense), filter_quality.run(source, text, dense)
    if first != second:
        raise ValueError("Repeated fixed filtering ranks/metrics differ")
    report = {
        "schema": "lentica-native-filter-fixed-v1",
        "source": source.provenance,
        "encoder_id": encoder.encoder_id,
        "environment": reporting.environment(),
        "repeat_equal": True,
        "query_encoding": f"All {len(source.cases)} queries actually encoded once; vectors shared by both independent retrieval repeats.",
        **first,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "domains": {name: value["quality"] for name, value in report["domains"].items()},
            }
        )
    )


if __name__ == "__main__":
    main()
