"""Measure fixed public cold/warm process lifecycle controls on local CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.evaluation.hybrid_eval import dataset, warm_quality
from backend.app.retrieval.hybrid import execution


def main() -> None:
    """Select exact native factories and first five queries, and write a fresh report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--text-index", required=True, type=Path)
    parser.add_argument("--vector-index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh report path")
    source = dataset.load_dataset(args.dataset)
    for directory, filename in ((args.text_index, "lentica.json"), (args.vector_index, "manifest.json")):
        manifest = json.loads((directory / filename).read_bytes())
        if manifest["source_version"] != dataset.dataset_identity(source):
            parser.error("Native index source identity differs from the selected dataset")
    providers = {
        "text": execution.ProviderSpec(
            "backend.app.retrieval.hybrid.fulltext", "TantivyRetriever", {"directory": str(args.text_index.resolve())}
        ),
        "dense": execution.ProviderSpec(
            "backend.app.retrieval.hybrid.native_providers",
            "NativeHnswProvider",
            {"model_directory": str(args.model_dir.resolve()), "index_directory": str(args.vector_index.resolve())},
        ),
    }
    report = warm_quality.run(
        providers, {doc.asset_id: doc.metadata for doc in source.documents}, [case.query for case in source.cases[:5]]
    )
    report["source"] = source.provenance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "latency_ms": report["latency_ms"]}))


if __name__ == "__main__":
    main()
