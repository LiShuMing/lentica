"""Optional bounded Parquet round trips for explicit public evaluation datasets."""

from __future__ import annotations

import collections
import json
from pathlib import Path

from backend.app.evaluation import dataset as evaluation_dataset
from backend.app.evaluation.hybrid_eval import dataset, reporting
from backend.app.retrieval.hybrid import documents, lexical, snapshots


def _validate(source: dataset.Dataset) -> None:
    ids = {document.asset_id for document in source.documents}
    if not ids or len(ids) != len(source.documents) or set(source.source_ids) != ids:
        raise ValueError("Corpus/source IDs must be complete and unique")
    if any(identifier > 2**63 - 1 for identifier in ids) or len(set(source.source_ids.values())) != len(ids):
        raise ValueError("Document IDs must fit signed int64 and source IDs must be unique")
    if len({case.query_id for case in source.cases}) != len(source.cases) or not source.cases:
        raise ValueError("Queries must be nonempty and uniquely identified")
    if any(
        not case.query_id or not case.query.strip() or not case.relevant_asset_ids or not case.relevant_asset_ids <= ids
        for case in source.cases
    ):
        raise ValueError("Judged queries require text and known positive document IDs")


def write(directory: Path, source: dataset.Dataset) -> reporting.JsonObject:
    """Create corpus/queries/binary-qrels Parquet files and a checksummed manifest.

    The existing BEIR loader treats every positive grade as binary relevance;
    this export preserves that evaluation semantics, not the original grades.
    No pickle, original-source modification, overwrite or implicit download.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    _validate(source)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    corpus = [
        {
            "doc_id": item.asset_id,
            "source_id": source.source_ids[item.asset_id],
            "title": item.title,
            "text": item.text,
            "metadata_json": snapshots.canonical(dict(item.metadata)),
        }
        for item in source.documents
    ]
    queries = [{"query_id": case.query_id, "text": case.query, "category": case.category} for case in source.cases]
    qrels = [
        {"query_id": case.query_id, "doc_id": identifier, "relevance": 1}
        for case in source.cases
        for identifier in sorted(case.relevant_asset_ids)
    ]
    schemas = {
        "corpus": pa.schema(
            [
                ("doc_id", pa.int64()),
                ("source_id", pa.string()),
                ("title", pa.string()),
                ("text", pa.string()),
                ("metadata_json", pa.string()),
            ]
        ),
        "queries": pa.schema([("query_id", pa.string()), ("text", pa.string()), ("category", pa.string())]),
        "qrels": pa.schema([("query_id", pa.string()), ("doc_id", pa.int64()), ("relevance", pa.int8())]),
    }
    for name, rows in (("corpus", corpus), ("queries", queries), ("qrels", qrels)):
        pq.write_table(
            pa.Table.from_pylist(rows, schema=schemas[name]), directory / f"{name}.parquet", compression="zstd"
        )
    frequencies = collections.Counter(
        term for item in source.documents for term in lexical.tokenize(item.title + " " + item.text)
    )
    manifest = {
        "schema": "lentica-parquet-binary-v1",
        "corpus_sha256": dataset.dataset_identity(source),
        "cases_sha256": dataset.case_identity(source.cases),
        "provenance": source.provenance,
        "sha256": {
            name: snapshots.sha256_file(directory / name)
            for name in ("corpus.parquet", "queries.parquet", "qrels.parquet")
        },
        "report": {
            "documents": len(corpus),
            "queries": len(queries),
            "positive_pairs": len(qrels),
            "ids_contiguous_from_zero": sorted(source.source_ids) == list(range(len(corpus))),
            "mean_tokens": sum(frequencies.values()) / len(corpus),
            "top10_tokens": frequencies.most_common(10),
            "samples": [{"doc_id": item.asset_id, "text_prefix": item.text[:100]} for item in source.documents[:5]],
        },
    }
    with (directory / "manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(snapshots.canonical(manifest) + "\n")
    return manifest


def read(directory: Path, *, max_bytes: int = 128 * 1024 * 1024) -> dataset.Dataset:
    """Reload a selected bounded export, verifying IDs, judgments and identities."""
    import pyarrow.parquet as pq

    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("Parquet byte budget must be a positive integer")
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "lentica-parquet-binary-v1":
        raise ValueError("Unsupported Parquet artifact format")
    tables = {}
    for name, max_rows in (("corpus", 100_000), ("queries", 50_000), ("qrels", 2_000_000)):
        path = directory / f"{name}.parquet"
        if path.stat().st_size > max_bytes or snapshots.sha256_file(path) != manifest["sha256"][path.name]:
            raise ValueError("Parquet artifact exceeds byte budget or differs from its checksum")
        parquet = pq.ParquetFile(path)
        expanded = sum(
            parquet.metadata.row_group(group).column(column).total_uncompressed_size
            for group in range(parquet.metadata.num_row_groups)
            for column in range(parquet.metadata.num_columns)
        )
        if parquet.metadata.num_rows > max_rows or expanded > max_bytes:
            raise ValueError("Expanded Parquet data exceeds evaluation bounds")
        tables[name] = parquet.read().to_pylist()
    corpus = [
        documents.CorpusDocument(row["doc_id"], row["text"], row["title"], json.loads(row["metadata_json"]))
        for row in tables["corpus"]
    ]
    positives: dict[str, set[int]] = collections.defaultdict(set)
    pairs = set()
    query_ids = {row["query_id"] for row in tables["queries"]}
    for row in tables["qrels"]:
        pair = (row["query_id"], row["doc_id"])
        if row["relevance"] != 1 or row["query_id"] not in query_ids or pair in pairs:
            raise ValueError("Invalid or duplicate binary qrel")
        pairs.add(pair)
        positives[row["query_id"]].add(row["doc_id"])
    result = dataset.Dataset(
        corpus,
        [
            evaluation_dataset.EvaluationCase(row["query_id"], row["text"], positives[row["query_id"]], row["category"])
            for row in tables["queries"]
        ],
        {row["doc_id"]: row["source_id"] for row in tables["corpus"]},
        manifest["provenance"],
    )
    _validate(result)
    if (
        dataset.dataset_identity(result) != manifest["corpus_sha256"]
        or dataset.case_identity(result.cases) != manifest["cases_sha256"]
    ):
        raise ValueError("Parquet corpus or query identity mismatch")
    return result
