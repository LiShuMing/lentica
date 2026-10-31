"""Bounded local BEIR/fixture loading and stable dataset identities."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
import re
import zipfile
from collections.abc import Sequence
from pathlib import Path

from backend.app.evaluation import dataset as evaluation_dataset
from backend.app.evaluation.hybrid_eval import reporting
from backend.app.retrieval.hybrid import documents as corpus_documents


@dataclasses.dataclass
class Dataset:
    """A selected corpus, judged queries, source-ID mapping and provenance."""

    documents: list[corpus_documents.CorpusDocument]
    cases: list[evaluation_dataset.EvaluationCase]
    source_ids: dict[int, str]
    provenance: reporting.JsonObject


def load_dataset(path: Path, *, split: str = "test") -> Dataset:
    """Load an explicit local BEIR archive or synthetic JSON fixture.

    Args:
        path: Local file, limited to 32 MiB; no download is performed.
        split: Official train/test qrels member for BEIR archives.

    Returns:
        Validated documents, judged queries, stable source IDs and provenance.
        BEIR uses only positive judgments, treated as binary relevance. Source
        document IDs are sorted as strings before assigning local integer IDs.

    Raises:
        ValueError: Split, archive bounds, IDs or judgments are invalid.
    """
    if split not in {"train", "test"}:
        raise ValueError("Unsupported benchmark split; choose train or test")
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Benchmark inputs are limited to 32 MiB")
    # Bound the compressed input here and each selected expanded member below;
    # a small zip file alone does not imply a small decoded corpus.
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.suffix == ".zip":
        return _load_beir(path, split, fingerprint)
    return _load_fixture(path, fingerprint)


def _read_archive(
    path: Path, split: str
) -> tuple[list[reporting.JsonObject], list[reporting.JsonObject], list[dict[str, str]], dict[str, str]]:
    """Read only the three bounded, uniquely named archive members."""
    with zipfile.ZipFile(path) as archive:
        names = {}
        judgments_member = f"qrels/{split}.tsv"
        for suffix in ("corpus.jsonl", "queries.jsonl", judgments_member):
            matches = [name for name in archive.namelist() if name.endswith("/" + suffix) or name == suffix]
            if len(matches) != 1:
                raise ValueError(f"Expected one archive member ending in {suffix}")
            name = matches[0]
            if archive.getinfo(name).file_size > 32 * 1024 * 1024:
                raise ValueError("Expanded archive member exceeds 32 MiB")
            names[suffix] = name
        with archive.open(names["corpus.jsonl"]) as stream:
            corpus = [json.loads(line) for line in stream if line.strip()]
        with archive.open(names["queries.jsonl"]) as stream:
            query_rows = [json.loads(line) for line in stream if line.strip()]
        with archive.open(names[judgments_member]) as stream:
            judgments = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8"), delimiter="\t"))
    return corpus, query_rows, judgments, names


def _load_beir(path: Path, split: str, fingerprint: str) -> Dataset:
    """Map stable source IDs and positive BEIR judgments into local IDs."""
    corpus, query_rows, judgments, names = _read_archive(path, split)
    corpus = sorted(corpus, key=lambda item: str(item["_id"]))
    mapping = {str(item["_id"]): index for index, item in enumerate(corpus)}
    if len(mapping) != len(corpus):
        raise ValueError("Duplicate source document IDs")
    documents = [
        corpus_documents.CorpusDocument(index, item["text"], item.get("title", "")) for index, item in enumerate(corpus)
    ]
    queries = {str(item["_id"]): item["text"] for item in query_rows}
    if len(queries) != len(query_rows):
        raise ValueError("Duplicate query IDs")
    relevant = {}
    for row in judgments:
        if int(row["score"]) > 0:
            relevant.setdefault(str(row["query-id"]), set()).add(mapping[str(row["corpus-id"])])
    cases = [
        evaluation_dataset.EvaluationCase(
            identifier,
            queries[identifier],
            ids,
            "numeric_or_identifier" if re.search(r"\d|\b[A-Z]{2,}\b", queries[identifier]) else "natural_language",
        )
        for identifier, ids in sorted(relevant.items())
    ]
    source_ids = {index: str(item["_id"]) for index, item in enumerate(corpus)}
    provenance = {
        "kind": f"BEIR {split} split",
        "archive_sha256": fingerprint,
        "members": names,
        "query_categories": "automatic surface-form heuristic",
    }
    return _validated_dataset(documents, cases, source_ids, provenance)


def _load_fixture(path: Path, fingerprint: str) -> Dataset:
    """Read the existing explicit JSON fixture shape."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = [corpus_documents.CorpusDocument(**item) for item in payload["documents"]]
    cases = [
        evaluation_dataset.EvaluationCase(
            item["query_id"], item["query"], set(item["relevant_asset_ids"]), item.get("category", "fixture")
        )
        for item in payload["queries"]
    ]
    source_ids = {item.asset_id: str(item.asset_id) for item in documents}
    provenance = {"kind": "local synthetic fixture", "fixture_sha256": fingerprint}
    return _validated_dataset(documents, cases, source_ids, provenance)


def _validated_dataset(
    documents: list[corpus_documents.CorpusDocument],
    cases: list[evaluation_dataset.EvaluationCase],
    source_ids: dict[int, str],
    provenance: reporting.JsonObject,
) -> Dataset:
    """Check shared corpus/query invariants before exposing a dataset."""
    valid_ids = {document.asset_id for document in documents}
    if not documents or not cases or len(valid_ids) != len(documents) or len(documents) > 10000 or len(cases) > 5000:
        raise ValueError("Require unique documents and nonempty bounded corpus/query judgments")
    if len({case.query_id for case in cases}) != len(cases):
        raise ValueError("Query IDs must be unique")
    for case in cases:
        if not case.relevant_asset_ids or not case.relevant_asset_ids <= valid_ids:
            raise ValueError("Every query needs relevance judgments referring to the selected corpus")
    return Dataset(documents, cases, source_ids, provenance)


def dataset_identity(dataset: Dataset) -> str:
    """Hash ordered document IDs, source IDs and title/body without queries."""
    return reporting.digest(
        [(doc.asset_id, dataset.source_ids[doc.asset_id], doc.title, doc.text) for doc in dataset.documents]
    )


def case_identity(cases: Sequence[evaluation_dataset.EvaluationCase]) -> str:
    """Hash ordered query text and sorted positive judgments."""
    return reporting.digest([(case.query_id, case.query, sorted(case.relevant_asset_ids)) for case in cases])
