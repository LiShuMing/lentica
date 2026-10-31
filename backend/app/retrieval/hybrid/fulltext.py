"""Explicit persistent Tantivy BM25 adapter with pre-window filter semantics."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from . import contracts, documents, lexical, snapshots


class TantivyRetriever:
    """Persistent reference-tokenized BM25, independent of the default product.

    Tantivy owns native scores. All metadata/allowed-ID conditions become an ID
    term-set conjunction before collection, with zero filter score. Equal score
    boundaries expand until ties can be ordered by stable external integer ID.
    Numeric fields use signed int64: this binding converts Python ints to I64,
    so an unsigned field would silently fail to match otherwise valid filters.
    """

    def __init__(self, directory: Path) -> None:
        """Reopen an explicitly built index, without tokenizing original sources."""
        import tantivy

        self.tantivy = tantivy
        self.directory = Path(directory)
        manifest = json.loads((self.directory / "lentica.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != "lentica-tantivy-v1":
            raise ValueError("Unsupported Tantivy artifact format")
        if snapshots.sha256_file(self.directory / "meta.json") != manifest["meta_sha256"]:
            raise ValueError("Tantivy metadata digest mismatch")
        self.metadata = {contracts.asset_id(int(key)): value for key, value in manifest["metadata"].items()}
        self.index = tantivy.Index.open(str(self.directory))
        self.schema = self.index.schema
        self.searcher = self.index.searcher()
        self.source_version = manifest["source_version"]

    @classmethod
    def build(
        cls, directory: Path, corpus: Sequence[documents.CorpusDocument], *, source_version: str
    ) -> TantivyRetriever:
        """Write a fresh full-text index and metadata; never overwrite an index.

        Text/title are indexed together using the reference word/CJK tokens and
        whitespace analyzer; no stemming. The index stores only external IDs,
        not final document bodies. Default engine BM25 parameters are 1.2/.75.
        """
        import tantivy

        if not source_version or len({item.asset_id for item in corpus}) != len(corpus):
            raise ValueError("Corpus IDs must be unique and source version explicit")
        if any(item.asset_id > 2**63 - 1 for item in corpus):
            raise ValueError("Persistent index IDs must fit signed int64")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        builder = tantivy.SchemaBuilder()
        builder.add_integer_field("id", stored=True, indexed=True)
        builder.add_text_field("terms", tokenizer_name="whitespace")
        index = tantivy.Index(builder.build(), path=str(directory))
        writer = index.writer(heap_size=15_000_000, num_threads=1)
        for item in sorted(corpus, key=lambda document: document.asset_id):
            document = tantivy.Document()
            document.add_integer("id", item.asset_id)
            document.add_text("terms", " ".join(lexical.tokenize(item.title + " " + item.text)))
            writer.add_document(document)
        writer.commit()
        writer.wait_merging_threads()
        payload = {
            "schema": "lentica-tantivy-v1",
            "source_version": source_version,
            "metadata": {str(item.asset_id): dict(item.metadata) for item in corpus},
            "meta_sha256": snapshots.sha256_file(directory / "meta.json"),
        }
        with (directory / "lentica.json").open("x", encoding="utf-8") as stream:
            stream.write(snapshots.canonical(payload) + "\n")
        return cls(directory)

    def retrieve(self, query: contracts.SearchQuery, limit: int) -> tuple[int, ...]:
        """Conjoin the complete eligible ID domain before collecting bounded hits."""
        return tuple(identifier for identifier, _ in self.retrieve_scored(query, limit))

    def retrieve_scored(self, query: contracts.SearchQuery, limit: int) -> tuple[tuple[int, float], ...]:
        """Expose native BM25 only for shard diagnostics, never cross-route fusion."""
        if limit <= 0:
            return ()
        terms = lexical.tokenize(query.text)
        eligible = [
            identifier for identifier, metadata in self.metadata.items() if query.filters.matches(identifier, metadata)
        ]
        if not terms or not eligible:
            return ()
        text = self.tantivy.Query.boolean_query(
            [(self.tantivy.Occur.Should, self.tantivy.Query.term_query(self.schema, "terms", term)) for term in terms]
        )
        if len(eligible) != len(self.metadata):
            allowed = self.tantivy.Query.term_set_query(self.schema, "id", eligible)
            text = self.tantivy.Query.boolean_query(
                [
                    (self.tantivy.Occur.Must, text),
                    (self.tantivy.Occur.Must, self.tantivy.Query.const_score_query(allowed, 0.0)),
                ]
            )
        return self._collect(text, limit, len(eligible))

    def _collect(self, query: object, limit: int, eligible_count: int) -> tuple[tuple[int, float], ...]:
        window = min(eligible_count, max(32, limit * 2))
        while True:
            result = self.searcher.search(query, limit=window)
            ranked = sorted(
                ((float(score), self.searcher.doc(address)["id"][0]) for score, address in result.hits),
                key=lambda item: (-item[0], item[1]),
            )
            if len(ranked) <= limit or len(ranked) == result.count or ranked[-1][0] < ranked[limit - 1][0]:
                return tuple((identifier, score) for score, identifier in ranked[:limit])
            window = min(eligible_count, window * 2)


def filtered_domain(metadata: Mapping[int, Mapping[str, object]], filters: contracts.SearchFilter) -> tuple[int, ...]:
    """Expose the exact eligible domain for parity diagnostics, never post-filter."""
    return tuple(sorted(identifier for identifier, values in metadata.items() if filters.matches(identifier, values)))
