"""Optional persistent HNSW cosine index with same-domain exact fallbacks."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from . import contracts, snapshots, vector

if TYPE_CHECKING:
    from numpy import typing as npt


class HnswRetriever(vector.ExactVectorRetriever):
    """Use approximate HNSW on large domains, exact scan on selective filters.

    Vectors are normalized; native inner-product scores never leave the adapter.
    Index bytes are trusted local build artifacts, not arbitrary external FAISS
    files. Checksums detect accidental changes, not a malicious index producer.
    Runtime search parameters are per-call, without mutating shared efSearch.
    """

    def __init__(self, directory: Path, encode_query: Callable[[str], npt.ArrayLike], *, query_encoder_id: str) -> None:
        """Load a verified local index and its normalized exact reference matrix."""
        import faiss
        import numpy as np

        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != "lentica-hnsw-v1":
            raise ValueError("Unsupported HNSW artifact format")
        for name in ("vectors.npy", "ids.npy", "index.faiss"):
            if snapshots.sha256_file(directory / name) != manifest["sha256"][name]:
                raise ValueError("HNSW artifact digest mismatch")
        ids = np.load(directory / "ids.npy", allow_pickle=False).tolist()
        super().__init__(
            ids,
            np.load(directory / "vectors.npy", allow_pickle=False),
            encode_query,
            encoder_id=manifest["encoder_id"],
            query_encoder_id=query_encoder_id,
            metadata={int(key): value for key, value in manifest["metadata"].items()},
        )
        self.index = faiss.read_index(str(directory / "index.faiss"))
        if self.index.ntotal != len(self.ids) or self.index.d != self.matrix.shape[1]:
            raise ValueError("HNSW index dimensions/count differ from the reference matrix")
        self.ef_search = manifest["ef_search"]
        self.exact_filter_threshold = manifest["exact_filter_threshold"]
        self.source_version = manifest["source_version"]

    @classmethod
    def build(
        cls,
        directory: Path,
        ids: Sequence[int],
        vectors: npt.ArrayLike,
        encode_query: Callable[[str], npt.ArrayLike],
        *,
        encoder_id: str,
        metadata: Mapping[int, Mapping[str, object]] | None = None,
        source_version: str,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
        exact_filter_threshold: int = 256,
    ) -> HnswRetriever:
        """Build a fresh HNSW artifact with bounded parameters and explicit identity."""
        import faiss
        import numpy as np

        if not source_version or any(type(value) is not int or value <= 0 for value in (m, ef_construction, ef_search)):
            raise ValueError("Source identity and positive integer HNSW parameters required")
        if (
            m > 128
            or ef_construction > 10000
            or ef_search > 10000
            or type(exact_filter_threshold) is not int
            or exact_filter_threshold < 0
        ):
            raise ValueError("HNSW parameters exceed supported build/query budgets")
        reference = vector.ExactVectorRetriever(
            ids, vectors, encode_query, encoder_id=encoder_id, query_encoder_id=encoder_id, metadata=metadata
        )
        if any(identifier > 2**63 - 1 for identifier in reference.ids):
            raise ValueError("Persistent index IDs must fit signed int64")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        index = faiss.IndexHNSWFlat(reference.matrix.shape[1], m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.add(reference.matrix)
        faiss.write_index(index, str(directory / "index.faiss"))
        np.save(directory / "vectors.npy", reference.matrix, allow_pickle=False)
        np.save(directory / "ids.npy", np.asarray(reference.ids, dtype=np.int64), allow_pickle=False)
        payload = {
            "schema": "lentica-hnsw-v1",
            "encoder_id": encoder_id,
            "source_version": source_version,
            "metadata": reference.metadata,
            "m": m,
            "ef_construction": ef_construction,
            "ef_search": ef_search,
            "exact_filter_threshold": exact_filter_threshold,
            "sha256": {
                name: snapshots.sha256_file(directory / name) for name in ("vectors.npy", "ids.npy", "index.faiss")
            },
        }
        with (directory / "manifest.json").open("x", encoding="utf-8") as stream:
            stream.write(
                snapshots.canonical(
                    {**payload, "metadata": {str(key): value for key, value in reference.metadata.items()}}
                )
                + "\n"
            )
        return cls(directory, encode_query, query_encoder_id=encoder_id)

    def retrieve(self, query: contracts.SearchQuery, limit: int) -> tuple[int, ...]:
        """Search the eligible domain; enforce ID ties at a truncated boundary.

        IDSelectorBatch uses matrix row numbers, not external unit IDs. Very
        selective domains and unresolved score-boundary ties use exact cosine.
        Negative/zero similarities remain eligible as in the exact reference.
        """
        import faiss
        import numpy as np

        if limit <= 0:
            return ()
        selected = [
            row
            for row, identifier in enumerate(self.ids)
            if query.filters.matches(identifier, self.metadata[identifier])
        ]
        if not selected:
            return ()
        encoded = np.asarray(self.encode_query(query.text), dtype=np.float32)
        if encoded.shape != (self.matrix.shape[1],) or not np.isfinite(encoded).all():
            raise ValueError("Query vector dimension or finite-value contract violated")
        norm = np.linalg.norm(encoded.astype(np.float64))
        if not norm:
            return ()
        encoded = np.ascontiguousarray(encoded / norm)
        if len(selected) <= max(limit, self.exact_filter_threshold):
            return self._exact(encoded, selected, limit)
        parameters = faiss.SearchParametersHNSW()
        parameters.efSearch = self.ef_search
        selected_rows = np.asarray(selected, dtype=np.int64)
        selector = faiss.IDSelectorBatch(selected_rows)
        parameters.sel = selector
        window = min(len(selected), max(limit * 2, self.ef_search))
        scores, rows = self.index.search(encoded.reshape(1, -1), window, params=parameters)
        admissible = set(selected)
        ranked = []
        for score, row in zip(scores[0], rows[0]):
            if row < 0:
                continue
            if int(row) not in admissible:
                raise ValueError("ANN provider violated the pre-window filter")
            ranked.append((float(score), self.ids[int(row)]))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        if len(ranked) < limit or (
            len(ranked) == window and window < len(selected) and ranked[-1][0] == ranked[min(limit, len(ranked)) - 1][0]
        ):
            return self._exact(encoded, selected, limit)
        return tuple(identifier for _, identifier in ranked[:limit])

    def _exact(self, encoded: npt.ArrayLike, selected: Sequence[int], limit: int) -> tuple[int, ...]:
        scores = self.matrix[list(selected)] @ encoded
        ranked = sorted(zip(selected, scores), key=lambda item: (-float(item[1]), self.ids[item[0]]))
        return tuple(self.ids[row] for row, _ in ranked[:limit])
