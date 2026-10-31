"""Exact dense cosine retrieval with explicit encoder-space compatibility."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

from . import contracts

if TYPE_CHECKING:
    from numpy import typing as npt


class ExactVectorRetriever:
    """Rank filtered dense vectors by normalized cosine similarity."""

    def __init__(
        self,
        ids: Sequence[int],
        vectors: npt.ArrayLike,
        encode_query: Callable[[str], npt.ArrayLike],
        *,
        encoder_id: str,
        query_encoder_id: str,
        metadata: Mapping[int, Mapping[str, object]] | None = None,
    ) -> None:
        """Snapshot and normalize ID-aligned document vectors.

        Args:
            ids: Unique asset IDs aligned with matrix rows.
            vectors: Finite two-dimensional document matrix.
            encode_query: Callable using the same vector space as documents.
            encoder_id: Explicit document encoder-space identity; equal vector
                dimensions alone do not establish comparable coordinates.
            query_encoder_id: Matching query encoder-space identity. The caller
                is responsible for assigning truthful model/version identities.
            metadata: Optional pre-window filter values per ID.

        Raises:
            ValueError: IDs, matrix shape/values or encoder identities are invalid.
        """
        import numpy as np

        if not encoder_id or encoder_id != query_encoder_id:
            raise ValueError("Document and query encoders must identify the same vector space")
        self.ids = tuple(contracts.asset_id(identifier) for identifier in ids)
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Vector IDs must be unique")
        matrix = np.array(vectors, dtype=np.float32, copy=True)
        if matrix.ndim != 2 or matrix.shape[0] != len(ids) or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
            raise ValueError("Vectors require one finite, fixed-dimension row per ID")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Copying isolates the index from caller mutation. Zero document vectors
        # remain zero and receive cosine score zero for nonzero queries.
        self.matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)
        self.matrix.flags.writeable = False
        self.encode_query = encode_query
        self.encoder_id = encoder_id
        self.metadata = {identifier: dict((metadata or {}).get(identifier, {})) for identifier in self.ids}

    def retrieve(self, query: contracts.SearchQuery, limit: int) -> tuple[int, ...]:
        """Return filtered cosine ranks, with no evidence for zero query vectors.

        All eligible documents can fill the window, including zero or negative
        cosine scores. This adapter imposes no minimum-similarity threshold.

        Raises:
            ValueError: Query encoding violates dimension or finite-value constraints.
        """
        if limit <= 0:
            return ()
        return self.retrieve_vector(self.encode_query(query.text), query.filters, limit)

    def retrieve_vector(self, encoded: npt.ArrayLike, filters: contracts.SearchFilter, limit: int) -> tuple[int, ...]:
        """Rank a caller-encoded query in this explicitly identified vector space.

        Multimodal adapters must verify encoder identity before supplying image
        vectors. Dimension checks alone cannot establish image/text alignment.
        """
        import numpy as np

        if limit <= 0:
            return ()
        vector = np.asarray(encoded, dtype=np.float32)
        if vector.shape != (self.matrix.shape[1],) or not np.isfinite(vector).all():
            raise ValueError("Query vector dimension or finite-value contract violated")
        norm = np.linalg.norm(vector)
        if not norm:
            return ()  # A zero vector cannot define a cosine direction.
        selected = [
            index for index, identifier in enumerate(self.ids) if filters.matches(identifier, self.metadata[identifier])
        ]
        if not selected:
            return ()
        # Score only eligible rows: filtering a global top-k afterward can lose
        # valid results when ineligible documents occupy the candidate window.
        scores = self.matrix[selected] @ (vector / norm)
        ordered = sorted(zip(selected, scores), key=lambda item: (-float(item[1]), self.ids[item[0]]))
        return tuple(self.ids[index] for index, _ in ordered[:limit])
