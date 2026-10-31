"""Explicit corpus-fitted latent semantic encoding with lazy numeric imports."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from importlib import metadata
from typing import TYPE_CHECKING

from backend.app.retrieval.hybrid import documents as corpus

if TYPE_CHECKING:
    from numpy import typing as npt


class LatentSemanticEncoder:
    """Corpus-fitted TF-IDF + truncated SVD, not a pretrained language model."""

    def __init__(self, documents: Sequence[corpus.CorpusDocument], *, dimensions: int = 128, seed: int = 42) -> None:
        """Fit TF-IDF and SVD on documents without using queries or judgments.

        Args:
            documents: Corpus whose title/body define the fitted vocabulary.
            dimensions: Requested latent dimensions, capped at one less than
                each matrix axis; the data's effective rank can be smaller.
            seed: Fixed random seed for truncated SVD.

        Raises:
            ValueError: The corpus cannot support a nonempty latent space.
        """
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        # English stop words and rare-term pruning are specific to this LSA
        # baseline, not shared with BM25 or the unprojected sparse routes.
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=1 if len(documents) < 20 else 2,
            max_df=1.0,
            max_features=30000,
            ngram_range=(1, 2),
        )
        matrix = self.vectorizer.fit_transform([doc.title + " " + doc.text for doc in documents])
        dimensions = min(dimensions, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if dimensions < 1:
            raise ValueError("Latent semantic evaluation needs at least two documents and two terms")
        # Seven randomized-SVD refinement iterations and a fixed seed make this
        # baseline repeatable in the recorded numeric environment. Explained
        # variance measures retained corpus variation, not retrieval accuracy.
        self.svd = TruncatedSVD(n_components=dimensions, n_iter=7, random_state=seed)
        self.vectors = normalize(self.svd.fit_transform(matrix))
        # Preserve document order and JSON serialization: both identify the
        # fitted space and are part of existing encoder IDs/selection artifacts.
        corpus_hash = hashlib.sha256(
            json.dumps([(doc.asset_id, doc.title, doc.text) for doc in documents], ensure_ascii=False).encode()
        ).hexdigest()
        self.encoder_id = f"corpus-lsa:{corpus_hash}:{dimensions}:{seed}:sklearn-{metadata.version('scikit-learn')}"
        self.dimensions = dimensions
        self.explained_variance = float(self.svd.explained_variance_ratio_.sum())

    def encode_query(self, text: str) -> npt.NDArray:
        """Transform one query with the already fitted vocabulary and projection.

        The dense retriever normalizes this projection before cosine scoring;
        a query with no vocabulary evidence projects to a zero vector.
        """
        return self.svd.transform(self.vectorizer.transform([text]))[0]
