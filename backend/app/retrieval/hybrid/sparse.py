"""Optional corpus-fitted sparse retrieval, independent of the search core."""

from __future__ import annotations

from collections.abc import Sequence

from . import contracts
from . import documents as corpus


class SparseTfidfRetriever:
    """Cosine similarity without a lossy dense projection or model downloads.

    Word ngrams preserve rare exact terms; char_wb ngrams can share evidence
    across spelling/inflection variants. Neither is a pretrained semantic model.
    """

    def __init__(
        self,
        documents: Sequence[corpus.CorpusDocument],
        *,
        analyzer: str = "word",
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int = 60000,
    ) -> None:
        """Fit the sparse representation on corpus text only.

        Args:
            documents: Nonempty corpus with unique IDs and optional filter metadata.
            analyzer: Word tokens or char_wb character fragments. char_wb pads
                word boundaries and constructs fragments within each word.
            ngram_range: Inclusive lower/upper lengths in words or characters,
                depending on the analyzer; (1, 2) includes words and word pairs.
            max_features: Vocabulary budget ranked by corpus term frequency;
                min_df=1 makes singleton terms eligible, not guaranteed to survive.

        Raises:
            ValueError: Corpus, analyzer, feature bounds or vocabulary are invalid.
        """
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        if analyzer not in {"word", "char_wb"}:
            raise ValueError("Sparse analyzer must be word or char_wb")
        if (
            len(ngram_range) != 2
            or any(type(value) is not int for value in ngram_range)
            or not 1 <= ngram_range[0] <= ngram_range[1] <= 8
            or type(max_features) is not int
            or not 1 <= max_features <= 1000000
        ):
            raise ValueError("Invalid sparse ngram/feature bounds")
        self.ids = tuple(document.asset_id for document in documents)
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError("Sparse corpus requires nonempty unique document IDs")
        self.metadata = {document.asset_id: dict(document.metadata) for document in documents}
        # Logarithmic TF (1 + log(count)) limits repeated-term dominance. L2
        # normalization makes a sparse dot product equal cosine similarity.
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=1,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
            **({"token_pattern": r"(?u)\b\w+\b"} if analyzer == "word" else {}),
        )
        self.matrix = self.vectorizer.fit_transform([document.title + " " + document.text for document in documents])

    def retrieve(self, query: contracts.SearchQuery, limit: int) -> tuple[int, ...]:
        """Return positive cosine matches after filtering, with stable ID ties.

        Queries use the corpus vocabulary/IDF without fitting new features.
        Out-of-vocabulary queries return nothing; unlike the dense adapter,
        documents with zero evidence never fill unused candidate positions.
        """
        if limit <= 0:
            return ()
        vector = self.vectorizer.transform([query.text])
        if not vector.nnz:
            return ()
        selected = [
            index
            for index, identifier in enumerate(self.ids)
            if query.filters.matches(identifier, self.metadata[identifier])
        ]
        if not selected:
            return ()
        # Reuse the CSR matrix for unfiltered queries instead of copying its
        # entire vocabulary-sized storage through fancy row indexing.
        matrix = self.matrix if len(selected) == len(self.ids) else self.matrix[selected]
        scores = (matrix @ vector.T).tocoo()
        ranked = sorted(
            ((self.ids[selected[row]], float(score)) for row, score in zip(scores.row, scores.data) if score > 0),
            key=lambda item: (-item[1], item[0]),
        )
        return tuple(identifier for identifier, _ in ranked[:limit])
