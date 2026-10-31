"""JSON-factory-compatible native providers for disposable or warm execution."""

from __future__ import annotations

from pathlib import Path

from . import ann, contracts, semantic


class NativeHnswProvider:
    """Construct one local BGE/index pair inside its owning CPU worker."""

    def __init__(self, model_directory: str, index_directory: str, *, threads: int = 2) -> None:
        if type(threads) is not int or not 1 <= threads <= 32:
            raise ValueError("Native provider CPU threads must lie within 1..32")
        import torch

        torch.set_num_threads(threads)
        self.encoder = semantic.NativeTextEncoder(Path(model_directory))
        self.index = ann.HnswRetriever(
            Path(index_directory), self.encoder.encode_query, query_encoder_id=self.encoder.encoder_id
        )

    def retrieve(self, query: contracts.SearchQuery, limit: int) -> tuple[int, ...]:
        """Actually encode the query and search the shared pre-window filter domain."""
        return self.index.retrieve(query, limit)
