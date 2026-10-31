"""Local-only native BGE/E5 encoders, separate from historical product spaces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

from . import artifacts

if TYPE_CHECKING:
    from numpy import typing as npt


@dataclass(frozen=True)
class TextProfile:
    """Native family semantics, including query/document asymmetry and truncation."""

    family: str = "bge-v1.5-en"
    max_tokens: int = 512
    batch_size: int = 16

    def __post_init__(self) -> None:
        if self.family not in {"bge-v1.5-en", "bge-v1.5-zh", "bge-m3", "e5"}:
            raise ValueError("Select an explicitly supported native text model profile")
        if type(self.max_tokens) is not int or not 8 <= self.max_tokens <= 8192:
            raise ValueError("Token budget must be between 8 and 8192")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 128:
            raise ValueError("Batch size must be between 1 and 128")

    @property
    def pooling(self) -> str:
        return "mean-mask" if self.family == "e5" else "cls"

    @property
    def query_prefix(self) -> str:
        return {
            "bge-v1.5-en": "Represent this sentence for searching relevant passages: ",
            "bge-v1.5-zh": "为这个句子生成表示以用于检索相关文章：",
            "bge-m3": "",
            "e5": "query: ",
        }[self.family]

    @property
    def document_prefix(self) -> str:
        return "passage: " if self.family == "e5" else ""


class NativeTextEncoder:
    """CPU float32 encoder with native CLS or masked-mean pooling and L2 output.

    The model is explicitly selected, never resolved from a global cache or
    downloaded at construction. Unlike the historical BgeTextEmbedder, BGE uses
    its native first-token pooling here. Encoder IDs deliberately differ.
    """

    def __init__(self, directory: Path, *, profile: TextProfile | None = None) -> None:
        self.profile = profile or TextProfile()
        files = artifacts.verify(
            directory,
            (
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.txt",
            ),
        )
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
        self.model = (
            AutoModel.from_pretrained(
                str(directory), local_files_only=True, trust_remote_code=False, use_safetensors=True
            )
            .to("cpu")
            .float()
            .eval()
        )
        if self.profile.max_tokens > self.model.config.max_position_embeddings:
            raise ValueError("Token budget exceeds the selected model's positional capacity")
        self.dimensions = int(self.model.config.hidden_size)
        self.encoding = {
            **asdict(self.profile),
            "pooling": self.profile.pooling,
            "query_prefix": self.profile.query_prefix,
            "document_prefix": self.profile.document_prefix,
            "normalization": "l2",
            "device": "cpu",
            "dtype": "float32",
            "torch": metadata.version("torch"),
            "transformers": metadata.version("transformers"),
        }
        self.encoder_id = artifacts.space_id("native-text-v1", files, self.encoding)

    def encode_documents(self, texts: Sequence[str]) -> npt.NDArray:
        """Encode document bodies with the profile's document prefix."""
        return self._encode(texts, self.profile.document_prefix)

    def encode_queries(self, texts: Sequence[str]) -> npt.NDArray:
        """Encode queries with an explicit family-specific retrieval instruction."""
        return self._encode(texts, self.profile.query_prefix)

    def encode_query(self, text: str) -> npt.NDArray:
        """Compatibility callable for exact/ANN text retrievers in this new space."""
        return self.encode_queries([text])[0]

    def _encode(self, texts: Sequence[str], prefix: str) -> npt.NDArray:
        import numpy as np

        if isinstance(texts, str) or any(not isinstance(text, str) for text in texts):
            raise ValueError("Encode a sequence of text strings")
        result = []
        for start in range(0, len(texts), self.profile.batch_size):
            batch = [prefix + text for text in texts[start : start + self.profile.batch_size]]
            tokens = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=self.profile.max_tokens
            )
            with self.torch.inference_mode():
                hidden = self.model(**tokens).last_hidden_state
                if self.profile.pooling == "cls":
                    pooled = hidden[:, 0]
                else:
                    mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                features = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            result.append(features.cpu().numpy())
        return np.concatenate(result, axis=0) if result else np.empty((0, self.dimensions), dtype=np.float32)
