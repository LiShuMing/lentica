"""Local native cross-encoder pair scoring, separate from rank-fusion scores."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import metadata
from pathlib import Path

from . import artifacts, contracts, units


class PairReranker:
    """Run one verified CPU ONNX text-pair model with its native tokenizer.

    The selected model returns one relevance logit. It is only an ordering score,
    not a calibrated probability or a safe no-answer threshold. Candidate text
    uses native/OCR/transcript evidence; generated captions remain excluded.
    """

    def __init__(
        self,
        directory: Path,
        *,
        threads: int = 2,
        batch_size: int = 16,
        max_tokens: int = 512,
        max_candidates: int = 200,
    ) -> None:
        if (
            any(type(value) is not int or value <= 0 for value in (threads, batch_size, max_tokens, max_candidates))
            or threads > 32
            or batch_size > 64
            or not 8 <= max_tokens <= 512
            or max_candidates > 1000
        ):
            raise ValueError("Pair model budgets exceed supported positive CPU/token/candidate limits")
        names = (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "onnx/model_quantized.onnx",
        )
        files = artifacts.verify(directory, names)
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        self.session = ort.InferenceSession(
            str(Path(directory) / names[-1]), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.inputs = tuple(item.name for item in self.session.get_inputs())
        if not {"input_ids", "attention_mask"} <= set(self.inputs) or set(self.inputs) - {
            "input_ids",
            "attention_mask",
            "token_type_ids",
        }:
            raise ValueError("Unsupported native pair tokenizer/graph input contract")
        self.batch_size, self.max_tokens, self.max_candidates = batch_size, max_tokens, max_candidates
        self.version = artifacts.space_id(
            "native-pair-onnx-v1",
            files,
            {
                "threads": threads,
                "batch_size": batch_size,
                "max_tokens": max_tokens,
                "max_candidates": max_candidates,
                "output": "single-relevance-logit",
                "onnxruntime": metadata.version("onnxruntime"),
                "transformers": metadata.version("transformers"),
            },
        )

    def score(self, query: str, texts: Sequence[str]) -> tuple[float, ...]:
        """Score bounded actual query/document pairs; never accept relevance labels.

        Native longest-first pair truncation is explicit. Input character bounds
        prevent unbounded pre-tokenization allocations; token budget applies to
        the combined pair including special tokens.
        """
        import numpy as np

        if (
            not isinstance(query, str)
            or not query.strip()
            or len(query) > 32_000
            or isinstance(texts, str)
            or len(texts) > self.max_candidates
            or any(not isinstance(text, str) or len(text) > 200_000 for text in texts)
        ):
            raise ValueError("Pair scoring input exceeds text/candidate budgets")
        scores = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            tokens = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation="longest_first",
                max_length=self.max_tokens,
                return_tensors="np",
            )
            inputs = {name: tokens[name].astype(np.int64) for name in self.inputs}
            logits = self.session.run(None, inputs)[0]
            if logits.shape != (len(batch), 1) or not np.isfinite(logits).all():
                raise ValueError("Pair model must return one finite logit per candidate")
            scores.extend(float(value) for value in logits[:, 0])
        return tuple(scores)

    def rerank(self, query: contracts.SearchQuery, candidates: Sequence[units.RetrievalUnit]) -> tuple[int, ...]:
        """Return the complete candidate permutation, with input-order ties."""
        ids = [unit.unit_id for unit in candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("Pair reranking requires unique candidate IDs")
        scores = self.score(query.text, [unit.text() for unit in candidates])
        return tuple(ids[index] for index in sorted(range(len(ids)), key=lambda index: (-scores[index], index)))
