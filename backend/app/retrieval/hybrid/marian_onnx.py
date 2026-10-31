"""Explicit local Chinese-to-English query adaptation with bounded native tokens."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from . import artifacts


@dataclass(frozen=True)
class Translation:
    """Keep original input separate from generated text and model provenance."""

    original: str
    text: str
    producer: str
    input_tokens: int
    output_tokens: int


class MarianOnnxTranslator:
    """Selected Marian export, greedy CPU decoding without a hidden KV-cache ABI.

    This explicit adaptation never edits a query in place or consults labels.
    Greedy decoding differs from the checkpoint's default six-beam search.
    Exhaustion fails rather than forcing EOS and returning a truncated query.
    """

    def __init__(self, directory: Path, *, threads: int = 2, max_tokens: int = 128) -> None:
        if (
            type(threads) is not int
            or not 1 <= threads <= 32
            or type(max_tokens) is not int
            or not 1 <= max_tokens <= 128
        ):
            raise ValueError("Marian CPU/output budgets exceed their positive limits")
        directory = Path(directory)
        names = (
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "source.spm",
            "target.spm",
            "vocab.json",
            "onnx/encoder_model_quantized.onnx",
            "onnx/decoder_model_quantized.onnx",
        )
        files = artifacts.verify(directory, names)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        generation = json.loads((directory / "generation_config.json").read_text(encoding="utf-8"))
        self._check_profile(config, generation)
        import onnxruntime as ort
        from transformers import MarianTokenizer

        self.tokenizer = MarianTokenizer.from_pretrained(str(directory), local_files_only=True)
        if self.tokenizer.eos_token_id != 0 or self.tokenizer.pad_token_id != 65000:
            raise ValueError("Marian tokenizer special tokens differ from the selected profile")
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        self.encoder = ort.InferenceSession(
            str(directory / names[-2]), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.decoder = ort.InferenceSession(
            str(directory / names[-1]), sess_options=options, providers=["CPUExecutionProvider"]
        )
        if {item.name for item in self.encoder.get_inputs()} != {"input_ids", "attention_mask"} or {
            item.name for item in self.decoder.get_inputs()
        } != {"input_ids", "encoder_attention_mask", "encoder_hidden_states"}:
            raise ValueError("Unsupported Marian encoder/decoder graph inputs")
        for session, output in ((self.encoder, "last_hidden_state"), (self.decoder, "logits")):
            if output not in {item.name for item in session.get_outputs()}:
                raise ValueError("Unsupported Marian graph outputs")
        self.max_tokens = max_tokens
        self.version = artifacts.space_id(
            "marian-zh-en-onnx-v1",
            files,
            {
                "decode": "greedy-no-cache-natural-eos",
                "max_tokens": max_tokens,
                "max_input_tokens": 256,
                "max_characters": 2048,
                "threads": threads,
                **{
                    name: metadata.version(name)
                    for name in ("onnxruntime", "transformers", "sentencepiece", "sacremoses")
                },
            },
        )

    @staticmethod
    def _check_profile(config: dict[str, object], generation: dict[str, object]) -> None:
        """Reject unsupported generation semantics instead of silently ignoring them."""
        expected = {
            "decoder_start_token_id": 65000,
            "eos_token_id": 0,
            "pad_token_id": 65000,
            "bos_token_id": 0,
            "forced_eos_token_id": 0,
            "bad_words_ids": [[65000]],
            "num_beams": 6,
            "max_length": 512,
        }
        if config.get("model_type") != "marian" or config.get("vocab_size") != 65001 or config.get("d_model") != 512:
            raise ValueError("Require the selected Marian Chinese-to-English profile")
        if any(config.get(key) != value or generation.get(key) != value for key, value in expected.items()):
            raise ValueError("Unsupported Marian native generation profile")
        allowed = set(expected) | {"renormalize_logits", "transformers_version"}
        if set(generation) - allowed or generation.get("renormalize_logits") is not True:
            raise ValueError("Unsupported Marian generation options")

    def translate(self, text: str) -> Translation:
        """Generate one bounded query, requiring natural EOS and nonempty output.

        Args:
            text: Explicit original Chinese query; identifiers are never rewritten
                automatically by a retrieval route.

        Returns:
            Original and generated strings plus immutable producer identity.

        Raises:
            ValueError: Invalid input, malformed features, empty output or token
                exhaustion. Provider/runtime failures propagate to the caller.
        """
        import numpy as np

        if not isinstance(text, str) or not text.strip() or len(text) > 2048:
            raise ValueError("Supply a nonempty query of at most2048 characters")
        tokens = self.tokenizer(text, return_tensors="np", truncation=False)
        ids, mask = tokens["input_ids"], tokens["attention_mask"]
        if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= 256 or mask.shape != ids.shape:
            raise ValueError("Query exceeds the256-token input budget or has malformed tokens")
        encoded = self.encoder.run(["last_hidden_state"], {"input_ids": ids, "attention_mask": mask})[0]
        if encoded.shape != (1, ids.shape[1], 512) or not np.isfinite(encoded).all():
            raise ValueError("Marian encoder returned invalid features")
        history = [65000]
        for step in range(self.max_tokens):
            logits = self.decoder.run(
                ["logits"],
                {
                    "input_ids": np.asarray([history], dtype=np.int64),
                    "encoder_attention_mask": mask,
                    "encoder_hidden_states": encoded,
                },
            )[0]
            if logits.shape != (1, len(history), 65001) or not np.isfinite(logits).all():
                raise ValueError("Marian decoder returned invalid logits")
            scores = logits[0, -1].copy()
            scores[65000] = -np.inf
            token = int(scores.argmax())
            if token == 0:
                translated = self.tokenizer.decode(history[1:], skip_special_tokens=True).strip()
                if not translated:
                    raise ValueError("Marian generated empty query text")
                return Translation(text, translated, self.version, ids.shape[1], step + 1)
            history.append(token)
        raise ValueError("Marian exhausted its token budget without EOS; no partial translation returned")
