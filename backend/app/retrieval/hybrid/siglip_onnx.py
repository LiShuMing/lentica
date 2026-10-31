"""Verified local SigLIP dual encoders; independent features, no joint attention."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

from . import artifacts

if TYPE_CHECKING:
    from numpy import typing as npt


class SiglipOnnxEncoder:
    """Native CPU text/image processing for the selected quantized SigLIP export.

    Separate pooled features are normalized for cosine ranking. The original
    learned logit scale/bias is deliberately absent: these scores are not
    calibrated probabilities. Construction reads only explicit local artifacts.
    """

    kind = "siglip"

    def __init__(
        self, directory: Path, *, threads: int = 2, batch_size: int = 8, max_image_pixels: int = 16_000_000
    ) -> None:
        if (
            type(threads) is not int
            or not 1 <= threads <= 32
            or type(batch_size) is not int
            or not 1 <= batch_size <= 64
            or type(max_image_pixels) is not int
            or max_image_pixels <= 0
        ):
            raise ValueError("SigLIP CPU/batch/pixel budgets must be positive and bounded")
        names = (
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "spiece.model",
            "onnx/text_model_quantized.onnx",
            "onnx/vision_model_quantized.onnx",
        )
        files = artifacts.verify(directory, names)
        import onnxruntime as ort
        from transformers import SiglipImageProcessor, SiglipTokenizer

        self.tokenizer = SiglipTokenizer.from_pretrained(str(directory), local_files_only=True)
        self.processor = SiglipImageProcessor.from_pretrained(str(directory), local_files_only=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        self.text_session = ort.InferenceSession(
            str(Path(directory) / names[-2]), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.image_session = ort.InferenceSession(
            str(Path(directory) / names[-1]), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.dimensions = 768
        for session, input_name in ((self.text_session, "input_ids"), (self.image_session, "pixel_values")):
            if {item.name for item in session.get_inputs()} != {input_name}:
                raise ValueError("Unsupported SigLIP input graph contract")
            outputs = {item.name: item for item in session.get_outputs()}
            if "pooler_output" not in outputs or outputs["pooler_output"].shape[-1] != self.dimensions:
                raise ValueError("SigLIP graph must expose aligned pooled features")
        self.batch_size, self.max_image_pixels = batch_size, max_image_pixels
        self.encoding = {
            "model": "siglip-base-patch16-224",
            "export": "separate-quantized-onnx-pooler",
            "max_tokens": 64,
            "padding": "max_length",
            "normalization": "l2-cosine-no-logit-scale-or-bias",
            "device": "cpu",
            "threads": threads,
            "batch_size": batch_size,
            "max_image_pixels": max_image_pixels,
            **{name: metadata.version(name) for name in ("onnxruntime", "transformers", "sentencepiece")},
        }
        self.encoder_id = artifacts.space_id("siglip-onnx-v1", files, self.encoding)

    def encode_texts(self, texts: Sequence[str]) -> npt.NDArray:
        """Use the native SentencePiece tokenizer, EOS and fixed64-token padding.

        A character limit bounds work before tokenization. Empty batches return
        an empty aligned matrix; strings must be supplied inside a sequence.
        """
        import numpy as np

        if isinstance(texts, str) or any(not isinstance(text, str) or len(text) > 32_000 for text in texts):
            raise ValueError("Supply bounded text strings in a sequence")
        rows = []
        for start in range(0, len(texts), self.batch_size):
            tokens = self.tokenizer(
                list(texts[start : start + self.batch_size]),
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="np",
            )
            rows.append(
                self.text_session.run(["pooler_output"], {"input_ids": tokens["input_ids"].astype(np.int64)})[0]
            )
        return self._normalize(rows)

    def encode_query(self, text: str) -> npt.NDArray:
        """One text in the same aligned space as selected image features."""
        return self.encode_texts([text])[0]

    def encode_images(self, paths: Sequence[Path]) -> npt.NDArray:
        """Read only explicit pixels, resize/normalize natively and close images."""
        from PIL import Image

        if isinstance(paths, (str, Path)):
            raise ValueError("Supply a sequence of explicit image paths")
        rows = []
        for start in range(0, len(paths), self.batch_size):
            images = []
            try:
                for path in paths[start : start + self.batch_size]:
                    with Image.open(path) as image:
                        if image.width * image.height > self.max_image_pixels:
                            raise ValueError("Image exceeds pixel budget")
                        images.append(image.convert("RGB"))
                pixels = self.processor(images=images, return_tensors="np")["pixel_values"]
                rows.append(self.image_session.run(["pooler_output"], {"pixel_values": pixels})[0])
            finally:
                for image in images:
                    image.close()
        return self._normalize(rows)

    def encode_image(self, path: Path) -> npt.NDArray:
        """One actual image query; combined-query semantics remain route based."""
        return self.encode_images([path])[0]

    def _normalize(self, rows: Sequence[npt.NDArray]) -> npt.NDArray:
        import numpy as np

        matrix = (
            np.concatenate(rows, axis=0).astype(np.float32)
            if rows
            else np.empty((0, self.dimensions), dtype=np.float32)
        )
        if matrix.ndim != 2 or matrix.shape[1] != self.dimensions or not np.isfinite(matrix).all():
            raise ValueError("SigLIP returned invalid aligned features")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("SigLIP returned zero pooled features")
        return matrix / norms
