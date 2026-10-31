"""Explicit local CPU CLIP ONNX text/image encoders with native preprocessing."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

from . import artifacts

if TYPE_CHECKING:
    from numpy import typing as npt


class ClipOnnxEncoder:
    """Aligned text/image features from verified separate CLIP ONNX graphs.

    The selected export is quantized; it is not numerically equivalent to the
    original float32 checkpoint. That identity is recorded rather than hidden.
    Construction reads only the explicit local directory, using CPU providers.
    """

    def __init__(
        self, directory: Path, *, threads: int = 2, batch_size: int = 8, max_image_pixels: int = 16_000_000
    ) -> None:
        if (
            type(threads) is not int
            or not 1 <= threads <= 32
            or type(batch_size) is not int
            or not 1 <= batch_size <= 64
        ):
            raise ValueError("CPU threads and batch sizes must be explicitly bounded")
        if type(max_image_pixels) is not int or max_image_pixels <= 0:
            raise ValueError("Image pixel budget must be a positive integer")
        names = (
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "merges.txt",
            "vocab.json",
            "onnx/text_model_quantized.onnx",
            "onnx/vision_model_quantized.onnx",
        )
        files = artifacts.verify(directory, names)
        import onnxruntime as ort
        from transformers import CLIPImageProcessor, CLIPTokenizerFast

        self.tokenizer = CLIPTokenizerFast.from_pretrained(str(directory), local_files_only=True)
        self.processor = CLIPImageProcessor.from_pretrained(str(directory), local_files_only=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        self.text_session = ort.InferenceSession(
            str(Path(directory) / names[-2]), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.image_session = ort.InferenceSession(
            str(Path(directory) / names[-1]), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.batch_size, self.max_image_pixels = batch_size, max_image_pixels
        self.dimensions = 512
        self.encoding = {
            "model": "clip-vit-base-patch32",
            "export": "separate-quantized-onnx",
            "max_tokens": 77,
            "normalization": "l2",
            "device": "cpu",
            "threads": threads,
            "batch_size": batch_size,
            "max_image_pixels": max_image_pixels,
            "onnxruntime": metadata.version("onnxruntime"),
            "transformers": metadata.version("transformers"),
        }
        self.encoder_id = artifacts.space_id("clip-onnx-v1", files, self.encoding)

    def encode_texts(self, texts: Sequence[str]) -> npt.NDArray:
        """Tokenize with native CLIP truncation/padding and return normalized features."""
        import numpy as np

        if isinstance(texts, str) or any(not isinstance(text, str) for text in texts):
            raise ValueError("Supply a sequence of text strings")
        rows = []
        for start in range(0, len(texts), self.batch_size):
            tokens = self.tokenizer(
                list(texts[start : start + self.batch_size]),
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="np",
            )
            rows.append(self.text_session.run(["text_embeds"], {"input_ids": tokens["input_ids"].astype(np.int64)})[0])
        return self._normalize(rows)

    def encode_query(self, text: str) -> npt.NDArray:
        """One text query in the same aligned space as image representations."""
        return self.encode_texts([text])[0]

    def encode_images(self, paths: Sequence[Path]) -> npt.NDArray:
        """Use actual selected pixels and native resize/crop/normalize processing."""
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
                rows.append(self.image_session.run(["image_embeds"], {"pixel_values": pixels})[0])
            finally:
                for image in images:
                    image.close()
        return self._normalize(rows)

    def encode_image(self, path: Path) -> npt.NDArray:
        """One image query; combination with text belongs to rank-level routes."""
        return self.encode_images([path])[0]

    def _normalize(self, rows: Sequence[npt.NDArray]) -> npt.NDArray:
        import numpy as np

        matrix = (
            np.concatenate(rows, axis=0).astype(np.float32)
            if rows
            else np.empty((0, self.dimensions), dtype=np.float32)
        )
        if matrix.ndim != 2 or matrix.shape[1] != self.dimensions or not np.isfinite(matrix).all():
            raise ValueError("CLIP graph returned invalid aligned features")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)
