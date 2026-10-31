"""Verified local CLAP independent sound/text encoders with native processing."""

from __future__ import annotations

import wave
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

from . import artifacts

if TYPE_CHECKING:
    from numpy import typing as npt


def load_wave(path: Path) -> npt.NDArray:
    """Read explicit48k mono PCM16, at most10seconds; no random crop/resample."""
    import numpy as np

    if Path(path).stat().st_size > 2 * 1024**2:
        raise ValueError("CLAP waveform exceeds2MiB input budget")
    with wave.open(str(path), "rb") as source:
        if (
            source.getnchannels() != 1
            or source.getsampwidth() != 2
            or source.getframerate() != 48000
            or source.getcomptype() != "NONE"
            or not 0 < source.getnframes() <= 480000
        ):
            raise ValueError("CLAP requires explicit nonempty48k mono PCM16, at most10seconds")
        frames = source.getnframes()
        data = source.readframes(frames)
        if len(data) != frames * 2:
            raise ValueError("CLAP waveform payload is truncated")
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


class ClapOnnxEncoder:
    """Native CPU quantized sound/text features; cosine is not event confidence.

    Short clips use native repeatpad. Long clips fail before the feature
    extractor's random truncation can occur. The selected text export has no
    attention-mask input: single-row, unpadded tokens preserve all-real-token
    attention instead of letting batch padding change the query representation.
    """

    kind = "clap"

    def __init__(self, directory: Path, *, threads: int = 2, batch_size: int = 4) -> None:
        if (
            type(threads) is not int
            or not 1 <= threads <= 32
            or type(batch_size) is not int
            or not 1 <= batch_size <= 16
        ):
            raise ValueError("CLAP CPU/batch budgets exceed supported positive limits")
        names = (
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "onnx/audio_model_quantized.onnx",
            "onnx/text_model_quantized.onnx",
        )
        files = artifacts.verify(directory, names)
        import onnxruntime as ort
        from transformers import AutoTokenizer, ClapFeatureExtractor

        self.processor = ClapFeatureExtractor.from_pretrained(str(directory), local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        self.audio_session = ort.InferenceSession(
            str(Path(directory) / names[-2]), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.text_session = ort.InferenceSession(
            str(Path(directory) / names[-1]), sess_options=options, providers=["CPUExecutionProvider"]
        )
        for session, input_name, output_name in (
            (self.audio_session, "input_features", "audio_embeds"),
            (self.text_session, "input_ids", "text_embeds"),
        ):
            if {item.name for item in session.get_inputs()} != {input_name}:
                raise ValueError("Unsupported CLAP native graph inputs")
            outputs = {item.name: item for item in session.get_outputs()}
            if output_name not in outputs or outputs[output_name].shape[-1] != 512:
                raise ValueError("CLAP graph must expose aligned512-dimensional features")
        self.dimensions, self.batch_size = 512, batch_size
        self.encoding = {
            "model": "clap-htsat-unfused",
            "export": "separate-quantized-onnx",
            "device": "cpu",
            "sample_rate": 48000,
            "max_samples": 480000,
            "audio_padding": "native-repeatpad-no-random-crop",
            "max_text_tokens": 512,
            "text_batch_size": 1,
            "text_padding": "none-input-only-export",
            "audio_batch_size": batch_size,
            "threads": threads,
            "normalization": "l2-cosine-no-logit-scale",
            **{name: metadata.version(name) for name in ("onnxruntime", "transformers")},
        }
        self.encoder_id = artifacts.space_id("clap-onnx-v1", files, self.encoding)

    def encode_texts(self, texts: Sequence[str]) -> npt.NDArray:
        """Actually encode native unpadded token sequences one at a time."""
        import numpy as np

        if isinstance(texts, str) or any(not isinstance(text, str) or len(text) > 32000 for text in texts):
            raise ValueError("CLAP text requires a sequence of bounded strings")
        rows = []
        for text in texts:
            tokens = self.tokenizer([text], padding=False, truncation=True, max_length=512, return_tensors="np")
            rows.append(self.text_session.run(["text_embeds"], {"input_ids": tokens["input_ids"].astype(np.int64)})[0])
        return self._normalize(rows)

    def encode_query(self, text: str) -> npt.NDArray:
        """One text query aligned with indexed sound vectors."""
        return self.encode_texts([text])[0]

    def encode_audios(self, paths: Sequence[Path]) -> npt.NDArray:
        """Encode explicit bounded waveforms with native mel/padding parameters."""
        if isinstance(paths, (str, Path)):
            raise ValueError("Supply a sequence of explicit CLAP waveform paths")
        rows = []
        for start in range(0, len(paths), self.batch_size):
            waveforms = [load_wave(path) for path in paths[start : start + self.batch_size]]
            features = self.processor(
                waveforms, sampling_rate=48000, return_tensors="np", padding="repeatpad", truncation="rand_trunc"
            )
            rows.append(self.audio_session.run(["audio_embeds"], {"input_features": features["input_features"]})[0])
        return self._normalize(rows)

    def encode_audio(self, path: Path) -> npt.NDArray:
        """One sound query; does not infer exact event boundaries within a clip."""
        return self.encode_audios([path])[0]

    def _normalize(self, rows: Sequence[npt.NDArray]) -> npt.NDArray:
        import numpy as np

        matrix = (
            np.concatenate(rows, axis=0).astype(np.float32)
            if rows
            else np.empty((0, self.dimensions), dtype=np.float32)
        )
        if matrix.ndim != 2 or matrix.shape[1] != self.dimensions or not np.isfinite(matrix).all():
            raise ValueError("CLAP returned invalid aligned features")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("CLAP returned zero embeddings")
        return matrix / norms
