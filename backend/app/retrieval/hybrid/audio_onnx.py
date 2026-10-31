"""Local CPU Whisper greedy transcription of explicitly bounded PCM waveforms."""

from __future__ import annotations

import json
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

from . import artifacts, contracts, units

if TYPE_CHECKING:
    from numpy import typing as npt


@dataclass(frozen=True)
class SpeechSegment:
    """Coarse chunk bounds, not estimated word/speech boundaries or confidence."""

    start_seconds: float
    end_seconds: float
    text: str


def load_wave(path: Path, *, max_seconds: int = 120) -> npt.NDArray:
    """Read explicit16kHz mono signed PCM16 WAV; reject excess rather than clip."""
    if type(max_seconds) is not int or not 1 <= max_seconds <= 120 or Path(path).stat().st_size > 4 * 1024**2:
        raise ValueError("Audio input exceeds the explicit duration/file budget")
    import numpy as np

    with wave.open(str(path), "rb") as audio:
        count = audio.getnframes()
        if (
            audio.getframerate() != 16000
            or audio.getnchannels() != 1
            or audio.getsampwidth() != 2
            or audio.getcomptype() != "NONE"
            or not 0 < count <= max_seconds * 16000
        ):
            raise ValueError("Require nonempty bounded16kHz mono PCM16 WAV; conversion must be explicit")
        encoded = audio.readframes(count)
        if len(encoded) != count * 2:
            raise ValueError("Audio sample payload is truncated")
    return np.frombuffer(encoded, dtype="<i2").astype(np.float32) / 32768.0


class WhisperOnnxTranscriber:
    """Verified quantized encoder/uncached decoder with native Whisper features.

    Greedy generation honors the selected checkpoint's forced prefix, suppress
    lists and no-timestamp mode. The deliberately simple uncached decoder bounds
    tokens and avoids an implicit KV-cache ABI. This adapter does not identify
    speakers, sound events, silence confidence or word-level temporal alignment.
    """

    def __init__(self, directory: Path, *, threads: int = 2, max_tokens: int = 128, max_seconds: int = 120) -> None:
        if (
            type(threads) is not int
            or not 1 <= threads <= 32
            or type(max_tokens) is not int
            or not 8 <= max_tokens <= 256
            or type(max_seconds) is not int
            or not 1 <= max_seconds <= 120
        ):
            raise ValueError("Whisper CPU/token/duration budgets exceed their positive limits")
        names = (
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "normalizer.json",
            "added_tokens.json",
            "merges.txt",
            "vocab.json",
            "onnx/encoder_model_quantized.onnx",
            "onnx/decoder_model_quantized.onnx",
        )
        files = artifacts.verify(directory, names)
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor, WhisperTokenizerFast

        self.features = WhisperFeatureExtractor.from_pretrained(str(directory), local_files_only=True)
        self.tokenizer = WhisperTokenizerFast.from_pretrained(str(directory), local_files_only=True)
        config = json.loads((Path(directory) / "generation_config.json").read_text())
        if config.get("is_multilingual") is not False or config.get("return_timestamps") is not False:
            raise ValueError("This adapter requires the selected English no-timestamp profile")
        self.prefix = tuple(self.tokenizer.prefix_tokens)
        if any(self.prefix[position] != token for position, token in config["forced_decoder_ids"]):
            raise ValueError("Whisper native prefix differs from selected generation config")
        self.eos = config["eos_token_id"]
        self.timestamp_begin = config["no_timestamps_token_id"] + 1
        self.suppress = tuple(config["suppress_tokens"])
        self.begin_suppress = tuple(config["begin_suppress_tokens"])
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        self.encoder = ort.InferenceSession(
            str(Path(directory) / names[-2]), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.decoder = ort.InferenceSession(
            str(Path(directory) / names[-1]), sess_options=options, providers=["CPUExecutionProvider"]
        )
        if {item.name for item in self.encoder.get_inputs()} != {"input_features"} or {
            item.name for item in self.decoder.get_inputs()
        } != {"input_ids", "encoder_hidden_states"}:
            raise ValueError("Unsupported Whisper encoder/decoder graph inputs")
        self.max_tokens, self.max_seconds = max_tokens, max_seconds
        self.version = artifacts.space_id(
            "whisper-onnx-v1",
            files,
            {
                "threads": threads,
                "max_tokens": max_tokens,
                "max_seconds": max_seconds,
                "sample_rate": 16000,
                "chunk_seconds": 30,
                "mode": "greedy-native-prefix-suppress-no-timestamps-uncached",
                **{name: metadata.version(name) for name in ("onnxruntime", "transformers")},
            },
        )

    def _transcribe_chunk(self, samples: npt.NDArray) -> str:
        import numpy as np

        inputs = self.features(samples, sampling_rate=16000, return_tensors="np")
        hidden = self.encoder.run(["last_hidden_state"], {"input_features": inputs["input_features"]})[0]
        if hidden.shape != (1, 1500, 384) or not np.isfinite(hidden).all():
            raise ValueError("Whisper encoder returned invalid tiny-model features")
        history = list(self.prefix)
        for step in range(self.max_tokens):
            logits = self.decoder.run(
                ["logits"], {"input_ids": np.asarray([history], dtype=np.int64), "encoder_hidden_states": hidden}
            )[0]
            if logits.shape != (1, len(history), 51864) or not np.isfinite(logits).all():
                raise ValueError("Whisper decoder returned invalid English logits")
            scores = logits[0, -1].copy()
            scores[list(self.suppress)] = -np.inf
            scores[self.timestamp_begin :] = -np.inf
            if step == 0:
                scores[list(self.begin_suppress)] = -np.inf
            token = int(scores.argmax())
            if token == self.eos:
                return self.tokenizer.decode(history[len(self.prefix) :], skip_special_tokens=True).strip()
            history.append(token)
        raise ValueError("Whisper exhausted its token budget without EOS; no partial transcript returned")

    def transcribe(self, path: Path) -> tuple[SpeechSegment, ...]:
        """Actually decode independent30-second chunks, preserving source offsets."""
        samples = load_wave(path, max_seconds=self.max_seconds)
        return tuple(
            SpeechSegment(
                start / 16000,
                min(start + 480000, len(samples)) / 16000,
                self._transcribe_chunk(samples[start : start + 480000]),
            )
            for start in range(0, len(samples), 480000)
        )

    def extract_units(
        self,
        path: Path,
        *,
        asset_id: int,
        first_unit_id: int,
        metadata_values: Mapping[str, str | int | float | bool | None] | None = None,
    ) -> tuple[units.RetrievalUnit, ...]:
        """Map real transcripts into resource-linked coarse audio units."""
        contracts.asset_id(asset_id)
        contracts.asset_id(first_unit_id)
        segments = self.transcribe(path)
        return tuple(
            units.RetrievalUnit(
                first_unit_id + index,
                asset_id,
                "audio",
                units.Locator(start_seconds=segment.start_seconds, end_seconds=segment.end_seconds),
                (units.TextEvidence("transcript", segment.text, "whisper-tiny-en-quantized-onnx", self.version),),
                metadata_values or {},
            )
            for index, segment in enumerate(segments)
        )
