"""Bounded explicit embedded-audio Parquet to anonymous read-only speech inputs."""

from __future__ import annotations

import hashlib
import io
import json
import wave
from importlib import metadata
from pathlib import Path

from backend.app.evaluation.hybrid_eval import reporting, speech_quality
from backend.app.retrieval.hybrid import audio_onnx, snapshots


def _decode(raw: bytes, destination: Path) -> float:
    """Decode native mono16k PCM16 only, without following an embedded path."""
    import av

    frames = []
    count = 0
    with av.open(io.BytesIO(raw)) as container:
        if len(container.streams.audio) != 1:
            raise ValueError("Speech source requires exactly one audio stream")
        for frame in container.decode(audio=0):
            # PCM WAV can omit a channel-layout label while declaring exactly
            # one channel. Count channels, rather than requiring the word mono.
            if (
                frame.sample_rate != 16000
                or len(frame.layout.channels) != 1
                or frame.format.name not in {"s16", "s16p"}
            ):
                raise ValueError("Speech source must already be native16k mono PCM16")
            count += frame.samples
            if count > 480000:
                raise ValueError("Speech source exceeds30seconds")
            samples = frame.to_ndarray()
            if samples.size != frame.samples:
                raise ValueError("Speech native channel/sample shape differs")
            frames.append(samples.astype("<i2").tobytes())
    if not count:
        raise ValueError("Empty speech source")
    with wave.open(str(destination), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(b"".join(frames))
    return count / 16000


def prepare(source: Path, directory: Path) -> reporting.JsonObject:
    """Write fixed first12 ID-sorted references and independently decoded clips.

    Only a new explicit directory is written. Source bytes/metadata stay intact;
    no URL or file path from the dataset is opened. Incomplete output has no
    completed manifest and cannot be used by load().
    """
    source, directory = Path(source), Path(directory)
    if directory.exists() or not 0 < source.stat().st_size <= 32 * 1024**2:
        raise ValueError("Speech preparation requires bounded source and fresh output")
    import pyarrow.parquet as pq

    before = snapshots.sha256_file(source)
    parquet = pq.ParquetFile(source)
    if not 12 <= parquet.metadata.num_rows <= 1000:
        raise ValueError("Speech Parquet row count exceeds fixed source budget")
    decoded_bytes = sum(
        parquet.metadata.row_group(group).column(column).total_uncompressed_size
        for group in range(parquet.metadata.num_row_groups)
        for column in range(parquet.metadata.num_columns)
    )
    if not 0 < decoded_bytes <= 128 * 1024**2:
        raise ValueError("Speech Parquet decompression exceeds128MiB")
    rows = parquet.read(columns=["id", "text", "audio"]).to_pylist()
    if any(not isinstance(row["id"], str) or not row["id"] for row in rows) or len({row["id"] for row in rows}) != len(
        rows
    ):
        raise ValueError("Speech source IDs must be unique nonempty strings")
    selected = sorted(rows, key=lambda row: row["id"])[:12]
    directory.mkdir(parents=True, exist_ok=False)
    items = []
    for index, row in enumerate(selected):
        raw = row["audio"]["bytes"]
        if not isinstance(raw, bytes) or not 0 < len(raw) <= 2 * 1024**2 or not speech_quality.words(row["text"]):
            raise ValueError("Invalid bounded embedded speech/reference")
        filename = f"clip-{index:05d}.wav"
        duration = _decode(raw, directory / filename)
        items.append(
            {
                "source_id": row["id"],
                "reference": row["text"],
                "filename": filename,
                "source_audio_sha256": hashlib.sha256(raw).hexdigest(),
                "sha256": snapshots.sha256_file(directory / filename),
                "duration_seconds": duration,
            }
        )
    if snapshots.sha256_file(source) != before:
        raise ValueError("Speech Parquet source changed during preparation")
    manifest = {
        "schema": "lentica-speech-source-v1",
        "parquet_sha256": before,
        "source_rows": len(rows),
        "selection": "first12 lexicographically sorted unique source IDs, fixed before inference",
        "conversion": "embedded native16k mono PCM16 decoded to WAV, no resampling/channel mixing",
        "versions": {name: metadata.version(name) for name in ("av", "pyarrow")},
        "items": items,
    }
    manifest["identity"] = reporting.digest(manifest)
    with (directory / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return load(directory)


def load(directory: Path) -> reporting.JsonObject:
    """Check complete dataset identity, fixed anonymous paths and actual inputs."""
    path = Path(directory) / "manifest.json"
    if path.stat().st_size > 1024**2:
        raise ValueError("Speech manifest exceeds1MiB")
    manifest = json.loads(path.read_bytes())
    identity = manifest.pop("identity")
    if (
        reporting.digest(manifest) != identity
        or manifest["schema"] != "lentica-speech-source-v1"
        or len(manifest["items"]) != 12
    ):
        raise ValueError("Speech dataset identity/schema/count differs")
    source_ids = [row["source_id"] for row in manifest["items"]]
    if source_ids != sorted(set(source_ids)):
        raise ValueError("Speech source IDs must be unique and ordered")
    for index, item in enumerate(manifest["items"]):
        if item["filename"] != f"clip-{index:05d}.wav" or not speech_quality.words(item["reference"]):
            raise ValueError("Speech filename/reference differs from declared protocol")
        audio = Path(directory) / item["filename"]
        if (
            snapshots.sha256_file(audio) != item["sha256"]
            or len(audio_onnx.load_wave(audio, max_seconds=30)) / 16000 != item["duration_seconds"]
        ):
            raise ValueError("Speech waveform hash/duration differs")
    return {**manifest, "identity": identity}
