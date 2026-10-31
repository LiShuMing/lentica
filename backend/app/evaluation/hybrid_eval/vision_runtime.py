"""Development-only, fixed-matrix SigLIP runtime profiling and identity lock."""

from __future__ import annotations

import time
from pathlib import Path

from backend.app.evaluation.hybrid_eval import mirror_images, reporting
from backend.app.retrieval.hybrid import siglip_onnx

CONFIGURATIONS = ((2, 8), (1, 1), (2, 4))


def profile(source: mirror_images.ImageDataset, directory: Path) -> reporting.JsonObject:
    """Time two fresh encodings of the fixed20 development images per config.

    No validation/test model inference or quality-based runtime selection occurs.
    Timings exclude model loading. Results describe this shared CPU observation,
    not a throughput guarantee; quantized cross-batch outputs can differ.
    """
    items = tuple(
        item
        for label in mirror_images.CLASSES
        for item in sorted(source.partition("dev"), key=lambda row: row.identifier)
        if item.label == label
    )
    selected = tuple(
        item for label in mirror_images.CLASSES for item in [row for row in items if row.label == label][:2]
    )
    if len(selected) != 20:
        raise ValueError("Vision runtime profile requires two development images per class")
    rows = []
    for threads, batch_size in CONFIGURATIONS:
        model = siglip_onnx.SiglipOnnxEncoder(directory, threads=threads, batch_size=batch_size)
        seconds, digests = [], []
        for _ in range(2):
            started = time.perf_counter()
            vectors = model.encode_images([source.directory / item.filename for item in selected])
            seconds.append(time.perf_counter() - started)
            digests.append(reporting.digest(vectors.tolist()))
        rows.append(
            {
                "threads": threads,
                "batch_size": batch_size,
                "encoder_id": model.encoder_id,
                "seconds": seconds,
                "embedding_sha256": digests,
                "repeat_vectors_equal": digests[0] == digests[1],
            }
        )
    winner = min(range(len(rows)), key=lambda index: (sum(rows[index]["seconds"]), index))
    report = {
        "schema": "lentica-vision-runtime-v1",
        "dataset_identity": source.identity,
        "development_ids": [item.identifier for item in selected],
        "configurations": rows,
        "selected_index": winner,
        "scope": "Two actual development encodings/config; no quality/test selection, model loading excluded, shared CPU.",
    }
    report["sha256"] = reporting.digest(report)
    return report


def frozen_model(
    source: mirror_images.ImageDataset, directory: Path, artifact: reporting.JsonObject
) -> siglip_onnx.SiglipOnnxEncoder:
    """Verify the development runtime lock before constructing a local model."""
    report = dict(artifact)
    digest = report.pop("sha256")
    if (
        reporting.digest(report) != digest
        or report["schema"] != "lentica-vision-runtime-v1"
        or report["dataset_identity"] != source.identity
        or type(report["selected_index"]) is not int
        or not 0 <= report["selected_index"] < len(CONFIGURATIONS)
        or len(report["configurations"]) != len(CONFIGURATIONS)
    ):
        raise ValueError("Frozen vision runtime source/config identity differs")
    selected = report["configurations"][report["selected_index"]]
    threads, batch_size = CONFIGURATIONS[report["selected_index"]]
    if selected["threads"] != threads or selected["batch_size"] != batch_size:
        raise ValueError("Frozen vision runtime matrix differs")
    model = siglip_onnx.SiglipOnnxEncoder(directory, threads=threads, batch_size=batch_size)
    if model.encoder_id != selected["encoder_id"]:
        raise ValueError("Frozen vision runtime model identity differs")
    return model
