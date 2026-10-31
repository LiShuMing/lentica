"""Independent reference-only word-error scoring of explicitly prepared speech."""

from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Sequence
from pathlib import Path

from backend.app.evaluation.hybrid_eval import reporting
from backend.app.retrieval.hybrid import audio_onnx, snapshots


def words(text: str) -> tuple[str, ...]:
    """Normalize NFKC/case and punctuation to spaces, without number expansion."""
    if not isinstance(text, str) or len(text) > 32000:
        raise ValueError("Word-error scoring requires bounded text")
    return tuple(re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFKC", text).casefold()).split())


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Count minimum unit-cost insertions/deletions/substitutions with bounded DP.

    Rolling rows bound memory. This returns the minimum total error count,
    without claiming a unique alignment or word-level acoustic timestamps.
    """
    if max(len(reference), len(hypothesis)) > 2000:
        raise ValueError("Word-error alignment exceeds2000-word budget")
    previous = list(range(len(hypothesis) + 1))
    for index, expected in enumerate(reference, 1):
        current = [index]
        for position, predicted in enumerate(hypothesis, 1):
            current.append(
                min(current[-1] + 1, previous[position] + 1, previous[position - 1] + (expected != predicted))
            )
        previous = current
    return previous[-1]


def evaluate(
    directory: Path, manifest: reporting.JsonObject, transcriber: audio_onnx.WhisperOnnxTranscriber
) -> reporting.JsonObject:
    """Run fresh waveform-only inference and score every fixed selected record.

    The explicit transcriber extension boundary can raise arbitrary provider
    exceptions. Each is reported and scored as an empty hypothesis (deletions),
    never dropped from the denominator. Input-integrity failures still abort.
    Reference text is accessed only after the corresponding inference call.
    """
    rows = []
    for item in manifest["items"]:
        path = Path(directory) / item["filename"]
        if snapshots.sha256_file(path) != item["sha256"]:
            raise ValueError("Prepared speech waveform identity differs")
        error = None
        started = time.perf_counter()
        try:
            segments = transcriber.transcribe(path)
            hypothesis = " ".join(segment.text for segment in segments)
        except Exception as failure:
            error, hypothesis = type(failure).__name__, ""
        seconds = time.perf_counter() - started
        reference_words, predicted_words = words(item["reference"]), words(hypothesis)
        if not reference_words:
            raise ValueError("Prepared speech requires a nonempty reference")
        count = edit_distance(reference_words, predicted_words)
        rows.append(
            {
                "source_id": item["source_id"],
                "reference": item["reference"],
                "hypothesis": hypothesis,
                "reference_words": len(reference_words),
                "errors": count,
                "wer": count / len(reference_words),
                "inference_seconds": seconds,
                "duration_seconds": item["duration_seconds"],
                "failure": error,
            }
        )
    total_words = sum(row["reference_words"] for row in rows)
    if not rows or not total_words:
        raise ValueError("Speech quality requires nonempty judged records")
    return {
        "schema": "lentica-speech-quality-v1",
        "dataset_identity": manifest["identity"],
        "transcriber_version": transcriber.version,
        "clips": len(rows),
        "failures": sum(row["failure"] is not None for row in rows),
        "reference_words": total_words,
        "errors": sum(row["errors"] for row in rows),
        "micro_wer": sum(row["errors"] for row in rows) / total_words,
        "inference_seconds": sum(row["inference_seconds"] for row in rows),
        "audio_seconds": sum(row["duration_seconds"] for row in rows),
        "transcript_sha256": reporting.digest([(row["source_id"], row["hypothesis"], row["failure"]) for row in rows]),
        "normalization": "NFKC, casefold, Unicode nonword punctuation to spaces, whitespace split; no number expansion",
        "scope": "Fixed first12 source-ID-sorted dummy LibriSpeech clean clips; unknown pretraining overlap; no tuning, Chinese/noise/word-timing/general accuracy or production latency claim.",
        "rows": rows,
    }
