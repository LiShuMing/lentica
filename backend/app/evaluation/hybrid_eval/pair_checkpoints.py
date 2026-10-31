"""Explicit append-only model-score checkpoints with source/config identities."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from backend.app.evaluation.hybrid_eval import reporting


class PairScoreCache:
    """Resume a long fixed experiment without confusing cached/fresh inference.

    This trusted local score artifact is separate from product retrieval caches.
    Each line is a complete committed query result. A truncated final line fails
    closed rather than guessing which work was completed. There is one writer.
    """

    def __init__(self, path: Path, identity: reporting.JsonObject) -> None:
        self.path = Path(path)
        self.scores: dict[str, tuple[float, ...]] = {}
        header = {"schema": "lentica-pair-score-checkpoint-v1", "identity": identity}
        if self.path.exists():
            if self.path.stat().st_size > 8 * 1024**2:
                raise ValueError("Pair score checkpoint exceeds 8 MiB")
            with self.path.open(encoding="utf-8") as stream:
                if json.loads(stream.readline()) != header:
                    raise ValueError("Pair checkpoint source/model/config identity differs")
                for index, line in enumerate(stream):
                    if index >= 5000 or len(line) > 64000:
                        raise ValueError("Pair score checkpoint row budget exceeded")
                    value = json.loads(line)
                    self._validate(value["key"], value["scores"])
                    if value.get("sha256") != reporting.digest([value["key"], value["scores"]]):
                        raise ValueError("Pair checkpoint score digest mismatch")
                    if value["key"] in self.scores:
                        raise ValueError("Duplicate pair score checkpoint key")
                    self.scores[value["key"]] = tuple(value["scores"])
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(header, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    @staticmethod
    def _validate(key: str, scores) -> None:
        if (
            not isinstance(key, str)
            or len(key) != 64
            or any(char not in "0123456789abcdef" for char in key)
            or not isinstance(scores, (list, tuple))
            or len(scores) > 1000
            or any(type(score) not in (int, float) or not math.isfinite(score) for score in scores)
        ):
            raise ValueError("Invalid pair score checkpoint key/values")

    def get(self, key: str, count: int) -> tuple[float, ...] | None:
        """Return one identity-bound result, rejecting inconsistent pair counts."""
        value = self.scores.get(key)
        if value is not None and len(value) != count:
            raise ValueError("Pair checkpoint candidate count mismatch")
        return value

    def put(self, key: str, scores: tuple[float, ...]) -> None:
        """Append/fsync a complete actual-inference result before exposing it."""
        self._validate(key, scores)
        if key in self.scores:
            raise ValueError("Cannot overwrite a completed pair score checkpoint")
        line = (
            json.dumps({"key": key, "scores": scores, "sha256": reporting.digest([key, scores])}, allow_nan=False)
            + "\n"
        )
        if len(line) > 64000 or len(self.scores) >= 5000 or self.path.stat().st_size + len(line) > 8 * 1024**2:
            raise ValueError("Pair checkpoint exceeds its bounded output budget")
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        self.scores[key] = tuple(scores)
