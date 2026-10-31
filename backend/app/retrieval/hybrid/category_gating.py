"""Opt-in two-model known-vocabulary image gate, independent of dataset labels."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from . import artifacts, vision_reranking

if TYPE_CHECKING:
    from numpy import typing as npt


@dataclass(frozen=True)
class GateBatch:
    """Aligned features/decisions; fast_accepts marks actual slow-call eligibility.

    Rejected slow-vector rows are zeros, never retrieval evidence. The decisions
    mask must accompany them. No relevance labels enter this numeric boundary.
    """

    fast_vectors: npt.NDArray
    fast_accepts: tuple[bool, ...]
    slow_accepts: tuple[bool, ...]
    margins: tuple[float | None, ...]
    fast_seconds: float
    slow_seconds: float
    total_seconds: float
    slow_vectors: npt.NDArray | None = None

    def decisions(self, minimum_margin: float) -> tuple[bool, ...]:
        """Apply an explicit margin; missing slow results never pass the gate."""
        if (
            type(minimum_margin) not in (float, int)
            or not math.isfinite(minimum_margin)
            or not -2 <= minimum_margin <= 2
        ):
            raise ValueError("Category margin must be finite and within cosine-difference bounds")
        return tuple(
            first and second and margin is not None and margin >= minimum_margin
            for first, second, margin in zip(self.fast_accepts, self.slow_accepts, self.margins)
        )


def _features(value: object, rows: int, dimensions: int | None = None) -> npt.NDArray:
    """Validate only at the numeric provider boundary; do not normalize silently."""
    import numpy as np

    matrix = np.asarray(value)
    if matrix.ndim != 2 or matrix.shape[0] != rows or not 0 < matrix.shape[1] <= 8192 or matrix.dtype.kind not in "fiu":
        raise ValueError("Category encoder returned misaligned feature rows/dimensions")
    if dimensions is not None and matrix.shape[1] != dimensions:
        raise ValueError("Category encoder image/text spaces differ")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(matrix).all() or not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Category encoder requires finite unit-normalized features")
    return matrix


class CascadedCategoryGate:
    """Reject by a fast vocabulary classifier before invoking a slower model.

    The supplied vocabulary and gallery membership are explicit configuration,
    not inferred ground truth. Models retain independent feature spaces. This
    is a known-category gate, not open-world confidence or arbitrary semantics.
    Construction actually encodes the fixed prompts once; scoring never hides
    failures or invokes the slow model for already rejected inputs.
    """

    def __init__(
        self,
        fast: vision_reranking.ImageTextEncoder,
        slow: vision_reranking.ImageTextEncoder,
        prompts: Mapping[str, str],
        gallery_classes: frozenset[str],
        *,
        batch_size: int = 8,
        fast_negative_margin: float | None = None,
    ) -> None:
        if (
            not 2 <= len(prompts) <= 256
            or any(not isinstance(key, str) or not key for key in prompts)
            or any(not isinstance(text, str) or not text.strip() or len(text) > 32000 for text in prompts.values())
            or not gallery_classes
            or not gallery_classes < set(prompts)
            or type(batch_size) is not int
            or not 1 <= batch_size <= 64
            or (
                fast_negative_margin is not None
                and (
                    type(fast_negative_margin) not in (int, float)
                    or not math.isfinite(fast_negative_margin)
                    or not 0 <= fast_negative_margin <= 2
                )
            )
        ):
            raise ValueError("Category gate requires bounded explicit vocabulary, proper gallery subset and batch")
        self._fast, self._slow = fast, slow
        self.labels, self.prompts = tuple(prompts), tuple(prompts.values())
        self.gallery_classes, self.batch_size = frozenset(gallery_classes), batch_size
        self._fast_negative_margin = fast_negative_margin
        self.fast_encoder_id, self.slow_encoder_id = fast.encoder_id, slow.encoder_id
        self._gallery = tuple(index for index, label in enumerate(self.labels) if label in self.gallery_classes)
        self._absent = tuple(index for index, label in enumerate(self.labels) if label not in self.gallery_classes)
        started = time.perf_counter()
        self._fast_text = _features(fast.encode_texts(self.prompts), len(self.labels)).copy()
        self._slow_text = _features(slow.encode_texts(self.prompts), len(self.labels)).copy()
        self.prompt_encoding_seconds = time.perf_counter() - started
        self.gate_id = artifacts.space_id(
            "category-cascade-v1" if fast_negative_margin is None else "category-rescue-v1",
            {"fast": self.fast_encoder_id, "slow": self.slow_encoder_id},
            {
                "prompts": list(zip(self.labels, self.prompts)),
                "gallery": sorted(self.gallery_classes),
                "batch_size": batch_size,
                "policy": "fast-gallery-prediction-then-slow-gallery-prediction-and-margin",
                **({"fast_negative_margin": fast_negative_margin} if fast_negative_margin is not None else {}),
            },
        )

    @property
    def fast_negative_margin(self) -> float | None:
        """Frozen rescue tolerance; None keeps the original strict policy."""
        return self._fast_negative_margin

    def score_images(self, paths: Sequence[Path]) -> GateBatch:
        """Actually encode bounded pixels, preserving input positions and failures.

        Returned fast features can be used by their own retrieval space. Slow
        class scores never become fast features or native fusion scores.
        """
        if isinstance(paths, (str, Path)) or len(paths) > self.batch_size:
            raise ValueError("Category gate requires an explicit bounded image batch")
        if self._fast.encoder_id != self.fast_encoder_id or self._slow.encoder_id != self.slow_encoder_id:
            raise ValueError("Category model identity changed after prompt construction")
        paths = tuple(Path(path) for path in paths)
        if any(not path.is_file() or path.stat().st_size > 8 * 1024**2 for path in paths):
            raise ValueError("Category image path/file budget differs")
        import numpy as np

        if not paths:
            return GateBatch(np.empty((0, self._fast_text.shape[1])), (), (), (), 0.0, 0.0, 0.0)
        started = time.perf_counter()
        fast_vectors = _features(self._fast.encode_images(paths), len(paths), self._fast_text.shape[1])
        fast_seconds = time.perf_counter() - started
        fast_scores = fast_vectors @ self._fast_text.T
        first = tuple(
            self.labels[int(row.argmax())] in self.gallery_classes
            if self.fast_negative_margin is None
            else float(row[list(self._absent)].max() - row[list(self._gallery)].max()) <= self.fast_negative_margin
            for row in fast_scores
        )
        positions = [index for index, accepted in enumerate(first) if accepted]
        second, margins = [False] * len(paths), [None] * len(paths)
        slow_seconds = 0.0
        aligned_slow = np.zeros((len(paths), self._slow_text.shape[1]), dtype=np.float32)
        if positions:
            slow_started = time.perf_counter()
            slow_vectors = _features(
                self._slow.encode_images([paths[index] for index in positions]),
                len(positions),
                self._slow_text.shape[1],
            )
            slow_seconds = time.perf_counter() - slow_started
            aligned_slow = np.zeros((len(paths), slow_vectors.shape[1]), dtype=slow_vectors.dtype)
            aligned_slow[positions] = slow_vectors
            scores = slow_vectors @ self._slow_text.T
            for index, row in zip(positions, scores):
                second[index] = self.labels[int(row.argmax())] in self.gallery_classes
                margins[index] = float(row[list(self._gallery)].max() - row[list(self._absent)].max())
        return GateBatch(
            fast_vectors,
            first,
            tuple(second),
            tuple(margins),
            fast_seconds,
            slow_seconds,
            time.perf_counter() - started,
            aligned_slow,
        )
