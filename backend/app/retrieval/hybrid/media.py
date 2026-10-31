"""Explicit text-evidence adapters and a limited pixel-color reference baseline."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from . import contracts, documents, lexical, units

if TYPE_CHECKING:
    from numpy import typing as npt

    from . import querying


class EvidenceTextRetriever:
    """BM25 over selected unit evidence; absent text does not create a fake hit."""

    def __init__(self, catalog: units.UnitCatalog, kinds: frozenset[str]) -> None:
        """Index explicit text kinds while inheriting the shared unit metadata."""
        self.retriever = lexical.Bm25Retriever(
            [
                documents.CorpusDocument(unit.unit_id, unit.text(kinds), metadata=unit.metadata)
                for unit in catalog.units.values()
                if unit.text(kinds).strip()
            ]
        )

    def retrieve(self, query: contracts.SearchQuery, limit: int) -> Sequence[int]:
        """Apply the unit filters before the lexical candidate limit."""
        return self.retriever.retrieve(query, limit)


_COLOR_NAMES = (
    ("red", "红色"),
    ("yellow", "黄色"),
    ("green", "绿色"),
    ("cyan", "青色"),
    ("blue", "蓝色"),
    ("magenta", "紫红色"),
)


class ColorHistogramRetriever:
    """Six-bin saturated-pixel color matching, not a visual-semantic model.

    Text queries recognize only the listed English/Chinese color words. There is
    no object, OCR, spatial or temporal understanding. This CPU reference makes
    image-derived evidence independently testable without model installations.
    """

    encoder_id = "reference-hsv-six-colors-32x32-s90-v45-v1"

    def __init__(self, catalog: units.UnitCatalog, images: Mapping[int, Path]) -> None:
        """Read only explicitly supplied images, mapping each path to a unit ID."""
        import numpy as np
        from PIL import Image

        self.ids = tuple(contracts.asset_id(identifier) for identifier in images)
        self.metadata = {identifier: catalog.units[identifier].metadata for identifier in self.ids}
        rows = []
        for path in images.values():
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB").resize((32, 32)).convert("HSV"), dtype=np.float32)
            visible = (pixels[:, :, 1] >= 90) & (pixels[:, :, 2] >= 45)
            # Hue wraps at red, so half-bin rounding uses modulo six.
            bins = ((pixels[:, :, 0] / 256 * 6 + 0.5).astype(int) % 6)[visible]
            histogram = np.bincount(bins, minlength=6).astype(np.float32)
            norm = np.linalg.norm(histogram)
            rows.append(histogram / norm if norm else histogram)
        self.matrix = np.asarray(rows, dtype=np.float32).reshape((-1, 6))
        self.matrix.flags.writeable = False

    def retrieve_image(self, image: Path, filters: contracts.SearchFilter, limit: int) -> tuple[int, ...]:
        """Match an explicit image's pixel histogram, without textual substitution.

        This reference supports color similarity only. It cannot identify the
        same object, scene or document when color is insufficient.
        """
        import numpy as np
        from PIL import Image

        if limit <= 0:
            return ()
        with Image.open(image) as source:
            pixels = np.asarray(source.convert("RGB").resize((32, 32)).convert("HSV"), dtype=np.float32)
        visible = (pixels[:, :, 1] >= 90) & (pixels[:, :, 2] >= 45)
        bins = ((pixels[:, :, 0] / 256 * 6 + 0.5).astype(int) % 6)[visible]
        return self._rank(np.bincount(bins, minlength=6).astype(np.float32), filters, limit)

    def retrieve(self, query: contracts.SearchQuery, limit: int) -> tuple[int, ...]:
        """Return positive color matches, pre-filtered and tied by unit ID."""
        import numpy as np

        if limit <= 0:
            return ()
        words = set(re.findall(r"[a-z]+", query.text.casefold()))
        # Longest aliases first: 紫红色 must not also be interpreted as 红色.
        aliases = sorted((chinese for _, chinese in _COLOR_NAMES), key=len, reverse=True)
        chinese_words = set(re.findall("|".join(aliases), query.text))
        vector = np.array(
            [float(english in words or chinese in chinese_words) for english, chinese in _COLOR_NAMES], dtype=np.float32
        )
        return self._rank(vector, query.filters, limit)

    def _rank(self, vector: npt.ArrayLike, filters: contracts.SearchFilter, limit: int) -> tuple[int, ...]:
        """Score only the shared eligible domain and omit absent color evidence."""
        import numpy as np

        norm = np.linalg.norm(vector)
        if not norm:
            return ()
        selected = [
            index for index, identifier in enumerate(self.ids) if filters.matches(identifier, self.metadata[identifier])
        ]
        if not selected:
            return ()
        scores = self.matrix[selected] @ (vector / norm)
        ranked = sorted(
            ((self.ids[index], float(score)) for index, score in zip(selected, scores) if score > 0),
            key=lambda item: (-item[1], item[0]),
        )
        return tuple(identifier for identifier, _ in ranked[:limit])


class ImageHistogramRoute:
    """Image-input adapter for the limited pixel histogram reference."""

    def __init__(self, retriever: ColorHistogramRetriever) -> None:
        self.retriever = retriever

    def retrieve(self, query: querying.MultimodalQuery, limit: int) -> tuple[int, ...]:
        """Use only actual supplied pixels; combined text is handled by other routes."""
        if query.image is None:
            return ()
        return self.retriever.retrieve_image(query.image, query.filters, limit)
