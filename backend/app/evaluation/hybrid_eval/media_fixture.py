"""Generate local image/PDF controls with explicit, non-model annotations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import documents, units

_COLORS = (
    ("red", "#e52b24"),
    ("yellow", "#e2cc22"),
    ("green", "#29b83f"),
    ("cyan", "#20c4ce"),
    ("blue", "#294fe2"),
    ("magenta", "#d42bcc"),
)


@dataclass(frozen=True)
class MediaCase:
    """Known generation conditions, separate from searchable evidence."""

    query_id: str
    query: hybrid.SearchQuery
    category: str
    relevant_assets: frozenset[int]
    relevant_units: frozenset[int]


@dataclass
class MediaFixture:
    """Generated media, extracted page text and explicitly synthetic annotations."""

    catalog: units.UnitCatalog
    images: dict[int, Path]
    assets: list[documents.CorpusDocument]
    cases: list[MediaCase]


def generate(directory: Path, *, groups: int = 12) -> MediaFixture:
    """Write a fresh bounded fixture directory; never overwrite user sources.

    Each group contains 18 colored screenshots and one eight-page PDF. A code
    has a true screenshot/page and misleading captions or near-match negatives.
    Pixel colors and native PDF text are actually read by the experiment. OCR
    transcriptions/captions are supplied annotations, not model execution.
    """
    import fitz
    from PIL import Image, ImageDraw

    if type(groups) is not int or not 1 <= groups <= 50:
        raise ValueError("Fixture groups must be between 1 and 50")
    directory.mkdir(parents=True, exist_ok=False)
    corpus, images, assets, cases = [], {}, [], []
    for group in range(groups):
        code = f"ERR_{1000 + group}"
        color_name = _COLORS[group % len(_COLORS)][0]
        visual_assets, visual_units = set(), set()
        for variant in range(18):
            identifier = group * 100 + variant
            color_index = (group + variant) % len(_COLORS)
            if variant in {1, 2}:
                color_index = group % len(_COLORS)
            if variant == 0:
                color_index = (group + 1) % len(_COLORS)
            name, color = _COLORS[color_index]
            extracted = code if variant == 1 else f"{code}_{variant}"
            caption = f"{color_name} application screen showing {code}" if variant == 0 else "application screen"
            path = directory / f"g{group}-screen{variant}.png"
            image = Image.new("RGB", (320, 180), color)
            ImageDraw.Draw(image).text((20, 75), extracted, fill="white")
            image.save(path)
            images[identifier] = path
            corpus.append(
                units.RetrievalUnit(
                    identifier,
                    identifier,
                    "image",
                    evidence=(
                        units.TextEvidence("ocr", extracted, "synthetic-supplied-transcription", "v1"),
                        units.TextEvidence("caption", caption, "synthetic-noisy-caption", "v1"),
                    ),
                    metadata={"group": group, "media_type": "image"},
                )
            )
            assets.append(documents.CorpusDocument(identifier, f"Generated screenshot {identifier}"))
            if name == color_name:
                visual_assets.add(identifier)
                visual_units.add(identifier)
        asset_id = group * 100 + 50
        pdf_path = directory / f"g{group}-manual.pdf"
        with fitz.open() as pdf:
            for page_index in range(8):
                page = pdf.new_page()
                native = f"Manual example {code}" if page_index == 2 else f"Manual example {code}_{page_index}"
                page.insert_text((72, 72), native)
            pdf.save(pdf_path, no_new_id=True)
        with fitz.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf):
                corpus.append(
                    units.RetrievalUnit(
                        group * 100 + 60 + page_index,
                        asset_id,
                        "page",
                        units.Locator(page=page_index + 1),
                        (
                            units.TextEvidence(
                                "native", page.get_text("text"), "pymupdf-native", str(fitz.VersionBind)
                            ),
                            units.TextEvidence("caption", f"screen showing {code}", "synthetic-noisy-caption", "v1"),
                        ),
                        {"group": group, "media_type": "document"},
                    )
                )
        assets.append(documents.CorpusDocument(asset_id, f"Generated manual {group}"))
        for category, text, media_type, relevant_assets, relevant_units in (
            ("identifier", code, None, {group * 100 + 1, asset_id}, {group * 100 + 1, group * 100 + 62}),
            ("visual_color", color_name, "image", visual_assets, visual_units),
            ("combined", f"{color_name} {code}", "image", {group * 100 + 1}, {group * 100 + 1}),
            ("page", code, "document", {asset_id}, {group * 100 + 62}),
        ):
            filters = {"group": group}
            if media_type:
                filters["media_type"] = media_type
            cases.append(
                MediaCase(
                    f"g{group}-{category}",
                    hybrid.SearchQuery(text, hybrid.SearchFilter(equals=filters)),
                    category,
                    frozenset(relevant_assets),
                    frozenset(relevant_units),
                )
            )
    return MediaFixture(units.UnitCatalog(corpus), images, assets, cases)
