"""Explicit CPU OCR and native PDF block extraction with source coordinates."""

from __future__ import annotations

import dataclasses
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from . import artifacts, contracts, snapshots, units


@dataclass(frozen=True)
class OcrSpan:
    """Recognized text with normalized original-image bounds and model confidence."""

    text: str
    bbox: tuple[float, float, float, float]
    confidence: float


class RapidOcrExtractor:
    """Run the explicitly installed RapidOCR model bundle on selected images.

    This is actual OCR, not a supplied caption/transcription. Package/model
    hashes and confidence threshold identify its output version. Confidence is
    an OCR score, not a calibrated probability that a search result is relevant.
    """

    def __init__(self, *, threads: int = 2, max_image_pixels: int = 16_000_000, max_regions: int = 512) -> None:
        if (
            any(type(value) is not int or value <= 0 for value in (threads, max_image_pixels, max_regions))
            or threads > 32
        ):
            raise ValueError("Explicit positive OCR budgets and at most 32 CPU threads are required")
        import rapidocr_onnxruntime

        directory = Path(rapidocr_onnxruntime.__file__).parent / "models"
        names = ("ch_PP-OCRv4_det_infer.onnx", "ch_ppocr_mobile_v2.0_cls_infer.onnx", "ch_PP-OCRv4_rec_infer.onnx")
        files = {name: snapshots.sha256_file(directory / name) for name in names}
        self.engine = rapidocr_onnxruntime.RapidOCR(
            det_model_path=str(directory / names[0]),
            cls_model_path=str(directory / names[1]),
            rec_model_path=str(directory / names[2]),
            det_use_cuda=False,
            cls_use_cuda=False,
            rec_use_cuda=False,
            det_use_dml=False,
            cls_use_dml=False,
            rec_use_dml=False,
            intra_op_num_threads=threads,
            inter_op_num_threads=1,
            text_score=0.5,
        )
        self.max_image_pixels, self.max_regions = max_image_pixels, max_regions
        self.version = artifacts.space_id(
            "rapidocr-cpu-v1",
            files,
            {
                "package": metadata.version("rapidocr-onnxruntime"),
                "onnxruntime": metadata.version("onnxruntime"),
                "text_score": 0.5,
                "threads": threads,
                "max_image_pixels": max_image_pixels,
                "max_regions": max_regions,
            },
        )

    def recognize(self, path: Path) -> tuple[OcrSpan, ...]:
        """Read one selected image; reject excessive size/regions rather than clip."""
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
            if width * height > self.max_image_pixels:
                raise ValueError("OCR input exceeds pixel budget")
        rows, _ = self.engine(str(path))
        if rows is None:
            return ()
        if len(rows) > self.max_regions:
            raise ValueError("OCR output exceeds region budget")
        result = []
        for points, text, confidence in rows:
            xs, ys = zip(*points)
            bbox = (
                max(0.0, min(xs) / width),
                max(0.0, min(ys) / height),
                min(1.0, max(xs) / width),
                min(1.0, max(ys) / height),
            )
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                continue
            units.Locator(bbox=bbox)
            result.append(OcrSpan(text, bbox, float(confidence)))
        return tuple(result)

    def extract_units(
        self,
        path: Path,
        *,
        asset_id: int,
        first_unit_id: int,
        page: int | None = None,
        metadata_values: Mapping[str, str | int | float | bool | None] | None = None,
    ) -> tuple[units.RetrievalUnit, ...]:
        """Assign caller-owned IDs to image/page regions, inheriting resource filters."""
        contracts.asset_id(asset_id)
        contracts.asset_id(first_unit_id)
        return tuple(
            units.RetrievalUnit(
                first_unit_id + offset,
                asset_id,
                "page" if page else "image",
                units.Locator(page=page, bbox=span.bbox),
                (units.TextEvidence("ocr", span.text, "rapidocr-onnxruntime", self.version),),
                metadata_values or {},
            )
            for offset, span in enumerate(self.recognize(path))
        )


def pdf_blocks(
    path: Path,
    *,
    asset_id: int,
    first_unit_id: int,
    max_pages: int = 100,
    max_blocks: int = 5000,
    metadata_values: Mapping[str, str | int | float | bool | None] | None = None,
) -> tuple[units.RetrievalUnit, ...]:
    """Extract native text blocks with displayed-page normalized bounding boxes.

    Rotation is applied to native coordinates before normalizing against page.rect.
    Image-only pages yield no native evidence and require a separately selected
    OCR operation. This layout baseline preserves blocks, not reading-order/table
    semantics. PDF bytes remain unchanged and no rendered images are written.
    """
    import fitz

    contracts.asset_id(asset_id)
    contracts.asset_id(first_unit_id)
    if any(type(value) is not int or value <= 0 for value in (max_pages, max_blocks)):
        raise ValueError("PDF extraction budgets must be positive integers")
    result = []
    with fitz.open(path) as document:
        if len(document) > max_pages:
            raise ValueError("PDF exceeds page budget")
        for page_index, page in enumerate(document):
            for block in page.get_text("blocks"):
                if block[6] != 0 or not block[4].strip():
                    continue
                if len(result) >= max_blocks:
                    raise ValueError("PDF exceeds text block budget")
                rect = (fitz.Rect(block[:4]) * page.rotation_matrix) & page.rect
                if rect.is_empty:
                    continue
                bounds = (
                    rect.x0 / page.rect.width,
                    rect.y0 / page.rect.height,
                    rect.x1 / page.rect.width,
                    rect.y1 / page.rect.height,
                )
                result.append(
                    units.RetrievalUnit(
                        first_unit_id + len(result),
                        asset_id,
                        "page",
                        units.Locator(page=page_index + 1, bbox=bounds),
                        (units.TextEvidence("native", block[4], "pymupdf-native-blocks", metadata.version("PyMuPDF")),),
                        metadata_values or {},
                    )
                )
    return tuple(result)


def pdf_with_ocr(
    path: Path,
    ocr: RapidOcrExtractor,
    *,
    asset_id: int,
    first_unit_id: int,
    max_pages: int = 100,
    max_blocks: int = 5000,
    dpi: int = 150,
    metadata_values: Mapping[str, str | int | float | bool | None] | None = None,
) -> tuple[units.RetrievalUnit, ...]:
    """Preserve native blocks and OCR image-only pages, without modifying a PDF.

    Args:
        path: Explicit local PDF. No library traversal is performed.
        ocr: Explicitly constructed local OCR engine with its own pixel budget.
        asset_id: Final parent resource identity.
        first_unit_id: Start of caller-reserved contiguous unit IDs.
        max_pages: Maximum document pages, checked before rendering.
        max_blocks: Combined native/OCR output bound; excess fails, never truncates.
        dpi: Rendering resolution, bounded to 72..300 pixels per inch.
        metadata_values: Resource filter metadata inherited by every region.

    Returns:
        Page-ordered units with normalized displayed-page coordinates. Temporary
        rendered pages are removed on success or failure. Pages with native text
        are not OCRed: this avoids duplicate votes, but does not cover mixed-page
        image text or infer table structure and reading order.

    Raises:
        ValueError: Page, block, rendering or OCR pixel budgets are exceeded.
    """
    import fitz

    if type(dpi) is not int or not 72 <= dpi <= 300:
        raise ValueError("PDF OCR rendering DPI must lie within 72..300")
    native = pdf_blocks(
        path,
        asset_id=asset_id,
        first_unit_id=first_unit_id,
        max_pages=max_pages,
        max_blocks=max_blocks,
        metadata_values=metadata_values,
    )
    by_page: dict[int, list[units.RetrievalUnit]] = {}
    for unit in native:
        by_page.setdefault(unit.locator.page, []).append(unit)
    result = []
    with fitz.open(path) as document, tempfile.TemporaryDirectory(prefix="lentica-pdf-ocr-") as directory:
        for page_index, page in enumerate(document):
            values = by_page.get(page_index + 1)
            if values is None:
                # Check before allocating a raster, not after a potentially huge page.
                scale = dpi / 72
                width, height = page.rect.width * scale, page.rect.height * scale
                if (int(width) + 1) * (int(height) + 1) > ocr.max_image_pixels:
                    raise ValueError("Rendered PDF page exceeds OCR pixel budget")
                rendered = Path(directory) / "page.png"
                pixmap = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
                pixmap.save(rendered)
                values = ocr.extract_units(
                    rendered,
                    asset_id=asset_id,
                    first_unit_id=first_unit_id + len(result),
                    page=page_index + 1,
                    metadata_values=metadata_values,
                )
            if len(result) + len(values) > max_blocks:
                raise ValueError("PDF exceeds combined native/OCR block budget")
            next_id = first_unit_id + len(result)
            result.extend(dataclasses.replace(unit, unit_id=next_id + offset) for offset, unit in enumerate(values))
    return tuple(result)
