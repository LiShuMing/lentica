"""Real optional OCR/native PDF behavior and local artifact identities."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest

from backend.app.retrieval.hybrid import artifacts, extraction, pair_reranking, semantic


def test_local_artifact_verification_rejects_changes_and_outside_paths(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("configuration")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"files": {"config.json": {"bytes": path.stat().st_size, "sha256": digest}}}
    (tmp_path / "lentica-artifacts.json").write_text(json.dumps(manifest))
    assert artifacts.verify(tmp_path, ["config.json"]) == {"config.json": digest}
    with pytest.raises(ValueError, match="inside"):
        artifacts.verify(tmp_path, ["../outside"])
    path.write_text("modified")
    with pytest.raises(ValueError, match="mismatch"):
        artifacts.verify(tmp_path, ["config.json"])
    assert artifacts.space_id("test", {"config": digest}, {"prefix": "query: "}) != artifacts.space_id(
        "test", {"config": digest}, {"prefix": "passage: "}
    )


def test_native_profiles_keep_bge_and_e5_pooling_prefixes_explicit():
    assert semantic.TextProfile().pooling == "cls"
    assert semantic.TextProfile(family="bge-m3").query_prefix == ""
    e5 = semantic.TextProfile(family="e5")
    assert e5.pooling == "mean-mask" and e5.query_prefix == "query: " and e5.document_prefix == "passage: "
    with pytest.raises(ValueError):
        semantic.TextProfile(family="unknown")
    with pytest.raises(ValueError):
        semantic.TextProfile(max_tokens=0)


def test_native_pdf_blocks_preserve_page_region_rotation_and_original_bytes(tmp_path):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "manual.pdf"
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((50, 60), "Native block ERR_429")
        rotated = document.new_page()
        rotated.insert_text((50, 60), "Rotated page")
        rotated.set_rotation(90)
        document.new_page()
        document.save(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    values = extraction.pdf_blocks(path, asset_id=10, first_unit_id=100, metadata_values={"tenant": "a"})
    assert [unit.unit_id for unit in values] == [100, 101]
    assert [unit.locator.page for unit in values] == [1, 2]
    assert all(unit.locator.bbox is not None and unit.metadata["tenant"] == "a" for unit in values)
    assert all(item.evidence[0].kind == "native" for item in values)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    with pytest.raises(ValueError, match="page budget"):
        extraction.pdf_blocks(path, asset_id=10, first_unit_id=0, max_pages=2)


def test_actual_packaged_ocr_preserves_regions_and_model_provenance(tmp_path):
    pytest.importorskip("rapidocr_onnxruntime")
    from PIL import Image, ImageDraw, ImageFont

    path = tmp_path / "screen.png"
    image = Image.new("RGB", (800, 160), "white")
    ImageDraw.Draw(image).text((30, 45), "ERROR 429 RETRY LATER", font=ImageFont.load_default(size=40), fill="black")
    image.save(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    extractor = extraction.RapidOcrExtractor()
    values = extractor.extract_units(path, asset_id=5, first_unit_id=20, metadata_values={"tenant": "a"})
    assert values and "ERROR" in " ".join(unit.text() for unit in values)
    assert all(unit.locator.bbox is not None for unit in values)
    assert all(
        unit.evidence[0].producer == "rapidocr-onnxruntime" and unit.evidence[0].version == extractor.version
        for unit in values
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    extractor.max_image_pixels = 10
    with pytest.raises(ValueError, match="pixel budget"):
        extractor.recognize(path)


def test_model_and_extraction_modules_import_without_runtime_dependencies():
    code = """
import sys
from backend.app.retrieval.hybrid import semantic, clip_onnx, siglip_onnx, vision_reranking, audio_onnx, pdf_layout, extraction, ann, fulltext, execution, pair_reranking, bundles, caching
assert not any(name.startswith(('numpy', 'torch', 'transformers', 'onnxruntime', 'PIL', 'tantivy', 'faiss')) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-S", "-c", code], check=True, timeout=10)


@pytest.mark.parametrize("kwargs", [{"threads": 0}, {"batch_size": 65}, {"max_tokens": 513}, {"max_candidates": 1001}])
def test_pair_budgets_fail_before_loading_artifacts_or_dependencies(tmp_path, kwargs):
    with pytest.raises(ValueError, match="budgets"):
        pair_reranking.PairReranker(tmp_path / "no-model", **kwargs)


def test_scanned_pdf_actual_ocr_keeps_rotation_pages_filters_and_source(tmp_path):
    fitz = pytest.importorskip("fitz")
    pytest.importorskip("rapidocr_onnxruntime")
    from PIL import Image, ImageDraw, ImageFont

    image_path = tmp_path / "scan.png"
    image = Image.new("RGB", (800, 180), "white")
    ImageDraw.Draw(image).text((30, 40), "SCANNED ERROR 429", font=ImageFont.load_default(size=40), fill="black")
    image.save(image_path)
    path = tmp_path / "scanned.pdf"
    with fitz.open() as document:
        page = document.new_page(width=800, height=180)
        page.insert_text((30, 70), "Native page ERR_200")
        page = document.new_page(width=800, height=180)
        page.insert_image(page.rect, filename=str(image_path))
        page.set_rotation(90)
        document.save(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    ocr = extraction.RapidOcrExtractor()
    values = extraction.pdf_with_ocr(path, ocr, asset_id=30, first_unit_id=100, metadata_values={"tenant": "a"})
    assert [unit.unit_id for unit in values] == list(range(100, 100 + len(values)))
    assert {unit.locator.page for unit in values} == {1, 2}
    assert any("ERROR" in unit.text() and unit.evidence[0].kind == "ocr" for unit in values if unit.locator.page == 2)
    assert all(unit.evidence[0].kind == "native" for unit in values if unit.locator.page == 1)
    assert all(unit.locator.bbox and unit.metadata["tenant"] == "a" for unit in values)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    with pytest.raises(ValueError, match="combined"):
        extraction.pdf_with_ocr(path, ocr, asset_id=30, first_unit_id=100, max_blocks=1)
    ocr.max_image_pixels = 100
    with pytest.raises(ValueError, match="pixel budget"):
        extraction.pdf_with_ocr(path, ocr, asset_id=30, first_unit_id=100)
