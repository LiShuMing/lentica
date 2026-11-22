from __future__ import annotations

import asyncio
import io
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi import HTTPException
from PIL import ExifTags, Image as PilImage
from PIL.TiffImagePlugin import IFDRational
from sqlalchemy import Column, MetaData, Table, create_engine, select

from backend.app.api import routes
from backend.app.db.migrations import migrate_ingestion_reliability
from backend.app.db.models import FaceDetection, File, Image, PhotoVector, ProcessingRun, Source
from backend.app.evaluation import ingestion_benchmark
from backend.app.ingestion import scanner
from backend.app.ingestion.images import inspect_image, make_thumbnail
from backend.app.ingestion.scanner import scan_documents
from backend.app.models.local_face import InsightFaceDetector
from backend.app.models.vision_service import BasicVisionClient, FallbackVisionClient, VisionServiceUnavailable
from backend.app.workers import image_worker, queue_controller
from backend.app.workers.image_worker import ImageWorker, enqueue_photo_index
from backend.app.workers.task_store import recover_expired


def _make_exif_jpeg(
    path: Path,
    *,
    size: tuple[int, int] = (40, 20),
    orientation: int = 6,
    taken_at: str = "2024:07:15 09:30:45",
    lat: tuple[IFDRational, IFDRational, IFDRational] = (
        IFDRational(33, 1),
        IFDRational(52, 1),
        IFDRational(72, 10),
    ),
    lat_ref: str = "S",
    lon: tuple[IFDRational, IFDRational, IFDRational] = (
        IFDRational(151, 1),
        IFDRational(12, 1),
        IFDRational(36, 10),
    ),
    lon_ref: str = "W",
) -> Path:
    img = PilImage.new("RGB", size, (255, 0, 0))
    # Paint the left half blue so rotation direction is verifiable in pixel space.
    for x in range(size[0] // 2):
        for y in range(size[1]):
            img.putpixel((x, y), (0, 0, 255))
    exif = PilImage.Exif()
    exif[274] = orientation
    exif.get_ifd(ExifTags.IFD.Exif)[36867] = taken_at  # DateTimeOriginal
    gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    gps_ifd[1] = lat_ref
    gps_ifd[2] = lat
    gps_ifd[3] = lon_ref
    gps_ifd[4] = lon
    img.save(path, "JPEG", quality=95, exif=exif.tobytes())
    return path


def test_nested_exif_and_signed_gps_extraction(tmp_path):
    photo_path = _make_exif_jpeg(tmp_path / "rotated_geo.jpg", size=(40, 20), orientation=6)
    meta = inspect_image(photo_path)
    assert meta.orientation == 6
    assert (meta.width, meta.height) == (20, 40)
    assert meta.exif_taken_at == datetime(2024, 7, 15, 9, 30, 45)
    assert meta.gps_lat is not None and meta.gps_lat < 0
    assert meta.gps_lon is not None and meta.gps_lon < 0
    assert abs(meta.gps_lat - -(33 + 52 / 60 + 7.2 / 3600)) < 1e-4
    assert abs(meta.gps_lon - -(151 + 12 / 60 + 3.6 / 3600)) < 1e-4


def test_corrupt_image_and_malformed_exif_are_safe(tmp_path):
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not-a-jpeg")
    meta = inspect_image(corrupt)
    assert meta.sha256 and meta.width is None and meta.exif_taken_at is None
    assert make_thumbnail(corrupt, tmp_path / "thumbs") is None

    zero_den = tmp_path / "zero_den.jpg"
    img = PilImage.new("RGB", (16, 16), "green")
    exif = PilImage.Exif()
    exif.get_ifd(ExifTags.IFD.Exif)[36867] = "bad-date"
    exif.get_ifd(ExifTags.IFD.GPSInfo)[2] = (IFDRational(1, 0), IFDRational(0, 1), IFDRational(0, 1))
    img.save(zero_den, "JPEG", exif=exif.tobytes())
    zero_meta = inspect_image(zero_den)
    assert (zero_meta.width, zero_meta.height) == (16, 16)
    assert zero_meta.exif_taken_at is None and zero_meta.gps_lat is None


def test_rotated_thumbnail_face_detection_and_face_crop_agree(db, tmp_path):
    photo_path = _make_exif_jpeg(tmp_path / "portrait_from_landscape.jpg", size=(40, 20), orientation=6)
    thumb = make_thumbnail(photo_path, tmp_path / "thumbs", size=128)
    assert thumb is not None
    with PilImage.open(thumb) as rendered:
        assert rendered.size == (20, 40)
        # Original left half was blue; after 90 deg CW (orientation 6) top half is blue.
        top_pixel = rendered.getpixel((10, 5))
        bottom_pixel = rendered.getpixel((10, 35))
        assert top_pixel[2] > top_pixel[0] and bottom_pixel[0] > bottom_pixel[2]

    # Verify InsightFaceDetector passes transposed (40x20 HxW) pixels to app.get.
    seen_shapes = []

    class FakeApp:
        def get(self, array):
            seen_shapes.append(array.shape)
            return []

    detector = InsightFaceDetector.__new__(InsightFaceDetector)
    detector.app = FakeApp()
    detector.detect(photo_path)
    assert seen_shapes == [(40, 20, 3)]

    # Verify /api/people/faces/{id}/thumb crops in the exact same transposed space.
    source = Source(name="Faces", root_path=str(tmp_path), kind="images")
    db.add(source)
    db.flush()
    file = File(
        source_id=source.id,
        canonical_path=str(photo_path),
        windows_path=photo_path.name,
        file_type="jpg",
        ingestion_status="indexed",
    )
    db.add(file)
    db.flush()
    face = FaceDetection(file_id=file.id, bbox_json=json.dumps([2, 2, 18, 12]), detection_score="0.99")
    db.add(face)
    db.commit()

    response = routes.face_thumbnail(face.id, db)

    async def _read_body() -> bytes:
        chunks = [chunk async for chunk in response.body_iterator]
        return b"".join(chunks)

    content = asyncio.run(_read_body())
    assert content
    with PilImage.open(io.BytesIO(content)) as crop:
        center = crop.getpixel((crop.width // 2, crop.height // 2))
        assert center[2] > center[0]


def test_real_vector_counts_fallbacks_and_stage_timings_match_persisted_rows(db, tmp_path, monkeypatch):
    photo_path = _make_exif_jpeg(tmp_path / "scene.jpg", size=(32, 24), orientation=1)
    monkeypatch.setattr(image_worker, "DATA_DIR", tmp_path / "data")

    class FailingPrimary:
        def analyze_photo(self, path):
            raise VisionServiceUnavailable("vllm offline")

    worker = ImageWorker("obs-fixture")
    fallback_client = FallbackVisionClient(FailingPrimary(), BasicVisionClient)
    monkeypatch.setattr(worker.models, "vision_client", lambda _cfg: fallback_client)

    task = enqueue_photo_index(
        db,
        {
            "root": str(tmp_path),
            "vision_backend": "vllm",
            "vision_fallback": "basic",
            "embedder_kind": "hash",
            "device": "cpu",
        },
    )
    worker._run_task(db, worker._claim_task(db))
    assert task.state == "succeeded"

    run = db.scalar(select(ProcessingRun).where(ProcessingRun.task_id == task.id))
    stats = json.loads(run.output_artifact_ids)
    persisted_count = len(db.scalars(select(PhotoVector.id)).all())

    assert stats["indexed"] == 1
    assert stats["vectors_written"] == persisted_count
    assert stats["image_vectors_written"] == 1
    assert stats["text_vectors_written"] == persisted_count - 1
    assert stats["vision_calls"] == 1
    assert stats["vision_fallbacks"] == 1
    assert stats["degraded"] is True
    assert stats["image_embed_calls"] == 1
    assert stats["text_embed_calls"] == 1
    for stage_key in (
        "inspect_exif_ms",
        "thumbnail_ms",
        "vision_ms",
        "image_embed_ms",
        "text_embed_ms",
        "db_commit_ms",
    ):
        assert stage_key in stats["stage_timings_ms"]
        assert stats["stage_timings_ms"][stage_key] >= 0.0

    load = routes._computational_load("index_photos", stats)
    assert load["total_vectors_written"] == persisted_count
    assert load["image_vectors_written"] == 1
    assert load["text_vectors_written"] == persisted_count - 1
    assert load["vision_fallbacks"] == 1
    assert load["degraded"] is True
    assert photo_path.exists()


def test_corrupt_queue_settings_returns_http_503(tmp_path, monkeypatch):
    bad_settings = tmp_path / "queue.json"
    bad_settings.write_text("{corrupt", encoding="utf-8")
    monkeypatch.setattr(queue_controller, "SETTINGS_PATH", bad_settings)
    with pytest.raises(HTTPException) as exc_info:
        routes.queue_status()
    assert exc_info.value.status_code == 503
    assert "refusing to enable defaults" in str(exc_info.value.detail)


def test_document_retry_budget_cooldown_and_hash_change_reset(db, tmp_path, monkeypatch):
    source = Source(name="Docs", root_path=str(tmp_path), kind="documents")
    db.add(source)
    db.commit()
    broken = tmp_path / "0_broken.txt"
    healthy = tmp_path / "1_healthy.txt"
    broken.write_text("bad-content", encoding="utf-8")
    healthy.write_text("healthy-content", encoding="utf-8")

    real_parse = scanner.parse_document

    def fail_on_broken(path: Path):
        if path.read_text(encoding="utf-8") == "bad-content":
            raise ValueError("corrupt document")
        return real_parse(path)

    monkeypatch.setattr(scanner, "parse_document", fail_on_broken)
    t0 = datetime(2026, 11, 21, 12, 0, 0)

    # First limit=1 scan records failure + cooldown on broken file.
    job1 = scan_documents(db, source_id=source.id, max_files=1, max_retries=2, retry_cooldown_seconds=60, now=t0)
    assert job1.status == "failed" and job1.failed_files == 1
    # Second limit=1 scan advances past broken file and indexes healthy file cleanly.
    job2 = scan_documents(db, source_id=source.id, max_files=1, max_retries=2, retry_cooldown_seconds=60, now=t0)
    assert job2.status == "succeeded" and job2.indexed_files == 1

    # Wrapping back during cooldown skips broken file without failing the job.
    job3 = scan_documents(
        db, source_id=source.id, max_files=1, max_retries=2, retry_cooldown_seconds=60, now=t0 + timedelta(seconds=30)
    )
    assert job3.status == "succeeded" and job3.skipped_files == 1 and job3.failed_files == 0

    # After cooldown expires, second attempt fails and exhausts max_retries=2.
    job4 = scan_documents(
        db,
        source_id=source.id,
        max_files=None,
        max_retries=2,
        retry_cooldown_seconds=60,
        now=t0 + timedelta(seconds=61),
    )
    assert job4.status == "failed" and job4.failed_files == 1
    broken_row = db.scalar(select(File).where(File.canonical_path == str(broken)))
    assert broken_row.failure_count == 2

    # Even far in the future, exhausted unchanged file stays skipped.
    job5 = scan_documents(
        db, source_id=source.id, max_files=None, max_retries=2, retry_cooldown_seconds=60, now=t0 + timedelta(hours=5)
    )
    assert job5.status == "succeeded" and job5.failed_files == 0

    # Fixing file bytes resets hash and succeeds immediately.
    broken.write_text("fixed-content", encoding="utf-8")
    job6 = scan_documents(
        db, source_id=source.id, max_files=None, max_retries=2, retry_cooldown_seconds=60, now=t0 + timedelta(hours=5)
    )
    assert job6.status == "succeeded" and job6.indexed_files == 1
    db.refresh(broken_row)
    assert broken_row.ingestion_status == "indexed" and broken_row.failure_count == 0


def test_photo_failure_cooldown_and_task_recovery_backoff(db, tmp_path, monkeypatch):
    monkeypatch.setattr(image_worker, "DATA_DIR", tmp_path / "data")
    bad = tmp_path / "0_bad.png"
    good = tmp_path / "1_good.png"
    PilImage.new("RGB", (16, 16), "red").save(bad)
    PilImage.new("RGB", (16, 16), "green").save(good)

    real_inspect = image_worker.inspect_image

    def fail_bad(path: Path):
        if path == bad:
            raise OSError("injected read error")
        return real_inspect(path)

    monkeypatch.setattr(image_worker, "inspect_image", fail_bad)
    worker = ImageWorker("cooldown-fixture")

    first = worker._index_photos(
        db,
        {
            "root": str(tmp_path),
            "vision_backend": "basic",
            "embedder_kind": "hash",
            "device": "cpu",
            "max_retries": 2,
            "retry_cooldown_seconds": 300,
        },
    )
    assert first.failed == 1 and first.indexed == 1

    # Immediate re-run skips the failing photo under cooldown rather than spinning.
    second = worker._index_photos(
        db,
        {
            "root": str(tmp_path),
            "vision_backend": "basic",
            "embedder_kind": "hash",
            "device": "cpu",
            "max_retries": 2,
            "retry_cooldown_seconds": 300,
        },
    )
    assert second.failed == 0 and second.cooldown_skipped == 1

    # Verify optional exponential backoff on recover_expired.
    task = enqueue_photo_index(db, {"root": str(tmp_path), "limit": 1})
    claimed = worker._claim_task(db)
    now = datetime(2026, 11, 21, 13, 0, 0)
    claimed.lease_expires_at = now - timedelta(seconds=1)
    db.commit()
    assert recover_expired(db, now=now, backoff_seconds=30) == 1
    db.refresh(claimed)
    assert claimed.state == "queued"
    assert claimed.next_attempt_at == now + timedelta(seconds=30)


def test_additive_v3_migration_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'v3.db'}")
    legacy = MetaData()
    Table(
        "files",
        legacy,
        Column("id", Image.id.type, primary_key=True),
        Column("canonical_path", File.canonical_path.type),
    )
    legacy.create_all(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("INSERT INTO files(id, canonical_path) VALUES (1, '/a.txt')")
        migrate_ingestion_reliability(conn)
        migrate_ingestion_reliability(conn)
        assert conn.exec_driver_sql("SELECT canonical_path, failure_count, next_retry_at FROM files").one() == (
            "/a.txt",
            0,
            None,
        )
    engine.dispose()


def test_ingestion_benchmark_cli_help_and_synthetic_run(tmp_path):
    help_run = subprocess.run(
        [sys.executable, "-S", "-m", "backend.app.evaluation.ingestion_benchmark", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--work-dir" in help_run.stdout

    report = ingestion_benchmark.run_benchmark(
        tmp_path / "bench",
        count=4,
        resolutions=((64, 64), (128, 128)),
        concurrencies=(1, 2),
    )
    assert report["baseline"]["indexed"] == 4
    assert report["baseline"]["vectors_written"] == report["baseline"]["persisted_vectors"]
    assert report["baseline"]["warm_skipped"] == 4
    assert report["prefetch_comparison"]["observed_max_queue_depth"] <= 4
    assert len(report["resolution_sweep"]) == 2
    assert report["fault_matrix"]["first_pass"]["failed"] == 1
    assert report["fault_matrix"]["cooldown_pass"]["cooldown_skipped"] == 1
    assert report["fault_matrix"]["recovery_pass"]["indexed"] == 1
    assert report["fault_matrix"]["final_indexed_files"] == 3
