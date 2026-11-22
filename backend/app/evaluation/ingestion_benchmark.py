"""Offline staged photo-ingestion benchmark and fault-recovery harness.

Measures isolated synthetic photo ingestion without touching private libraries,
downloading weights, or modifying production databases. Import and ``--help``
remain safe under ``python -S``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


def percentile(values: list[float], ratio: float) -> float:
    """Compute a linear-interpolated percentile over a non-empty numeric list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * ratio
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return round(ordered[lower], 3)
    weight = index - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 3)


def generate_synthetic_images(
    directory: Path,
    count: int,
    *,
    width: int = 320,
    height: int = 240,
    include_rotated: bool = True,
) -> list[Path]:
    """Generate deterministic synthetic JPEG fixtures in an isolated directory."""
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(count):
        image = Image.new("RGB", (width, height), ((index * 37) % 255, (index * 73) % 255, (index * 113) % 255))
        exif = image.getexif()
        if include_rotated and index % 3 == 0:
            exif[274] = 6  # 90 deg CW orientation
        path = directory / f"img_{index:04d}.jpg"
        image.save(path, "JPEG", quality=85, exif=exif.tobytes())
        paths.append(path)
    return paths


def _run_staged_worker(
    work_dir: Path,
    photo_dir: Path,
    *,
    embedder_kind: str = "hash",
    vision_backend: str = "basic",
    device: str = "cpu",
    max_pixels: int = 768 * 768,
    db_name: str = "bench.db",
) -> dict[str, Any]:
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from backend.app.db import models  # noqa: F401
    from backend.app.db.models import PhotoVector
    from backend.app.db.session import Base
    from backend.app.workers import image_worker
    from backend.app.workers.image_worker import ImageWorker, enqueue_photo_index

    db_path = work_dir / db_name
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    original_data_dir = image_worker.DATA_DIR
    image_worker.DATA_DIR = work_dir / "data"
    try:
        with Session(engine) as db:
            worker = ImageWorker("ingestion-benchmark")
            payload = {
                "root": str(photo_dir),
                "vision_backend": vision_backend,
                "captioner_kind": "basic" if vision_backend == "basic" else "qwen-vl",
                "embedder_kind": embedder_kind,
                "device": device,
                "max_pixels": max_pixels,
            }
            cold_start = time.perf_counter()
            cold_stats = worker._index_photos(db, payload)
            cold_elapsed = time.perf_counter() - cold_start

            warm_start = time.perf_counter()
            warm_stats = worker._index_photos(db, payload)
            warm_elapsed = time.perf_counter() - warm_start

            persisted_vectors = len(db.scalars(select(PhotoVector.id)).all())
            task = enqueue_photo_index(db, payload)
            return {
                "cold_elapsed_s": round(cold_elapsed, 4),
                "cold_throughput_img_s": round(cold_stats.indexed / max(cold_elapsed, 1e-6), 3),
                "warm_elapsed_s": round(warm_elapsed, 4),
                "warm_skipped": warm_stats.skipped,
                "indexed": cold_stats.indexed,
                "failed": cold_stats.failed,
                "vectors_written": cold_stats.vectors_written,
                "persisted_vectors": persisted_vectors,
                "stage_timings_ms": dict(cold_stats.stage_timings_ms),
                "sample_task_id": task.id,
            }
    finally:
        image_worker.DATA_DIR = original_data_dir
        engine.dispose()


def _run_prefetch_comparison(paths: list[Path], thumb_dir: Path, *, queue_bound: int = 4) -> dict[str, Any]:
    """Compare synchronous inspect+thumbnail+embed vs bounded-queue CPU prefetch."""
    from backend.app.ingestion.images import inspect_image, make_thumbnail
    from backend.app.models.local_clip import HashEmbedder

    embedder = HashEmbedder()

    sync_latencies: list[float] = []
    sync_start = time.perf_counter()
    for path in paths:
        t0 = time.perf_counter()
        inspect_image(path)
        make_thumbnail(path, thumb_dir / "sync")
        embedder.embed_image(path)
        sync_latencies.append((time.perf_counter() - t0) * 1000.0)
    sync_elapsed = time.perf_counter() - sync_start

    bounded: queue.Queue[Path | None] = queue.Queue(maxsize=max(1, queue_bound))
    max_depth = 0
    prefetched: list[tuple[Path, Any]] = []
    lock = threading.Lock()

    def producer() -> None:
        nonlocal max_depth
        for path in paths:
            meta = inspect_image(path)
            make_thumbnail(path, thumb_dir / "prefetch")
            bounded.put(path)
            with lock:
                max_depth = max(max_depth, bounded.qsize())
                prefetched.append((path, meta))
        bounded.put(None)

    prefetch_latencies: list[float] = []
    prefetch_start = time.perf_counter()
    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    while True:
        item = bounded.get()
        if item is None:
            break
        t0 = time.perf_counter()
        embedder.embed_image(item)
        prefetch_latencies.append((time.perf_counter() - t0) * 1000.0)
    thread.join(timeout=30)
    prefetch_elapsed = time.perf_counter() - prefetch_start

    return {
        "image_count": len(paths),
        "queue_bound": queue_bound,
        "observed_max_queue_depth": max_depth,
        "sync_elapsed_s": round(sync_elapsed, 4),
        "sync_p50_ms": percentile(sync_latencies, 0.50),
        "sync_p95_ms": percentile(sync_latencies, 0.95),
        "prefetch_elapsed_s": round(prefetch_elapsed, 4),
        "prefetch_consumer_p50_ms": percentile(prefetch_latencies, 0.50),
        "prefetch_consumer_p95_ms": percentile(prefetch_latencies, 0.95),
    }


def _run_concurrency_sweep(paths: list[Path], workers_list: list[int]) -> list[dict[str, Any]]:
    """Measure isolated CPU hashing/inspection concurrency scaling."""
    from backend.app.ingestion.images import inspect_image

    results: list[dict[str, Any]] = []
    for workers in workers_list:
        started = time.perf_counter()
        if workers <= 1:
            metas = [inspect_image(path) for path in paths]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                metas = list(pool.map(inspect_image, paths))
        elapsed = time.perf_counter() - started
        results.append(
            {
                "concurrency": workers,
                "count": len(metas),
                "elapsed_s": round(elapsed, 4),
                "throughput_img_s": round(len(metas) / max(elapsed, 1e-6), 3),
            }
        )
    return results


def _run_fault_matrix(work_dir: Path) -> dict[str, Any]:
    """Run synthetic fault-injection and recovery checks (not an exactly-once claim)."""
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from backend.app.db import models  # noqa: F401
    from backend.app.db.models import File, PhotoVector
    from backend.app.db.session import Base
    from backend.app.models.vision_service import VisionServiceUnavailable
    from backend.app.workers import image_worker
    from backend.app.workers.image_worker import ImageWorker

    photo_dir = work_dir / "fault_photos"
    generated = generate_synthetic_images(photo_dir, 3, width=64, height=64)
    corrupt_path = photo_dir / "img_0000.jpg"
    corrupt_path.write_bytes(b"not-a-valid-jpeg")

    engine = create_engine(f"sqlite:///{work_dir / 'fault.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    original_data_dir = image_worker.DATA_DIR
    image_worker.DATA_DIR = work_dir / "fault_data"
    try:
        with Session(engine) as db:
            worker = ImageWorker("fault-matrix")
            # Scenario 1: simulated vision OOM / unavailable failure on 2nd image
            base_vision = worker.models.vision_client
            oom_triggered = {"done": False}

            class OomVisionClient:
                def analyze_photo(self, image_path: Path) -> dict[str, object]:
                    if image_path == generated[1] and not oom_triggered["done"]:
                        oom_triggered["done"] = True
                        raise VisionServiceUnavailable("Simulated CUDA OOM")
                    return {"caption": f"synthetic {image_path.name}", "tags": ["synthetic"]}

            worker.models.vision_client = lambda _cfg: OomVisionClient()  # type: ignore[method-assign]
            try:
                first = worker._index_photos(
                    db,
                    {
                        "root": str(photo_dir),
                        "vision_backend": "basic",
                        "embedder_kind": "hash",
                        "device": "cpu",
                        "retry_cooldown_seconds": 60,
                    },
                )
                # Scenario 2: rerun immediately - failed image is in cooldown, indexed images skipped cleanly
                second = worker._index_photos(
                    db,
                    {
                        "root": str(photo_dir),
                        "vision_backend": "basic",
                        "embedder_kind": "hash",
                        "device": "cpu",
                        "retry_cooldown_seconds": 60,
                    },
                )
                # Scenario 3: expire cooldown on OOM image and recover idempotently
                oom_row = db.scalar(select(File).where(File.canonical_path == str(generated[1])))
                if oom_row:
                    oom_row.next_retry_at = None
                    db.commit()
                recovered = worker._index_photos(
                    db,
                    {
                        "root": str(photo_dir),
                        "vision_backend": "basic",
                        "embedder_kind": "hash",
                        "device": "cpu",
                        "retry_cooldown_seconds": 60,
                    },
                )
                indexed_files = len(db.scalars(select(File.id).where(File.ingestion_status == "indexed")).all())
                vector_rows = len(db.scalars(select(PhotoVector.id)).all())
                return {
                    "first_pass": {"indexed": first.indexed, "failed": first.failed},
                    "cooldown_pass": {
                        "indexed": second.indexed,
                        "failed": second.failed,
                        "cooldown_skipped": second.cooldown_skipped,
                    },
                    "recovery_pass": {"indexed": recovered.indexed, "failed": recovered.failed},
                    "final_indexed_files": indexed_files,
                    "final_vector_rows": vector_rows,
                    "guarantee": "at-least-once with idempotent vector replacement (not exactly-once)",
                }
            finally:
                worker.models.vision_client = base_vision  # type: ignore[method-assign]
    finally:
        image_worker.DATA_DIR = original_data_dir
        engine.dispose()


def run_benchmark(
    work_dir: Path,
    *,
    count: int = 12,
    resolutions: tuple[tuple[int, int], ...] = ((256, 256), (512, 512)),
    concurrencies: tuple[int, ...] = (1, 2, 4),
    embedder_kind: str = "hash",
    vision_backend: str = "basic",
    device: str = "cpu",
) -> dict[str, Any]:
    """Execute the complete isolated ingestion benchmark and fault-recovery matrix."""
    work_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = work_dir / "photos_baseline"
    baseline_paths = generate_synthetic_images(baseline_dir, count, width=320, height=240)
    baseline = _run_staged_worker(
        work_dir,
        baseline_dir,
        embedder_kind=embedder_kind,
        vision_backend=vision_backend,
        device=device,
        db_name="baseline.db",
    )
    prefetch = _run_prefetch_comparison(baseline_paths, work_dir / "thumbs", queue_bound=4)
    concurrency = _run_concurrency_sweep(baseline_paths, list(concurrencies))

    resolution_sweep: list[dict[str, Any]] = []
    for width, height in resolutions:
        res_dir = work_dir / f"photos_{width}x{height}"
        generate_synthetic_images(res_dir, max(3, count // 3), width=width, height=height)
        res_report = _run_staged_worker(
            work_dir,
            res_dir,
            embedder_kind=embedder_kind,
            vision_backend=vision_backend,
            device=device,
            max_pixels=width * height,
            db_name=f"res_{width}x{height}.db",
        )
        resolution_sweep.append({"width": width, "height": height, **res_report})

    fault_matrix = _run_fault_matrix(work_dir)
    return {
        "mode": {"embedder_kind": embedder_kind, "vision_backend": vision_backend, "device": device},
        "baseline": baseline,
        "prefetch_comparison": prefetch,
        "concurrency_sweep": concurrency,
        "resolution_sweep": resolution_sweep,
        "fault_matrix": fault_matrix,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated photo-ingestion staged benchmark and fault matrix.")
    parser.add_argument("--work-dir", type=Path, help="Fresh isolated directory for synthetic fixtures and SQLite DB.")
    parser.add_argument("--output", type=Path, help="Write JSON report to this path (must not already exist).")
    parser.add_argument("--count", type=int, default=12, help="Number of synthetic images for baseline run.")
    parser.add_argument("--embedder-kind", default="hash", choices=["hash", "clip"])
    parser.add_argument("--vision-backend", default="basic", choices=["basic", "vllm", "transformers"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    if args.output is not None and args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")

    if args.work_dir is not None:
        report = run_benchmark(
            args.work_dir,
            count=args.count,
            embedder_kind=args.embedder_kind,
            vision_backend=args.vision_backend,
            device=args.device,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="lentica-ingest-bench-") as tmp:
            report = run_benchmark(
                Path(tmp),
                count=args.count,
                embedder_kind=args.embedder_kind,
                vision_backend=args.vision_backend,
                device=args.device,
            )

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
