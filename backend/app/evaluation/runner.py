from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from backend.app.core.config import CLIP_MODEL, TEXT_MODEL
from backend.app.db.session import SessionLocal
from backend.app.evaluation.dataset import load_cases
from backend.app.evaluation.metrics import aggregate_metrics
from backend.app.models.local_clip import build_embedder
from backend.app.retrieval.photo_search import PhotoSearch


def run(fixture: Path, *, embedder_kind: str = "hybrid", device: str = "cpu", limit: int = 10) -> dict[str, object]:
    cases = load_cases(fixture)
    visual = build_embedder("hash" if embedder_kind == "hash" else "clip", CLIP_MODEL, device)
    text = visual if embedder_kind == "hash" else build_embedder("bge", TEXT_MODEL, device)
    db = SessionLocal()
    try:
        search = PhotoSearch(db, visual, text)
        rows = []
        latencies: list[float] = []
        for case in cases:
            started = time.perf_counter()
            _answer, hits = search.search(case.query, limit=limit)
            latencies.append((time.perf_counter() - started) * 1000)
            rows.append((case, [hit.file_id for hit in hits]))
    finally:
        db.close()
    report = aggregate_metrics(rows)
    report["latency_ms"] = {"p50": _percentile(latencies, 0.50), "p95": _percentile(latencies, 0.95)}
    report["config"] = {"fixture": str(fixture), "embedder_kind": embedder_kind, "device": device, "limit": limit}
    return report


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int((len(ordered) - 1) * percentile))], 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--embedder-kind", choices=["hash", "hybrid"], default="hybrid")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.fixture, embedder_kind=args.embedder_kind, device=args.device, limit=args.limit)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
