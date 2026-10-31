from __future__ import annotations

import json
import subprocess
import sys

import pytest

from backend.app.evaluation.hybrid_benchmark import load_dataset, run


def write_fixture(path):
    payload = {
        "documents": [
            {"asset_id": 0, "text": "battery charging connector power adapter"},
            {"asset_id": 1, "text": "power supply charger battery connector"},
            {"asset_id": 2, "text": "network socket connection error ERR_301"},
            {"asset_id": 3, "text": "wireless network router disconnect socket"},
            {"asset_id": 4, "text": "camera photo lens shutter aperture"},
            {"asset_id": 5, "text": "photo image camera focus lens"},
        ],
        "queries": [
            {"query_id": "q1", "query": "ERR_301", "relevant_asset_ids": [2]},
            {"query_id": "q2", "query": "battery power adapter", "relevant_asset_ids": [0, 1]},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_benchmark_repeats_rankings_and_uses_real_adapters(tmp_path):
    path = tmp_path / "fixture.json"
    write_fixture(path)
    dataset = load_dataset(path)
    first, second = [run(dataset, dimensions=3, seed=42) for _ in range(2)]
    assert first["ranking_sha256"] == second["ranking_sha256"]
    assert len(first["configurations"]) == 11
    assert first["final_fetch"]["calls"] == 22
    assert first["final_fetch"]["max_records_per_call"] <= 10
    assert first["encoder"]["dimensions"] == 3
    assert first["configurations"]["text-only"]["metrics"]["overall"]["recall_at_10"] == 1
    for row in first["configurations"].values():
        assert row["metrics"]["queries"] == 2
        assert all(0 <= value <= 1 for value in row["metrics"]["overall"].values())


def test_benchmark_rejects_judgments_outside_corpus(tmp_path):
    path = tmp_path / "fixture.json"
    payload = write_fixture(path)
    payload["queries"][0]["relevant_asset_ids"] = [999]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="judgments"):
        load_dataset(path)


def test_benchmark_help_does_not_load_models_or_database():
    subprocess.run(
        [sys.executable, "-S", "-m", "backend.app.evaluation.hybrid_benchmark", "--help"],
        check=True,
        capture_output=True,
        timeout=10,
    )
