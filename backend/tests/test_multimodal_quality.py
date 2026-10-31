"""Evidence-based diagnostics and image/PDF reference experiments."""

from __future__ import annotations

import subprocess
import sys

import pytest

from backend.app.evaluation.dataset import EvaluationCase
from backend.app.evaluation.hybrid_eval import diagnostics, multimodal, quality
from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import media, units


def test_candidate_loss_attribution_and_exclusive_pairs_are_independent_of_ranking():
    cases = [EvaluationCase("q", "query", {1, 2, 3, 4}, "fixture")]
    pools = {"q": {"text": [1, 2, 9], "visual": [2, 3, 8]}}
    report = diagnostics.candidate_report(
        cases, pools, {"q": [1]}, windows={"text": 2, "visual": 2}, candidates={"q": [1, 2]}
    )
    assert report["mean_bounded_union_recall"] == 0.75
    assert report["mean_fused_candidate_recall"] == 0.5
    assert report["exclusive_relevant_pairs_at_configured_windows"] == {"text": 1, "visual": 1}
    row = report["per_query"][0]
    assert [
        row[key] for key in ("not_in_route_union", "lost_at_candidate_fusion", "lost_after_candidates", "found_top10")
    ] == [1, 1, 1, 1]
    assert row["oracle_ndcg_at_10"] > row["ndcg_at_10"]
    assert report["by_category"]["fixture"]["queries"] == 1


def test_empty_diagnostic_judgments_fail_explicitly():
    with pytest.raises(ValueError, match="judged"):
        diagnostics.candidate_report(
            [EvaluationCase("q", "query", set(), "fixture")],
            {"q": {"text": []}},
            {"q": []},
            windows={"text": 1},
            candidates={"q": []},
        )


def test_color_adapter_reads_pixels_and_respects_filters(tmp_path):
    from PIL import Image

    red, blue = tmp_path / "red.png", tmp_path / "blue.png"
    Image.new("RGB", (50, 50), "red").save(red)
    Image.new("RGB", (50, 50), "blue").save(blue)
    catalog = units.UnitCatalog(
        [units.RetrievalUnit(1, 10, "image", metadata={"source": "a"}), units.RetrievalUnit(2, 20, "image")]
    )
    route = media.ColorHistogramRetriever(catalog, {1: red, 2: blue})
    assert route.retrieve(hybrid.SearchQuery("red object"), 10) == (1,)
    assert route.retrieve(hybrid.SearchQuery("蓝色"), 10) == (2,)
    assert route.retrieve(hybrid.SearchQuery("unrecognized object"), 10) == ()
    assert route.retrieve(hybrid.SearchQuery("red", hybrid.SearchFilter(allowed_ids=frozenset({2}))), 1) == ()
    assert route.retrieve(hybrid.SearchQuery("red"), 0) == ()


def test_generated_media_runs_real_routes_preserves_locations_and_repeats(tmp_path):
    first = multimodal.run(tmp_path / "first", groups=1)
    second = multimodal.run(tmp_path / "second", groups=1)
    assert first["ranking_sha256"] == second["ranking_sha256"]
    assert first["dataset"]["generated_files_sha256"] == second["dataset"]["generated_files_sha256"]
    assert first["dataset"]["queries"] == 4 and first["dataset"]["images"] == 18
    # Assert provenance and location integrity rather than a baked-in accuracy
    # target: these controls intentionally do not certify model generalization.
    assert "supplied synthetic OCR" in first["scope"]
    for name, report in first["comparisons"].items():
        for row in report["per_query"]:
            assert len(row["best_unit_ids"]) == len(row["locators"])
            if name != "flat-text":
                assert len(row["asset_ids"]) == len(set(row["asset_ids"]))
        assert report["final_fetch"]["max_per_call"] <= 10
    page = next(row for row in first["comparisons"]["multiroute"]["per_query"] if row["category"] == "page")
    assert page["asset_ids"] == [50] and page["locators"][0]["page"] == 3
    with pytest.raises(FileExistsError):
        multimodal.run(tmp_path / "first", groups=1)


def test_quality_workflow_keeps_selection_frozen_and_actual_rankings_equal(tmp_path):
    from backend.app.evaluation.hybrid_eval import dataset, workflow
    from backend.tests.test_hybrid_optimization import write_archive

    path = tmp_path / "fixture.zip"
    write_archive(path)
    train, test = dataset.load_dataset(path, split="train"), dataset.load_dataset(path)
    selection = workflow.select(train, test)
    checksum = selection["lock_sha256"]
    report = quality.run(selection, train, test)
    assert selection["lock_sha256"] == checksum == report["selection_lock_sha256"]
    assert report["selected_diagnostics"]["queries"] == 2
    for actual in report["actual_pipeline"].values():
        assert actual["final_fetch"]["calls"] == 2
        assert actual["final_fetch"]["max_per_call"] <= 10
        assert actual["candidate_feature_units"]["max_per_query"] <= 100


@pytest.mark.parametrize("module", ["hybrid_diagnostics", "multimodal_benchmark"])
def test_new_cli_help_without_site_packages(module):
    subprocess.run(
        [sys.executable, "-S", "-m", f"backend.app.evaluation.{module}", "--help"],
        check=True,
        capture_output=True,
        timeout=10,
    )
