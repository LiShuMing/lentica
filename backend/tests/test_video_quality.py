"""Reference metrics and real final-byte boundaries; fixtures are not model quality."""

import copy
import dataclasses
import hashlib
from pathlib import Path

import pytest

from backend.app.evaluation.hybrid_eval import video_metrics, video_quality, video_source, video_window_io
from backend.app.retrieval.hybrid import action_model, contracts, temporal, units, video_frames, video_windows


@pytest.fixture
def fixture_videos(tmp_path):
    videos, cases = [], []
    for identifier, part in enumerate(("dev", "validation", "test"), 1):
        data = f"explicit source fixture {identifier}".encode()
        name = f"video-{identifier}.mp4"
        (tmp_path / name).write_bytes(data)
        reference = video_source.SourceCase(
            f"{identifier:05d}",
            f"{identifier:04d}",
            part,
            12,
            (video_source.VideoEvent("c106", 0, 4), video_source.VideoEvent("c153", 6, 10)),
            len(data),
            0,
        )
        case = video_source.VideoCase(reference, identifier, name, hashlib.sha256(data).hexdigest())
        cases.append(case)
        metadata = {"source": "public-charades", "partition": part}
        frames = tuple(
            video_frames.FrameSample(
                units.RetrievalUnit(
                    identifier * 1000 + i,
                    identifier,
                    "video",
                    units.Locator(start_seconds=i / 5, end_seconds=i / 5 + 1 / 30),
                    metadata=metadata,
                ),
                tmp_path / f"unused-{identifier}-{i}.png",
                i * 6,
                "a" * 64,
            )
            for i in range(60)
        )
        sampled = video_frames.VideoFrames(tmp_path, case.sha256, "fixture-frames", 30, 360, frames)
        proposals = video_windows.build(sampled, first_unit_id=identifier * 1000 + 500)
        videos.append(video_window_io.PreparedVideo(case, sampled, proposals))
    selection = video_source.VideoSelection(
        Path("unused.json"), "fixture-selection", video_source.LABELS, tuple(c.reference for c in cases)
    )
    dataset = video_source.VideoDataset(tmp_path, "fixture-dataset", selection, tuple(cases))
    return dataset, tuple(videos)


def test_union_coverage_does_not_double_count_or_hide_unobservable_references(fixture_videos):
    _, videos = fixture_videos
    video = videos[0]
    reference = dataclasses.replace(
        video.source.reference,
        events=(video_source.VideoEvent("c106", 0, 1.2),) * 2 + (video_source.VideoEvent("c106", 20, 21),),
    )
    changed = dataclasses.replace(video, source=dataclasses.replace(video.source, reference=reference))
    ids = [w.unit.unit_id for w in changed.proposals.windows]
    row = video_metrics.single((changed,), "c106", ids)
    assert row["unit"]["relevant"] == []  # two identical1.2s spans never cover half of4s
    assert row["resource"]["relevant"] == [1]
    assert len(row["events"]) == 3 and row["events"][-1]["clipped_iou"] == 0
    assert video_metrics.iou(8, 12, video_source.VideoEvent("c106", 8, 16), 12) == 1
    assert video_metrics.iou(8, 12, video_source.VideoEvent("c106", 8, 16)) == 0.5
    missing = video_metrics.single((changed,), "c121", ids)
    assert missing["resource"]["no_answer_returned"] and missing["resource"]["recall_at_10"] == 0


def test_order_requires_each_step_localization_and_preserves_reverse_negatives(fixture_videos):
    _, videos = fixture_videos
    video = videos[0]
    a, b = video.proposals.windows[0].unit, video.proposals.windows[3].unit
    matches = temporal.sequence([("first", (a,)), ("second", (b,))], contracts.SearchFilter(), min_gap_seconds=0.5)
    row = video_metrics.ordered((video,), "c106", "c153", matches)
    assert row["localized_hit_at_5"] and row["iou_05_hit_at_5"]
    reverse = video_metrics.ordered((video,), "c153", "c106", matches)
    assert reverse["no_answer_returned"] and not reverse["localized_hit_at_5"]
    oracle = video_metrics.oracle((video,), "c106", "c153")
    assert oracle["raw_gold_asset_recall"] == 1 and oracle["reference_support_matches"] == 1
    # Right parent, wrong times: resource hit alone is not localization.
    wrong = video_metrics.ordered((video,), "c106", "c153", [dataclasses.replace(matches[0], evidence=(b, b))])
    assert wrong["resource_hit_at_5"] and not wrong["localized_hit_at_5"]


class FixtureModel:
    """Deterministic feature provider solely for composition/failure boundary checks."""

    version = "fixture-not-a-model"
    labels = tuple(video_source.LABELS.values())

    def score(self, frames, *, input_kind="original"):
        if input_kind == "original" and frames[0].unit.locator.start_seconds == 2:
            raise RuntimeError("fixture runtime failure")
        return action_model.ActionScores(self.labels, (1.0,) * 6, self.version, input_kind)


def test_all_queries_and_missing_windows_stay_in_denominator_and_final_reads_are_real(fixture_videos):
    source, videos = fixture_videos
    report = video_quality.evaluate(source, videos, route="videomae", action=FixtureModel())
    assert report["query_count"] == 108 and len(report["features"]["errors"]) == 3
    assert len(report["features"]["windows"]) == 12  # one of five windows fails per source
    assert len(report["features"]["controls"]) == 6
    for partition in report["partitions"].values():
        assert len(partition["single"]) == 6 and len(partition["sequence"]) == 30
        assert partition["summary"]["sequence_positive_queries"] == 1
        assert partition["summary"]["single_no_answer_queries"] == 4
        for query in partition["single"]:
            assert query["final_fetch"]["calls"] == 1
            assert query["final_fetch"]["bytes"] > 0
            assert len(query["unit"]["relevant"]) == (
                3 if query["code"] == "c153" else 2 if query["code"] == "c106" else 0
            )
    repeated = video_quality.evaluate(source, videos, route="videomae", action=FixtureModel())
    assert report["numeric_sha256"] == repeated["numeric_sha256"]
    assert report["ranking_sha256"] == repeated["ranking_sha256"]
    candidate = copy.deepcopy(report)
    baseline = copy.deepcopy(report)
    baseline["route"] = "siglip"
    assert not video_quality.compare(candidate, baseline)["accepted"]
    (source.directory / source.cases[0].filename).write_bytes(b"changed original")
    with pytest.raises(ValueError, match="source"):
        video_quality.evaluate(source, videos, route="videomae", action=FixtureModel())


def test_query_scores_reject_duplicate_rank_ids_and_foreign_sources(fixture_videos):
    _, videos = fixture_videos
    identifier = videos[0].proposals.windows[0].unit.unit_id
    with pytest.raises(ValueError, match="unique"):
        video_metrics.single(videos, "c106", [identifier, identifier])
    with pytest.raises(ValueError, match="partition"):
        video_metrics.single(videos, "c106", [99999])
