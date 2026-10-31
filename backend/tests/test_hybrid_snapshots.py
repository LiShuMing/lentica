"""Persistence, reader isolation, invalidation and failed activation checks."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys

import pytest

from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import cache_keys, querying, snapshots, stages, units


def sample_unit(identifier=1, parent=10, text="ERR_42"):
    return units.RetrievalUnit(
        identifier,
        parent,
        "page",
        units.Locator(page=3, bbox=(0.1, 0.1, 0.5, 0.3)),
        (units.TextEvidence("native", text, "native-parser", "v1"),),
        {"tenant": "a"},
    )


def build_sample(root):
    return snapshots.build(
        root,
        [sample_unit()],
        {10: {"title": "manual"}},
        snapshots.SnapshotIdentity("source-v1", "model-v1", "config-v1"),
        vectors={"space-v1": {1: [1.0, 0.5]}},
    )


def test_reopen_and_rollback_keep_reader_pinned_and_records_separate(tmp_path):
    first = build_sample(tmp_path)
    assert not (tmp_path / "CURRENT.json").exists()
    snapshots.activate(tmp_path, first.snapshot_id)
    old = snapshots.current(tmp_path)
    assert old.fetch_units([1])[1] == sample_unit()
    assert old.fetch([10]) == {10: {"title": "manual"}}
    assert old.vectors() == {"space-v1": {1: (1.0, 0.5)}}
    second = snapshots.derive(
        old,
        tmp_path,
        dataclasses.replace(old.identity, source_version="source-v2"),
        delete_units=frozenset({1}),
        upserts=[sample_unit(2, text="ERR_55")],
        record_upserts={10: {"title": "updated"}},
        vector_upserts={"space-v1": {2: [0.5, 1.0]}},
    )
    assert second.parent_id == first.snapshot_id
    snapshots.activate(tmp_path, second.snapshot_id)
    assert snapshots.current(tmp_path).fetch_units([2])[2].text() == "ERR_55"
    assert old.fetch_units([1])[1].text() == "ERR_42"
    assert old.fetch([10])[10]["title"] == "manual"
    snapshots.activate(tmp_path, old.snapshot_id)
    assert snapshots.current(tmp_path).snapshot_id == first.snapshot_id
    assert old.fetch_units([]) == old.fetch([]) == {}
    with pytest.raises(KeyError):
        second.fetch_units([1])


def test_unit_changes_and_model_versions_cannot_reuse_stale_vectors(tmp_path):
    original = build_sample(tmp_path)
    changed = snapshots.derive(original, tmp_path, original.identity, upserts=[sample_unit(text="changed")])
    assert changed.vectors() == {}
    new_model = snapshots.derive(original, tmp_path, dataclasses.replace(original.identity, model_version="new-model"))
    assert new_model.vectors() == {}
    deleted = snapshots.derive(original, tmp_path, original.identity, delete_assets=frozenset({10}))
    assert deleted.all_units() == () and deleted.vectors() == {}
    assert original.vectors() == {"space-v1": {1: (1.0, 0.5)}}
    with pytest.raises(ValueError, match="existing"):
        snapshots.derive(original, tmp_path, original.identity, delete_units=frozenset({999}))
    with pytest.raises(ValueError, match="same identity"):
        snapshots.derive(original, tmp_path, original.identity, delete_units=frozenset({1}), upserts=[sample_unit()])


def test_corrupt_snapshot_cannot_replace_current(tmp_path):
    first = build_sample(tmp_path)
    snapshots.activate(tmp_path, first.snapshot_id)
    before = (tmp_path / "CURRENT.json").read_bytes()
    second = snapshots.derive(first, tmp_path, first.identity)
    with second.path.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        snapshots.activate(tmp_path, second.snapshot_id)
    assert (tmp_path / "CURRENT.json").read_bytes() == before
    assert snapshots.current(tmp_path).snapshot_id == first.snapshot_id
    with pytest.raises(ValueError, match="identity"):
        snapshots.activate(tmp_path, "../../private")


def test_failed_pointer_replace_leaves_previous_reference(tmp_path, monkeypatch):
    first = build_sample(tmp_path)
    snapshots.activate(tmp_path, first.snapshot_id)
    before = (tmp_path / "CURRENT.json").read_bytes()
    second = snapshots.derive(first, tmp_path, first.identity)

    def fail(*args):
        raise OSError("injected replace failure")

    monkeypatch.setattr(snapshots.os, "replace", fail)
    with pytest.raises(OSError):
        snapshots.activate(tmp_path, second.snapshot_id)
    assert (tmp_path / "CURRENT.json").read_bytes() == before
    assert not list(tmp_path.glob(".CURRENT-*.tmp"))


def test_cache_keys_include_filter_and_all_version_boundaries(tmp_path):
    first = build_sample(tmp_path)
    query = hybrid.SearchQuery("ERR_42")
    config = hybrid.SearchConfig((hybrid.Branch("text"),))
    key = snapshots.cache_key(first, query, config)
    requests = [
        hybrid.SearchQuery("ERR_43"),
        hybrid.SearchQuery("ERR_42", hybrid.SearchFilter(equals={"tenant": "b"})),
        hybrid.SearchQuery("ERR_42", hybrid.SearchFilter(allowed_ids=frozenset())),
        hybrid.SearchQuery("ERR_42", hybrid.SearchFilter(allowed_ids=frozenset({1}))),
    ]
    assert all(snapshots.cache_key(first, item, config) != key for item in requests)
    assert snapshots.cache_key(first, query, dataclasses.replace(config, rrf_constant=1)) != key
    second = snapshots.derive(first, tmp_path, first.identity)
    assert snapshots.cache_key(second, query, config) != key
    manifest = json.loads((first.directory / "manifest.json").read_text())
    manifest["identity"]["model_version"] = "wrong-model"
    (first.directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="versions differ"):
        snapshots.Snapshot(first.directory)


@pytest.mark.parametrize("rows", [{1: []}, {1: [float("nan")]}, {99: [1.0]}, {1: [1], 2: [1, 2]}])
def test_invalid_vector_rows_fail_before_activation(tmp_path, rows):
    with pytest.raises(ValueError):
        snapshots.build(
            tmp_path, [sample_unit()], {10: "asset"}, snapshots.SnapshotIdentity("s", "m", "c"), vectors={"v": rows}
        )
    assert not (tmp_path / "CURRENT.json").exists()


def test_snapshot_can_reopen_without_site_packages(tmp_path):
    original = build_sample(tmp_path)
    code = """
import sys
from pathlib import Path
from backend.app.retrieval.hybrid import snapshots
value = snapshots.Snapshot(Path(sys.argv[1]))
assert value.fetch_units([1])[1].text() == 'ERR_42'
assert value.vectors()['space-v1'][1] == (1.0, 0.5)
"""
    subprocess.run([sys.executable, "-S", "-c", code, str(original.directory)], check=True, timeout=10)


def test_multimodal_cache_binds_pixels_family_capabilities_and_stage_budgets(tmp_path):
    from PIL import Image

    snapshot = build_sample(tmp_path)
    path = tmp_path / "query.png"
    Image.new("RGB", (8, 8), "red").save(path)
    query = querying.MultimodalQuery("red", path)
    config = stages.StageConfig(hybrid.SearchConfig((hybrid.Branch("image"),)))
    route = querying.RouteSpec(hybrid.Branch("image"), family="pixels")
    key = cache_keys.multimodal_key(snapshot, query, config, [route])
    assert key != cache_keys.multimodal_key(snapshot, query, dataclasses.replace(config, rerank_window=20), [route])
    assert key != cache_keys.multimodal_key(snapshot, query, config, [dataclasses.replace(route, family="other")])
    assert key != cache_keys.multimodal_key(snapshot, query, config, [dataclasses.replace(route, state="not_indexed")])
    Image.new("RGB", (8, 8), "blue").save(path)
    assert key != cache_keys.multimodal_key(snapshot, query, config, [route])
    with pytest.raises(ValueError, match="multimodal cache"):
        snapshots.cache_key(snapshot, query, config.search)
