"""Real bundle versions never mix feature, text and vector generations."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys

import pytest

from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import ann, bundles, documents, fulltext, snapshots, units


def build_version(root, revision, *, wrong_model=False):
    pytest.importorskip("tantivy")
    faiss = pytest.importorskip("faiss")
    faiss.omp_set_num_threads(1)
    unit = units.RetrievalUnit(
        revision,
        10,
        "text",
        evidence=(units.TextEvidence("native", f"ERR_{revision}", "fixture", "v1"),),
        metadata={"tenant": "a"},
    )
    identity = snapshots.SnapshotIdentity(f"source-{revision}", "model-1", "config-1")
    snapshot = snapshots.build(root, [unit], {10: {"revision": revision}}, identity)
    text = fulltext.TantivyRetriever.build(
        root / f"text-{revision}",
        [documents.CorpusDocument(unit.unit_id, unit.text(), metadata=unit.metadata)],
        source_version=identity.source_version,
    )
    ann.HnswRetriever.build(
        root / f"vector-{revision}",
        [unit.unit_id],
        [[1, 0]],
        lambda text: [1, 0],
        encoder_id="wrong" if wrong_model else identity.model_version,
        source_version=identity.source_version,
        metadata={unit.unit_id: unit.metadata},
    )
    return snapshot, text.directory, root / f"vector-{revision}"


def test_bundle_switch_rollback_pinned_reader_and_complete_file_integrity(tmp_path):
    snapshot, text, vector = build_version(tmp_path, 1)
    first = bundles.build(tmp_path, snapshot, text_directory=text, vector_directory=vector)
    bundles.activate(tmp_path, first.bundle_id)
    old = bundles.current(tmp_path)
    snapshot, text, vector = build_version(tmp_path, 2)
    second = bundles.build(tmp_path, snapshot, text_directory=text, vector_directory=vector)
    bundles.activate(tmp_path, second.bundle_id)
    assert bundles.current(tmp_path).snapshot.fetch([10]) == {10: {"revision": 2}}
    assert old.snapshot.fetch([10]) == {10: {"revision": 1}}
    assert old.open_text().retrieve(hybrid.SearchQuery("ERR_1"), 10) == (1,)
    assert old.open_vector(lambda text: [1, 0], encoder_id="model-1").retrieve(hybrid.SearchQuery("query"), 10) == (1,)
    assert bundles.cache_namespace(old) != bundles.cache_namespace(second)
    bundles.activate(tmp_path, old.bundle_id)
    before = (tmp_path / "BUNDLE_CURRENT.json").read_bytes()
    with (second.directory / "vector" / "index.faiss").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        bundles.activate(tmp_path, second.bundle_id)
    assert (tmp_path / "BUNDLE_CURRENT.json").read_bytes() == before
    assert bundles.current(tmp_path).bundle_id == old.bundle_id
    # The immutable wrapper itself is stdlib-only, even when it contains native indexes.
    code = "from pathlib import Path; from backend.app.retrieval.hybrid import bundles; import sys; value = bundles.IndexBundle(Path(sys.argv[1])); assert value.snapshot.fetch([10])[10]['revision'] == 1; assert 'numpy' not in sys.modules"
    subprocess.run([sys.executable, "-S", "-c", code, str(old.directory)], check=True, timeout=10)


def test_bundle_rejects_mixed_source_or_model_before_publication(tmp_path):
    snapshot, text, vector = build_version(tmp_path, 1, wrong_model=True)
    with pytest.raises(ValueError, match="encoder"):
        bundles.build(tmp_path, snapshot, text_directory=text, vector_directory=vector)
    changed = snapshots.derive(snapshot, tmp_path, dataclasses.replace(snapshot.identity, source_version="new-source"))
    with pytest.raises(ValueError, match="source"):
        bundles.build(tmp_path, changed, text_directory=text)
    assert not (tmp_path / "BUNDLE_CURRENT.json").exists()


def test_failed_bundle_replace_preserves_pointer_and_cleans_temporary(tmp_path, monkeypatch):
    snapshot = snapshots.build(tmp_path, [], {}, snapshots.SnapshotIdentity("s", "m", "c"))
    first = bundles.build(tmp_path, snapshot)
    bundles.activate(tmp_path, first.bundle_id)
    before = (tmp_path / "BUNDLE_CURRENT.json").read_bytes()

    def fail(*args):
        raise OSError("injected atomic replacement failure")

    monkeypatch.setattr(bundles.os, "replace", fail)
    with pytest.raises(OSError):
        bundles.activate(tmp_path, first.bundle_id)
    assert (tmp_path / "BUNDLE_CURRENT.json").read_bytes() == before
    assert not list(tmp_path.glob(".BUNDLE_CURRENT-*.tmp"))
    with pytest.raises(ValueError, match="identity"):
        bundles.activate(tmp_path, "../../other")
    manifest = json.loads((first.directory / "bundle.json").read_text())
    manifest["files"]["../../outside"] = "0" * 64
    (first.directory / "bundle.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="path"):
        bundles.IndexBundle(first.directory)
