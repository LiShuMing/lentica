"""Prepared public-image protocol uses complete archives, bounded paths and groups."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile

import pytest

from backend.app.evaluation.hybrid_eval import mirror_images


def archive_fixture(root, *, bad_path=False, duplicate=False, train_count=2, test_count=1):
    from PIL import Image

    path = root / "images.tgz"
    with tarfile.open(path, "w:gz") as archive:
        for category, label in enumerate(mirror_images.CLASSES):
            for index in range(train_count + test_count):
                source = "train" if index < train_count else "test"
                sample = 0 if duplicate and category == 0 and index == 2 else index
                image = Image.new("RGB", (32, 32), (category * 20, sample * 40, 10))
                stream = io.BytesIO()
                image.save(stream, format="PNG")
                encoded = stream.getvalue()
                name = f"cifar10/{source}/{label}/{index}.png"
                if bad_path and category == index == 0:
                    name = "../outside.png"
                member = tarfile.TarInfo(name)
                member.size = len(encoded)
                archive.addfile(member, io.BytesIO(encoded))
    record = {
        "source": "synthetic archive protocol fixture",
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (root / "acquisition.json").write_text(json.dumps(record))
    return path


def test_image_source_partitions_exclude_identical_pixels_and_reopen_without_mutating_source(tmp_path):
    archive = archive_fixture(tmp_path, duplicate=True)
    before = archive.read_bytes()
    result = mirror_images.prepare(archive, tmp_path / "selected", per_class=1)
    assert len(result.partition("dev")) == len(result.partition("validation")) == 10
    assert len(result.partition("test")) == 9
    assert len({item.pixel_sha256 for item in result.items}) == 29
    assert all(item.filename[:5].isdigit() for item in result.items)
    assert mirror_images.load(result.directory).identity == result.identity
    assert archive.read_bytes() == before
    image = result.directory / result.items[0].filename
    image.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        mirror_images.load(result.directory)


def test_public_archive_path_traversal_is_rejected_without_extracting(tmp_path):
    archive = archive_fixture(tmp_path, bad_path=True)
    with pytest.raises(ValueError, match="path"):
        mirror_images.prepare(archive, tmp_path / "selected", per_class=1)
    assert not (tmp_path / "selected").exists()
    assert not (tmp_path / "outside.png").exists()


def test_next_trial_excludes_all_prior_original_sources_and_exact_pixels(tmp_path):
    archive = archive_fixture(tmp_path, train_count=4, test_count=2)
    first = mirror_images.prepare(archive, tmp_path / "first", per_class=1)
    second = mirror_images.prepare(archive, tmp_path / "second", per_class=1, exclude=[first])
    assert not {item.source_id for item in first.items} & {item.source_id for item in second.items}
    assert not {item.pixel_sha256 for item in first.items} & {item.pixel_sha256 for item in second.items}
    assert [len(second.partition(name)) for name in ("dev", "validation", "test")] == [10, 10, 10]
