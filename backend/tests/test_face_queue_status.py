from __future__ import annotations

import pytest

from backend.app.db.models import File, Image, Source
from backend.app.workers.image_worker import _first_unindexed_face, enqueue_next_face_batch, enqueue_photo_index


def add_photo(db, status):
    source = Source(name="Fixture", root_path="/fixture", kind="images")
    db.add(source)
    db.flush()
    photo = File(source_id=source.id, canonical_path="/fixture/test.png", windows_path="/fixture/test.png",
                 file_type="png", ingestion_status="indexed")
    db.add(photo)
    db.flush()
    db.add(Image(file_id=photo.id, face_analysis_status=status))
    db.commit()
    return photo


@pytest.mark.parametrize("status", ["completed", "completed_no_face", "unavailable"])
def test_terminal_face_statuses_do_not_enqueue(db, status):
    add_photo(db, status)
    assert _first_unindexed_face(db) is None
    assert enqueue_next_face_batch(db, {}) is None


@pytest.mark.parametrize("status", ["pending", "failed"])
def test_face_backfill_is_prioritized_and_not_duplicated(db, status):
    photo = add_photo(db, status)
    assert _first_unindexed_face(db) == photo.id
    photo_task = enqueue_photo_index(db, {})
    face_task = enqueue_next_face_batch(db, {})
    assert face_task.priority < photo_task.priority
    assert enqueue_next_face_batch(db, {}) is None
