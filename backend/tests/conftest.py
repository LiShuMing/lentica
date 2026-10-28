from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool


_test_root = tempfile.TemporaryDirectory(prefix="lentica-tests-")
os.environ["LENTICA_DATA_DIR"] = str(Path(_test_root.name) / "data")
os.environ["LENTICA_DOCUMENTS_SOURCE"] = str(Path(_test_root.name) / "documents")
os.environ["LENTICA_PHOTOS_SOURCE"] = str(Path(_test_root.name) / "photos")
os.environ["LENTICA_QUEUE_ENABLED"] = "false"
os.environ["LENTICA_ENABLE_FACES"] = "false"
os.environ["LENTICA_WORKER_MODE"] = "smoke"


@pytest.fixture
def db():
    from backend.app.db import models  # noqa: F401
    from backend.app.db.session import Base

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()
