from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.db.session import SessionLocal, init_db
from backend.app.ingestion.scanner import ensure_default_sources
from backend.app.workers.queue_controller import queue_controller


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    db = SessionLocal()
    try:
        ensure_default_sources(db)
    finally:
        db.close()
    queue_controller.start()
    try:
        yield
    finally:
        queue_controller.stop()


app = FastAPI(title="Lentica", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
