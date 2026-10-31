from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass, replace

from sqlalchemy import event, select
from sqlalchemy.orm import Session, joinedload

from backend.app.db.models import Chunk, File, Source


@dataclass(frozen=True)
class RetrievedSource:
    file_id: int
    chunk_id: int
    path: str
    windows_path: str
    page_or_segment: str
    text: str
    score: float


@dataclass
class DocumentIndex:
    sources: list[RetrievedSource]
    vectorizer: object
    matrix: object


_indexes: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_lock = threading.RLock()


@event.listens_for(Session, "after_flush")
def _dirty(session, *_args):
    session.info["document_index_dirty"] = True


@event.listens_for(Session, "do_orm_execute")
def _sql_dirty(state):
    if not state.is_select:
        state.session.info["document_index_dirty"] = True


@event.listens_for(Session, "after_commit")
@event.listens_for(Session, "after_rollback")
def _clean(session):
    session.info.pop("document_index_dirty", None)


def _build_index(db: Session) -> DocumentIndex:
    from sklearn.feature_extraction.text import TfidfVectorizer

    rows = db.execute(
        select(Chunk, File)
        .join(File, Chunk.file_id == File.id)
        .join(Source, Source.id == File.source_id)
        .options(joinedload(File.document))
        .where(File.ingestion_status == "indexed", Source.enabled.is_(True))
        .order_by(Chunk.id)
    ).all()
    sources = [
        RetrievedSource(
            file.id, chunk.id, file.canonical_path, file.windows_path, chunk.page_or_segment, chunk.text, 0.0
        )
        for chunk, file in rows
    ]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=60000)
    if not rows:
        return DocumentIndex(sources, vectorizer, None)
    corpus = [
        f"{file.document.title if file.document else ''}\n{file.windows_path}\n"
        f"{file.canonical_path}\n{chunk.page_or_segment}\n{chunk.text}"
        for chunk, file in rows
    ]
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        matrix = None
    return DocumentIndex(sources, vectorizer, matrix)


def _index(db: Session) -> DocumentIndex:
    if db.new or db.dirty or db.deleted:
        db.flush()
    revision = (
        db.connection().exec_driver_sql("SELECT revision FROM document_search_revision WHERE id = 1").scalar_one()
    )
    if db.info.get("document_index_dirty"):
        return _build_index(db)  # Never publish uncommitted snapshots into the cache.
    with _lock:
        cache = _indexes.setdefault(db.get_bind(), OrderedDict())
        if revision not in cache:
            cache[revision] = _build_index(db)
        cache.move_to_end(revision)
        while len(cache) > 2:
            cache.popitem(last=False)
        return cache[revision]


def search_knowledge(db: Session, query: str, limit: int = 8) -> list[RetrievedSource]:
    from sklearn.metrics.pairwise import cosine_similarity

    if not query.strip() or limit <= 0:
        return []
    index = _index(db)
    if index.matrix is None:
        return []
    scores = cosine_similarity(index.vectorizer.transform([query]), index.matrix).flatten()
    results = []
    for source, score in zip(index.sources, scores):
        boosted = float(score) + _exact_match_boost(query, source.text, source.path, source.windows_path)
        if boosted > 0:
            results.append(replace(source, score=boosted))
    return sorted(results, key=lambda item: (-item.score, item.chunk_id))[:limit]


def _exact_match_boost(query: str, content: str, path: str, windows_path: str) -> float:
    lowered_query = query.lower()
    haystack = f"{windows_path}\n{path}\n{content}".lower()
    if lowered_query and lowered_query in haystack:
        return 0.4
    terms = [term for term in lowered_query.replace("？", " ").replace("?", " ").split() if len(term) >= 2]
    return min(0.2, 0.05 * sum(1 for term in terms if term in haystack))
