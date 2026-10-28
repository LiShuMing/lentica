from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import Chunk, File


@dataclass(frozen=True)
class RetrievedSource:
    file_id: int
    chunk_id: int
    path: str
    windows_path: str
    page_or_segment: str
    text: str
    score: float


def search_knowledge(db: Session, query: str, limit: int = 8) -> list[RetrievedSource]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    rows = db.execute(
        select(Chunk, File).join(File, Chunk.file_id == File.id).where(File.ingestion_status == "indexed")
    ).all()
    if not rows:
        return []

    corpus = [_retrieval_text(chunk, file) for chunk, file in rows]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=60000)
    matrix = vectorizer.fit_transform(corpus + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:limit]

    results: list[RetrievedSource] = []
    for index, score in ranked:
        chunk, file = rows[index]
        boosted_score = float(score) + _exact_match_boost(query, chunk.text, file)
        if boosted_score <= 0:
            continue
        results.append(
            RetrievedSource(
                file_id=file.id,
                chunk_id=chunk.id,
                path=file.canonical_path,
                windows_path=file.windows_path,
                page_or_segment=chunk.page_or_segment,
                text=chunk.text,
                score=boosted_score,
            )
        )
    return results


def _retrieval_text(chunk: Chunk, file: File) -> str:
    title = file.document.title if file.document else ""
    return f"{title}\n{file.windows_path}\n{file.canonical_path}\n{chunk.page_or_segment}\n{chunk.text}"


def _exact_match_boost(query: str, text: str, file: File) -> float:
    lowered_query = query.lower()
    haystack = f"{file.windows_path}\n{file.canonical_path}\n{text}".lower()
    if lowered_query and lowered_query in haystack:
        return 0.4
    terms = [term for term in lowered_query.replace("？", " ").replace("?", " ").split() if len(term) >= 2]
    return min(0.2, 0.05 * sum(1 for term in terms if term in haystack))
