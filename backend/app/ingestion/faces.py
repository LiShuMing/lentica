from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from backend.app.db.models import FaceDetection, File, Image, PersonAlias, PersonCluster
from backend.app.models.local_face import FaceRuntimeUnavailable, InsightFaceDetector


@dataclass
class FaceIndexStats:
    photos: int = 0
    faces: int = 0
    clusters_created: int = 0
    unavailable: bool = False
    error: str | None = None


class FaceIndexer:
    def __init__(self, detector: InsightFaceDetector | None = None, threshold: float = 0.58) -> None:
        self._detector = detector
        self._detector_failed = False
        self.threshold = threshold

    def index_photo(self, db: Session, file: File, image_path: Path, force: bool = False) -> FaceIndexStats:
        stats = FaceIndexStats(photos=1)
        image = db.scalar(select(Image).where(Image.file_id == file.id))
        try:
            detector = self._detector_or_raise()
        except FaceRuntimeUnavailable as exc:
            if image:
                image.face_analysis_status = "unavailable"
            stats.unavailable = True
            stats.error = str(exc)
            return stats

        if not force and db.scalar(select(FaceDetection.id).where(FaceDetection.file_id == file.id)):
            if image:
                image.face_analysis_status = "completed"
                image.face_indexed_at = datetime.utcnow()
            return stats

        db.execute(delete(FaceDetection).where(FaceDetection.file_id == file.id))
        try:
            observations = detector.detect(image_path)
        except Exception as exc:  # noqa: BLE001
            if image:
                image.face_analysis_status = "failed"
            stats.error = repr(exc)
            return stats

        for observation in observations:
            embedding = _normalize(observation.embedding)
            cluster, created = _assign_cluster(db, embedding, file.id, self.threshold)
            db.add(
                FaceDetection(
                    file_id=file.id,
                    cluster_id=cluster.id,
                    bbox_json=json.dumps(observation.bbox),
                    embedding_json=json.dumps(embedding),
                    detection_score=f"{observation.score:.6f}",
                    detector="insightface",
                )
            )
            stats.faces += 1
            if created:
                stats.clusters_created += 1
        if image:
            image.face_analysis_status = "completed" if observations else "completed_no_face"
            image.face_indexed_at = datetime.utcnow()
        return stats

    def _detector_or_raise(self) -> InsightFaceDetector:
        if self._detector_failed:
            raise FaceRuntimeUnavailable("Face detector unavailable in this Python environment.")
        if self._detector is None:
            try:
                self._detector = InsightFaceDetector()
            except FaceRuntimeUnavailable:
                self._detector_failed = True
                raise
        return self._detector


def people_for_file(db: Session, file_id: int) -> list[dict[str, object]]:
    rows = db.execute(
        select(PersonCluster, func.count(FaceDetection.id))
        .join(FaceDetection, FaceDetection.cluster_id == PersonCluster.id)
        .where(FaceDetection.file_id == file_id, PersonCluster.ignored.is_(False))
        .group_by(PersonCluster.id)
    ).all()
    aliases_by_cluster = _aliases_by_cluster(db, [cluster.id for cluster, _count in rows])
    return [
        {
            "cluster_id": cluster.id,
            "label": cluster.label,
            "name": cluster.display_name or cluster.label,
            "aliases": aliases_by_cluster.get(cluster.id, []),
            "face_count": int(count),
        }
        for cluster, count in rows
    ]


def matching_person_file_ids(db: Session, query: str) -> tuple[set[int], list[str]]:
    query_lower = query.lower()
    clusters = db.scalars(select(PersonCluster).where(PersonCluster.ignored.is_(False))).all()
    aliases_by_cluster = _aliases_by_cluster(db, [cluster.id for cluster in clusters])
    identity_clusters: dict[str, list[int]] = {}
    matched_names: set[str] = set()
    for cluster in clusters:
        terms = [cluster.label]
        if cluster.display_name:
            terms.append(cluster.display_name)
        terms.extend(aliases_by_cluster.get(cluster.id, []))
        matched_terms = [term for term in terms if _term_matches_query(term, query_lower)]
        if not matched_terms:
            continue
        # An explicitly requested cluster stays distinct. Shared user names/aliases
        # identify one logical person even when face clustering split them.
        identity = _identity_key(cluster.id, cluster.label, matched_terms)
        identity_clusters.setdefault(identity, []).append(cluster.id)
        matched_names.update(matched_terms)
    if not identity_clusters:
        return set(), []
    files_by_identity = [
        {
            int(file_id)
            for file_id in db.scalars(select(FaceDetection.file_id).where(FaceDetection.cluster_id.in_(cluster_ids))).all()
        }
        for cluster_ids in identity_clusters.values()
    ]
    if len(files_by_identity) == 1:
        return files_by_identity[0], sorted(matched_names)
    co_occurrence = set.intersection(*files_by_identity) if files_by_identity else set()
    any_match = set.union(*files_by_identity) if files_by_identity else set()
    return (co_occurrence or any_match), sorted(matched_names)


def refresh_people_context_for_cluster(db: Session, cluster_id: int) -> int:
    file_ids = db.scalars(select(FaceDetection.file_id).where(FaceDetection.cluster_id == cluster_id)).all()
    refreshed = 0
    for file_id in set(int(item) for item in file_ids):
        if refresh_people_context_for_file(db, file_id):
            refreshed += 1
    return refreshed


def refresh_people_context_for_file(db: Session, file_id: int) -> bool:
    image = db.scalar(select(Image).where(Image.file_id == file_id))
    if not image:
        return False
    context = _parse_context(image.context_json)
    detected_people = people_for_file(db, file_id)
    context["person_clusters"] = detected_people
    removable_terms = _all_person_terms(db)
    people_terms = [term for term in _as_text_list(context.get("people")) if term.lower() not in removable_terms]
    for person in detected_people:
        for term in [person.get("name"), person.get("label"), *list(person.get("aliases") or [])]:
            text = str(term or "").strip()
            if text and text not in people_terms:
                people_terms.append(text)
    context["people"] = people_terms
    image.context_json = json.dumps(context, ensure_ascii=False)
    return True


def _assign_cluster(
    db: Session, embedding: list[float], cover_file_id: int, threshold: float
) -> tuple[PersonCluster, bool]:
    clusters = db.scalars(select(PersonCluster).where(PersonCluster.ignored.is_(False))).all()
    best_cluster: PersonCluster | None = None
    best_score = -1.0
    for cluster in clusters:
        if not cluster.centroid_embedding_json:
            continue
        score = _cosine(embedding, json.loads(cluster.centroid_embedding_json))
        if score > best_score:
            best_score = score
            best_cluster = cluster

    if best_cluster is None or best_score < threshold:
        cluster = PersonCluster(
            label="pending",
            centroid_embedding_json=json.dumps(embedding),
            face_count=1,
            cover_file_id=cover_file_id,
            confidence=f"{max(best_score, 0.0):.6f}",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(cluster)
        db.flush()
        cluster.label = f"person_{cluster.id:03d}"
        return cluster, True

    old_count = max(0, int(best_cluster.face_count or 0))
    old_centroid = json.loads(best_cluster.centroid_embedding_json)
    centroid = _normalize([(old_centroid[idx] * old_count) + embedding[idx] for idx in range(len(embedding))])
    best_cluster.centroid_embedding_json = json.dumps(centroid)
    best_cluster.face_count = old_count + 1
    best_cluster.confidence = f"{best_score:.6f}"
    if not best_cluster.cover_file_id:
        best_cluster.cover_file_id = cover_file_id
    best_cluster.updated_at = datetime.utcnow()
    return best_cluster, False


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [float(value / norm) for value in vector]


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    return float(sum(left * right for left, right in zip(a, b)))


def _aliases_by_cluster(db: Session, cluster_ids: list[int]) -> dict[int, list[str]]:
    if not cluster_ids:
        return {}
    rows = db.execute(select(PersonAlias.cluster_id, PersonAlias.alias).where(PersonAlias.cluster_id.in_(cluster_ids))).all()
    result: dict[int, list[str]] = {}
    for cluster_id, alias in rows:
        text = str(alias or "").strip()
        if text:
            result.setdefault(int(cluster_id), []).append(text)
    return result


def _term_matches_query(term: str, query_lower: str) -> bool:
    text = term.strip().lower()
    if not text:
        return False
    return text in query_lower


def _identity_key(cluster_id: int, label: str, matched_terms: list[str]) -> str:
    if label.lower() in {term.lower() for term in matched_terms}:
        return f"cluster:{cluster_id}"
    return f"person:{matched_terms[0].lower()}"


def _all_person_terms(db: Session) -> set[str]:
    clusters = db.scalars(select(PersonCluster)).all()
    terms: set[str] = set()
    for cluster in clusters:
        for value in [cluster.label, cluster.display_name]:
            text = str(value or "").strip().lower()
            if text:
                terms.add(text)
    aliases = db.scalars(select(PersonAlias.alias)).all()
    for alias in aliases:
        text = str(alias or "").strip().lower()
        if text:
            terms.add(text)
    return terms


def _parse_context(value: str | None) -> dict[str, object]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _as_text_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value:
        return [str(value).strip()]
    return []
