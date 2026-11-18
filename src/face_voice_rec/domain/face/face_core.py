import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np

from face_voice_rec.config import (
    AMBIGUITY_MARGIN,
    EMBEDDING_DIM,
    FACE_UPSCALE_TARGET_SIZE,
    FACE_UPSCALE_THRESHOLD,
    QDRANT_FACE_COLLECTION,
    QDRANT_FACE_COLLECTION,
    RECOGNITION_THRESHOLD,
    UNKNOWN_MATCH_THRESHOLD,
)
from face_voice_rec.storage.database import (
    add_embedding,
    create_person,
    create_unknown,
    get_conn,
    list_embeddings,
    list_unknown_samples_with_embeddings,
)
from face_voice_rec.storage.qdrant_store import (
    embedding_point_id,
    get_qdrant_vector_store,
    unknown_point_id,
)
from qdrant_client import models
from face_voice_rec.storage.safe_numpy import load_embedding_array
from face_voice_rec.domain.face.face_backend import get_face_backend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

_MODEL_LOCK = threading.RLock()
_MODEL_READY = False


def _normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        return None
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return (vector / norm).astype(np.float32)


def _load_embedding_file(path):
    """Load an embedding saved by older code paths or by UniFace object arrays."""
    data = load_embedding_array(path)

    if isinstance(data, np.ndarray) and data.dtype == object:
        data = np.asarray(data.tolist(), dtype=np.float32)

    return np.asarray(data, dtype=np.float32).reshape(-1)

def face_quality(face_crop):
    """
    Estimate face-image quality using:
    - face size
    - sharpness
    """

    if face_crop is None or face_crop.size == 0:
        return 0.0

    h, w = face_crop.shape[:2]

    gray = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2GRAY,
    )

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    ).var()

    # Normalize sharpness.
    sharpness_score = min(
        sharpness / 500.0,
        1.0,
    )

    # Face size score.
    size_score = min(
        (w * h) / (160 * 160),
        1.0,
    )

    quality = (
        0.6 * sharpness_score
        + 0.4 * size_score
    )

    return float(quality)

def build_face_model():
    """Explicitly load the active face recognition model once."""
    global _MODEL_READY
    with _MODEL_LOCK:
        if not _MODEL_READY:
            logging.info("Loading ArcFace model...")
            backend = get_face_backend()
            backend.build_model("ArcFace")
            _MODEL_READY = True
            logging.info("ArcFace model loaded.")


def preprocess_face_for_recognition(
    face_crop,
    target_size=FACE_UPSCALE_TARGET_SIZE,
    upscale_threshold=FACE_UPSCALE_THRESHOLD,
):
    """Upscale small face crops while preserving their aspect ratio."""
    if face_crop is None or face_crop.size == 0:
        return face_crop

    height, width = face_crop.shape[:2]
    if max(height, width) >= upscale_threshold:
        return face_crop

    scale = target_size / max(height, width)
    return cv2.resize(
        face_crop,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_CUBIC,
    )


def extract_embedding(face_crop):
    if face_crop is None or face_crop.size == 0:
        return None

    try:
        build_face_model()
        face_crop = preprocess_face_for_recognition(face_crop)
        backend = get_face_backend()
        embedding = backend.extract_embedding(face_crop)
        if embedding is None:
            return None

        if embedding.shape[0] != EMBEDDING_DIM:
            logging.error(
                "Unexpected ArcFace embedding size: %s (expected %s)",
                embedding.shape[0],
                EMBEDDING_DIM,
            )
            return None

        return embedding

    except Exception:
        logging.exception("ArcFace embedding extraction failed")
        return None


def load_or_extract_embedding(embedding_path, image_path):
    """Load a valid sample embedding, rebuilding it from the image if needed."""
    try:
        embedding = _load_embedding_file(Path(embedding_path))
        normalized = _normalize(embedding)
        if normalized is not None and len(normalized) == EMBEDDING_DIM:
            return normalized
    except Exception:
        logging.warning("Invalid sample embedding: %s", embedding_path)

    image = cv2.imread(str(image_path))
    if image is None:
        return None

    rebuilt = extract_embedding(image)
    if rebuilt is None:
        return None

    np.save(embedding_path, rebuilt)
    return rebuilt


class FaceIndex:
    """Qdrant-backed face search with the existing ambiguity policy."""

    def __init__(self, vector_store=None):
        self._vector_store = vector_store or get_qdrant_vector_store()

    def reload(self):
        """Compatibility no-op; Qdrant is queried directly for every search."""

    def rebuild(self):
        """Compatibility no-op; use the validated backfill for repair/sync."""

    def search(self, query_embedding, threshold=RECOGNITION_THRESHOLD):
        query = _normalize(query_embedding)
        if query is None:
            return None
        groups = self._vector_store.search_grouped(
            QDRANT_FACE_COLLECTION,
            query,
            group_by="person_id",
            limit=2,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="record_type",
                        match=models.MatchValue(value="enrolled_embedding"),
                    ),
                    models.FieldCondition(
                        key="modality",
                        match=models.MatchValue(value="face"),
                    ),
                ]
            ),
        )
        if not groups or not groups[0].hits:
            return None

        best = groups[0].hits[0]
        best_score = float(best.score)
        second_score = (
            float(groups[1].hits[0].score)
            if len(groups) > 1 and groups[1].hits
            else -1.0
        )
        if best_score < threshold:
            return None
        if second_score >= 0.0 and best_score - second_score < AMBIGUITY_MARGIN:
            return None
        return {
            "embedding_id": int(best.payload["embedding_id"]),
            "similarity": best_score,
            "second_similarity": second_score,
        }


def save_embedding_for_person(person_name, embedding, image_path=None, quality=0.0):
    """
    Adds an embedding to the shared database/index.
    Returns (person_id, embedding_id).
    """
    from face_voice_rec.config import KNOWN_FACES_DIR

    normalized_embedding = _normalize(embedding)
    if normalized_embedding is None or len(normalized_embedding) != EMBEDDING_DIM:
        raise ValueError("Invalid face embedding")

    person_id = create_person(person_name)

    embedding_dir = KNOWN_FACES_DIR / str(person_id)
    embedding_dir.mkdir(parents=True, exist_ok=True)

    embedding_id_placeholder = "new"
    path = embedding_dir / f"{embedding_id_placeholder}.npy"
    np.save(path, normalized_embedding)

    embedding_id = add_embedding(
        person_id=person_id,
        embedding_path=path,
        image_path=image_path,
        quality=quality,
    )

    final_path = embedding_dir / f"{embedding_id}.npy"
    os.replace(path, final_path)

    # Repair DB path after the embedding ID is known.
    from face_voice_rec.storage.database import get_conn
    with get_conn() as conn:
        conn.execute(
            "UPDATE face_embeddings SET embedding_path=? WHERE embedding_id=?",
            (str(final_path), embedding_id),
        )

    get_qdrant_vector_store().upsert(
        QDRANT_FACE_COLLECTION,
        embedding_point_id("face", embedding_id),
        normalized_embedding,
        {
            "record_type": "enrolled_embedding",
            "modality": "face",
            "embedding_id": int(embedding_id),
            "person_id": int(person_id),
            "quality": float(quality or 0.0),
        },
    )

    return person_id, embedding_id

def find_matching_unknown(query_embedding):
    """
    Find an existing unresolved unknown identity that matches
    the supplied face embedding.

    Returns:
        {
            "unknown_id": int,
            "similarity": float,
            "sample_id": int,
        }

    or None.
    """

    query = _normalize(query_embedding)

    if query is None:
        return None

    rows = list_unknown_samples_with_embeddings()

    if not rows:
        return None

    points = get_qdrant_vector_store().search(
        QDRANT_FACE_COLLECTION,
        query,
        limit=2,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="record_type",
                    match=models.MatchValue(value="unknown_sample"),
                ),
                models.FieldCondition(
                    key="modality",
                    match=models.MatchValue(value="face"),
                ),
                models.FieldCondition(
                    key="unknown_id",
                    match=models.MatchAny(any=sorted({int(row[1]) for row in rows})),
                ),
            ]
        ),
    )

    if not points:
        return None

    best = points[0]
    best_similarity = float(best.score)
    second_similarity = float(points[1].score) if len(points) > 1 else -1.0

    # Not similar enough
    if best_similarity < UNKNOWN_MATCH_THRESHOLD:
        return None

    # Too ambiguous
    if (
        len(points) > 1
        and (best_similarity - second_similarity) < 0.03
    ):
        logging.info(
            "Unknown match rejected due to ambiguity: "
            "best=%.3f second=%.3f",
            best_similarity,
            second_similarity,
        )
        return None

    return {
        "unknown_id": int(best.payload["unknown_id"]),
        "similarity": best_similarity,
        "sample_id": int(best.payload["sample_id"]),
    }


def index_unknown_face_sample(sample_id, unknown_id, quality, embedding):
    """Store a captured unresolved face embedding in Qdrant."""
    vector = _normalize(embedding)
    if vector is None or vector.shape[0] != EMBEDDING_DIM:
        raise ValueError("Invalid face embedding")
    get_qdrant_vector_store().upsert(
        QDRANT_FACE_COLLECTION,
        unknown_point_id("face", int(sample_id)),
        vector,
        {
            "record_type": "unknown_sample",
            "modality": "face",
            "sample_id": int(sample_id),
            "unknown_id": int(unknown_id),
            "quality": float(quality or 0.0),
        },
    )

def register_unknown(
    label,
    embedding,
    image_path=None,
):
    from face_voice_rec.config import UNKNOWN_FACES_DIR

    UNKNOWN_FACES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    unknown_id_hint = (
        label
        .replace("/", "_")
        .replace("\\", "_")
    )

    embedding_path = (
        UNKNOWN_FACES_DIR
        / f"{unknown_id_hint}.npy"
    )

    np.save(
        embedding_path,
        _normalize(embedding),
    )

    return create_unknown(
        label=label,
        image_path=image_path,
        embedding_path=embedding_path,
    )


def enroll_unknown_face_track(unknown_id, person_id):
    """Enroll the saved samples for one unknown face track to an existing person."""
    from face_voice_rec.storage.database import (
        list_embeddings,
        list_persons,
        list_unknown_samples,
        list_unknowns,
        resolve_unknown,
    )

    unknown_id = int(unknown_id)
    person_id = int(person_id)
    unknown = next(
        (
            row
            for row in list_unknowns(include_resolved=True)
            if int(row[0]) == unknown_id
        ),
        None,
    )
    if unknown is None:
        raise ValueError(f"Unknown face track {unknown_id} does not exist")
    resolved_person_id = unknown[5]
    if resolved_person_id is not None and int(resolved_person_id) != person_id:
        raise ValueError(
            f"Unknown face track {unknown_id} is already linked to another person"
        )

    people = {int(row[0]): row[1] for row in list_persons()}
    person_name = people.get(person_id)
    if person_name is None:
        raise ValueError(f"Person {person_id} does not exist")

    existing_image_paths = {
        str(Path(row[4]).resolve()).casefold()
        for row in list_embeddings()
        if int(row[1]) == person_id and row[4]
    }
    samples = list_unknown_samples(unknown_id)
    if not samples:
        samples = [
            (
                None,
                unknown[2],
                unknown[3],
                0.0,
                0.0,
                None,
            )
        ]

    embeddings_added = 0
    for sample_id, image_path, embedding_path, quality, _angle_score, _created in samples:
        if (
            image_path
            and str(Path(image_path).resolve()).casefold() in existing_image_paths
        ):
            continue
        embedding = load_or_extract_embedding(embedding_path, image_path)
        if embedding is None:
            logging.warning(
                "Could not enroll unknown face sample %s for person %s",
                sample_id if sample_id is not None else unknown_id,
                person_id,
            )
            continue
        save_embedding_for_person(
            person_name,
            embedding,
            image_path=image_path,
            quality=quality,
        )
        embeddings_added += 1
        if image_path:
            existing_image_paths.add(str(Path(image_path).resolve()).casefold())

    if embeddings_added == 0 and resolved_person_id is None:
        raise ValueError(
            f"Unknown face track {unknown_id} has no valid samples to enroll"
        )
    resolve_unknown(unknown_id, person_id)
    return embeddings_added

