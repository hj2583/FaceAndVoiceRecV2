import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np

from config import (
    AMBIGUITY_MARGIN,
    EMBEDDING_DIM,
    FACE_UPSCALE_TARGET_SIZE,
    FACE_UPSCALE_THRESHOLD,
    INDEX_PATH,
    RECOGNITION_THRESHOLD,
    UNKNOWN_MATCH_THRESHOLD,
)
from database import (
    add_embedding,
    create_person,
    create_unknown,
    get_conn,
    list_embeddings,
    list_unknown_samples_with_embeddings,
)
from face_backend import get_face_backend

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
    data = np.load(path, allow_pickle=True)

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


class FaceIndex:
    """
    Small/medium face database index implemented with NumPy.
    This deliberately replaces FAISS to avoid the Windows _swigfaiss DLL issue.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.ids = np.empty((0,), dtype=np.int64)
        self.embeddings = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        self.reload()

    def reload(self):
        with self.lock:
            if INDEX_PATH.exists():
                try:
                    data = np.load(INDEX_PATH, allow_pickle=True)
                    ids = np.asarray(data["ids"], dtype=np.int64)
                    embeddings = np.asarray(data["embeddings"], dtype=np.float32)

                    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
                        raise ValueError("Invalid embedding dimensions")

                    if len(ids) != len(embeddings):
                        raise ValueError("Index ids/embeddings length mismatch")

                    self.ids = ids
                    self.embeddings = embeddings
                    return
                except Exception:
                    logging.exception("Could not load NumPy face index; rebuilding.")

            self.rebuild()

    def rebuild(self):
        rows = list_embeddings()
        ids = []
        embeddings = []

        for embedding_id, _person_id, _name, embedding_path, _image_path, _quality in rows:
            path = Path(embedding_path)
            if not path.exists():
                continue

            try:
                emb = _load_embedding_file(path)
                emb = _normalize(emb)
                if emb is None or len(emb) != EMBEDDING_DIM:
                    continue

                ids.append(int(embedding_id))
                embeddings.append(emb)
            except Exception:
                logging.exception("Failed loading embedding: %s", path)

        with self.lock:
            self.ids = np.asarray(ids, dtype=np.int64)
            if embeddings:
                self.embeddings = np.vstack(embeddings).astype(np.float32)
            else:
                self.embeddings = np.empty((0, EMBEDDING_DIM), dtype=np.float32)

            self._save_locked()

    def _save_locked(self):
        temp = INDEX_PATH.with_suffix(".tmp.npz")
        np.savez_compressed(
            temp,
            ids=self.ids,
            embeddings=self.embeddings,
        )
        os.replace(temp, INDEX_PATH)

    def add(self, embedding_id, embedding):
        emb = _normalize(embedding)
        if emb is None:
            raise ValueError("Invalid embedding")

        with self.lock:
            self.ids = np.append(self.ids, int(embedding_id))
            if len(self.embeddings) == 0:
                self.embeddings = emb.reshape(1, -1)
            else:
                self.embeddings = np.vstack([self.embeddings, emb])
            self._save_locked()

    def search(self, query_embedding, threshold=RECOGNITION_THRESHOLD):
        query = _normalize(query_embedding)
        if query is None:
            return None

        with self.lock:
            if len(self.ids) == 0:
                return None

            similarities = self.embeddings @ query
            order = np.argsort(-similarities)

            best_idx = int(order[0])
            best_score = float(similarities[best_idx])

            embedding_ids = [int(self.ids[index]) for index in order]
            placeholders = ",".join("?" for _ in embedding_ids)
            with get_conn() as conn:
                person_rows = conn.execute(
                    "SELECT embedding_id, person_id "
                    "FROM face_embeddings "
                    f"WHERE embedding_id IN ({placeholders})",
                    embedding_ids,
                ).fetchall()

            person_by_embedding = {
                int(embedding_id): int(person_id)
                for embedding_id, person_id in person_rows
            }
            best_person_id = person_by_embedding.get(
                int(self.ids[best_idx]),
                int(self.ids[best_idx]),
            )

            second_score = -1.0
            for candidate_idx in order[1:]:
                candidate_id = int(self.ids[candidate_idx])
                candidate_person_id = person_by_embedding.get(
                    candidate_id,
                    candidate_id,
                )
                if candidate_person_id != best_person_id:
                    second_score = float(similarities[candidate_idx])
                    break

            if best_score < threshold:
                return None

            if second_score >= 0.0 and (best_score - second_score) < AMBIGUITY_MARGIN:
                return None

            return {
                "embedding_id": int(self.ids[best_idx]),
                "similarity": best_score,
                "second_similarity": second_score,
            }


def save_embedding_for_person(person_name, embedding, image_path=None, quality=0.0):
    """
    Adds an embedding to the shared database/index.
    Returns (person_id, embedding_id).
    """
    from config import KNOWN_FACES_DIR

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
    from database import get_conn
    with get_conn() as conn:
        conn.execute(
            "UPDATE face_embeddings SET embedding_path=? WHERE embedding_id=?",
            (str(final_path), embedding_id),
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

    scores = []

    for sample_id, unknown_id, embedding_path, quality in rows:

        path = Path(embedding_path)

        if not path.exists():
            continue

        try:
            stored_embedding = _load_embedding_file(path)
            stored_embedding = _normalize(stored_embedding)

            if stored_embedding is None:
                continue

            similarity = float(
                np.dot(query, stored_embedding)
            )

            scores.append(
                (
                    similarity,
                    int(unknown_id),
                    int(sample_id),
                )
            )

        except Exception:
            logging.exception(
                "Failed loading unknown embedding: %s",
                path,
            )

    if not scores:
        return None

    scores.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    best_similarity, best_unknown_id, best_sample_id = scores[0]

    second_similarity = (
        scores[1][0]
        if len(scores) > 1
        else -1.0
    )

    # Not similar enough
    if best_similarity < UNKNOWN_MATCH_THRESHOLD:
        return None

    # Too ambiguous
    if (
        len(scores) > 1
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
        "unknown_id": best_unknown_id,
        "similarity": best_similarity,
        "sample_id": best_sample_id,
    }

def register_unknown(
    label,
    embedding,
    image_path=None,
):
    from config import UNKNOWN_FACES_DIR

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