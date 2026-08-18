import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace

from config import (
    AMBIGUITY_MARGIN,
    EMBEDDING_DIM,
    INDEX_PATH,
    RECOGNITION_THRESHOLD,
    UNKNOWN_MATCH_THRESHOLD,
)
from database import (
    add_embedding,
    create_person,
    create_unknown,
    list_embeddings,
    list_unknown_samples_with_embeddings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

_MODEL_LOCK = threading.RLock()
_MODEL_READY = False


def _normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm <= 1e-8:
        return None
    return (vector / norm).astype(np.float32)

def face_quality(face_crop):
    """
    Estimate face-image quality using:
    - face size
    - sharpness
    """

    if face_crop is None or face_crop.size == 0:
        return 0.0

    h, w = face_crop.shape[:2]

    if h < 50 or w < 50:
        return 0.0

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
    """
    Explicitly load ArcFace once. DeepFace caches the model internally too.
    """
    global _MODEL_READY
    with _MODEL_LOCK:
        if not _MODEL_READY:
            logging.info("Loading ArcFace model...")
            DeepFace.build_model("ArcFace")
            _MODEL_READY = True
            logging.info("ArcFace model loaded.")


def extract_embedding(face_crop):
    if face_crop is None or face_crop.size == 0:
        return None

    try:
        build_face_model()

        result = DeepFace.represent(
            img_path=face_crop,
            model_name="ArcFace",
            detector_backend="skip",
            enforce_detection=False,
            align=True,
        )

        if not result:
            return None

        embedding = _normalize(result[0]["embedding"])

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
                    data = np.load(INDEX_PATH, allow_pickle=False)
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
                emb = np.load(path).astype(np.float32).reshape(-1)
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

            second_score = (
                float(similarities[order[1]])
                if len(order) > 1
                else -1.0
            )

            if best_score < threshold:
                return None

            if len(order) > 1 and (best_score - second_score) < AMBIGUITY_MARGIN:
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

    person_id = create_person(person_name)

    embedding_dir = KNOWN_FACES_DIR / str(person_id)
    embedding_dir.mkdir(parents=True, exist_ok=True)

    embedding_id_placeholder = "new"
    path = embedding_dir / f"{embedding_id_placeholder}.npy"
    np.save(path, _normalize(embedding))

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
            stored_embedding = np.load(
                path
            ).astype(np.float32)

            stored_embedding = _normalize(
                stored_embedding
            )

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