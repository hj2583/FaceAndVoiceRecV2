"""Speaker-embedding extraction and voiceprint matching.

Mirrors face_core.py's shape: extract a fixed-size, L2-normalized
embedding, then match it against enrolled voiceprints by cosine similarity.
"""

import logging
import threading
from pathlib import Path

import numpy as np

import config
from database import get_conn, list_voice_embeddings

logger = logging.getLogger(__name__)

EMBEDDING_DIM = config.VOICE_EMBEDDING_DIM

_MODEL_LOCK = threading.RLock()
_MODEL = None

# Minimum audio length SpeechBrain's ECAPA-TDNN model can embed reliably.
_MIN_SAMPLES = 400


def _normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        return None
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return (vector / norm).astype(np.float32)


def _load_model():
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            from speechbrain.inference.speaker import EncoderClassifier

            logger.info("Loading SpeechBrain speaker embedding model...")
            _MODEL = EncoderClassifier.from_hparams(source=config.VOICE_EMBEDDING_MODEL)
            logger.info("SpeechBrain speaker embedding model loaded.")
        return _MODEL


def extract_voice_embedding(pcm_int16_bytes, sample_rate=16000):
    """Extract a speaker embedding from raw PCM16 mono audio, or None on failure."""
    if not pcm_int16_bytes:
        return None

    audio = np.frombuffer(pcm_int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size < _MIN_SAMPLES:
        return None

    try:
        import torch

        model = _load_model()
        with torch.no_grad():
            tensor = torch.from_numpy(audio).unsqueeze(0)
            embedding = model.encode_batch(tensor)
        embedding = embedding.squeeze().cpu().numpy()
        embedding = _normalize(embedding)
        if embedding is None or embedding.shape[0] != EMBEDDING_DIM:
            return None
        return embedding
    except Exception:
        logger.exception("Voice embedding extraction failed")
        return None


def match_voice_embedding(query_embedding, threshold=None, ambiguity_margin=None):
    """Compare a query embedding against every enrolled voiceprint.

    Returns {"person_id", "person_name", "similarity"} or None.
    """
    threshold = config.VOICE_MATCH_THRESHOLD if threshold is None else threshold
    ambiguity_margin = (
        config.VOICE_AMBIGUITY_MARGIN if ambiguity_margin is None else ambiguity_margin
    )

    query = _normalize(query_embedding)
    if query is None:
        return None

    rows = list_voice_embeddings()
    if not rows:
        return None

    scores = []
    for embedding_id, person_id, person_name, embedding_path, _quality in rows:
        path = Path(embedding_path)
        if not path.exists():
            continue
        try:
            stored = _normalize(np.load(path))
            if stored is None or stored.shape[0] != EMBEDDING_DIM:
                continue
            similarity = float(np.dot(query, stored))
            scores.append((similarity, person_id, person_name))
        except Exception:
            logger.exception("Failed loading voice embedding: %s", path)

    if not scores:
        return None

    scores.sort(key=lambda item: item[0], reverse=True)
    best_similarity, best_person_id, best_person_name = scores[0]

    second_similarity = next(
        (score for score, person_id, _name in scores[1:] if person_id != best_person_id),
        -1.0,
    )

    if best_similarity < threshold:
        return None
    if second_similarity >= 0.0 and (best_similarity - second_similarity) < ambiguity_margin:
        return None

    return {
        "person_id": best_person_id,
        "person_name": best_person_name,
        "similarity": best_similarity,
    }
