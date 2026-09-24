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


from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from audio_core import read_wav_pcm, SAMPLE_RATE, SAMPLE_WIDTH
from database import (
    add_unknown_voice_sample,
    create_unknown_voice,
    list_unknown_voice_samples_with_embeddings,
)


def _read_region_pcm(wav_path, start_seconds, end_seconds):
    pcm = read_wav_pcm(wav_path)
    start_byte = int(start_seconds * SAMPLE_RATE) * SAMPLE_WIDTH
    end_byte = int(end_seconds * SAMPLE_RATE) * SAMPLE_WIDTH
    return pcm[start_byte:end_byte]


def find_matching_unknown_voice(query_embedding, threshold=None):
    threshold = config.VOICE_MATCH_THRESHOLD if threshold is None else threshold
    query = _normalize(query_embedding)
    if query is None:
        return None

    rows = list_unknown_voice_samples_with_embeddings()
    if not rows:
        return None

    scores = []
    for sample_id, unknown_voice_id, embedding_path, _quality in rows:
        path = Path(embedding_path)
        if not path.exists():
            continue
        try:
            stored = _normalize(np.load(path))
            if stored is None:
                continue
            scores.append((float(np.dot(query, stored)), unknown_voice_id, sample_id))
        except Exception:
            logger.exception("Failed loading unknown voice embedding: %s", path)

    if not scores:
        return None

    scores.sort(key=lambda item: item[0], reverse=True)
    best_similarity, best_unknown_voice_id, best_sample_id = scores[0]
    if best_similarity < threshold:
        return None

    return {
        "unknown_voice_id": best_unknown_voice_id,
        "similarity": best_similarity,
        "sample_id": best_sample_id,
    }


def register_unknown_voice(label, embedding):
    config.UNKNOWN_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace("/", "_").replace("\\", "_")
    embedding_path = config.UNKNOWN_VOICES_DIR / f"{safe_label}.npy"
    np.save(embedding_path, _normalize(embedding))
    unknown_voice_id = create_unknown_voice(label, embedding_path)
    add_unknown_voice_sample(unknown_voice_id, embedding_path, quality=1.0)
    return unknown_voice_id


def diarize_meeting_audio(wav_path, speech_regions):
    """Cluster VAD speech regions into speakers and label each region.

    Returns [(speaker_label, start_seconds, end_seconds), ...] suitable for
    transcription_core.transcribe_with_diarization's `diarize` parameter.
    """
    embeddings = []
    valid_regions = []
    for start, end in speech_regions:
        pcm = _read_region_pcm(wav_path, start, end)
        embedding = extract_voice_embedding(pcm)
        if embedding is None:
            continue
        embeddings.append(embedding)
        valid_regions.append((start, end))

    if not embeddings:
        return []

    if len(embeddings) == 1:
        cluster_ids = [1]
    else:
        distances = pdist(np.vstack(embeddings), metric="cosine")
        linkage_matrix = linkage(distances, method="average")
        cluster_ids = fcluster(
            linkage_matrix,
            t=config.VOICE_CLUSTER_DISTANCE_THRESHOLD,
            criterion="distance",
        )

    cluster_centroids = {}
    for cluster_id, embedding in zip(cluster_ids, embeddings):
        cluster_centroids.setdefault(cluster_id, []).append(embedding)
    cluster_centroids = {
        cluster_id: _normalize(np.mean(vectors, axis=0))
        for cluster_id, vectors in cluster_centroids.items()
    }

    cluster_labels = {}
    unknown_counter = 0
    for cluster_id, centroid in cluster_centroids.items():
        match = match_voice_embedding(centroid) if centroid is not None else None
        if match is not None:
            cluster_labels[cluster_id] = match["person_name"]
            continue
        unknown_counter += 1
        cluster_labels[cluster_id] = f"Unknown Speaker {unknown_counter}"

    return [
        (cluster_labels[cluster_id], start, end)
        for cluster_id, (start, end) in zip(cluster_ids, valid_regions)
    ]
