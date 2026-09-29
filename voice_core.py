"""Speaker-embedding extraction and voiceprint matching.

Mirrors face_core.py's shape: extract a fixed-size, L2-normalized
embedding, then match it against enrolled voiceprints by cosine similarity.
"""

import logging
import json
import threading
import uuid
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


def _embedding_window_quality(pcm_int16_bytes):
    """Return signal diagnostics for a candidate speaker window."""
    audio = np.frombuffer(pcm_int16_bytes, dtype=np.int16).astype(np.float32)
    if audio.size == 0:
        return None

    normalized = audio / 32768.0
    speech_mask = np.abs(normalized) >= 0.01
    rms = float(np.sqrt(np.mean(np.square(normalized))))
    clipping_ratio = float(np.mean(np.abs(audio) >= 32760))
    speech_ratio = float(np.mean(speech_mask))
    quality = max(0.0, min(1.0, speech_ratio))
    if rms < config.VOICE_MIN_RMS:
        quality *= rms / max(config.VOICE_MIN_RMS, 1e-8)
    if clipping_ratio > config.VOICE_MAX_CLIPPING_RATIO:
        quality *= max(
            0.0,
            1.0 - (clipping_ratio - config.VOICE_MAX_CLIPPING_RATIO),
        )
    return {
        "speech_ratio": speech_ratio,
        "rms": rms,
        "clipping_ratio": clipping_ratio,
        "quality": quality,
    }


def _write_diagnostics(records, summary):
    if not config.VOICE_DIAGNOSTICS_ENABLED:
        return
    path = Path(config.VOICE_DIAGNOSTICS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"summary": summary, "windows": records}, indent=2),
        encoding="utf-8",
    )


def _load_model():
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            from speechbrain.inference.speaker import EncoderClassifier

            from audio_core import TORCH_DEVICE

            logger.info("Loading SpeechBrain speaker embedding model on %s...", TORCH_DEVICE)
            _MODEL = EncoderClassifier.from_hparams(
                source=config.VOICE_EMBEDDING_MODEL,
                run_opts={"device": TORCH_DEVICE},
            )
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

        from audio_core import TORCH_DEVICE

        model = _load_model()
        with torch.no_grad():
            tensor = torch.from_numpy(audio).unsqueeze(0).to(TORCH_DEVICE)
            embedding = model.encode_batch(tensor)
        embedding = embedding.squeeze().cpu().numpy()
        embedding = _normalize(embedding)
        if embedding is None or embedding.shape[0] != EMBEDDING_DIM:
            return None
        return embedding
    except Exception:
        logger.exception("Voice embedding extraction failed")
        return None


def has_enrolled_voiceprints():
    """Cheap check used to skip voice embedding work when nothing can match."""
    return bool(list_voice_embeddings())


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

    scores_by_person = {}
    for embedding_id, person_id, person_name, embedding_path, quality in rows:
        path = Path(embedding_path)
        if not path.exists():
            continue
        try:
            stored = _normalize(np.load(path))
            if stored is None or stored.shape[0] != EMBEDDING_DIM:
                continue
            similarity = float(np.dot(query, stored))
            scores_by_person.setdefault(
                person_id,
                {"name": person_name, "scores": []},
            )["scores"].append((similarity, float(quality)))
        except Exception:
            logger.exception("Failed loading voice embedding: %s", path)

    if not scores_by_person:
        return None

    person_scores = []
    for person_id, profile in scores_by_person.items():
        candidates = sorted(
            profile["scores"],
            key=lambda item: item[0],
            reverse=True,
        )[: config.VOICE_PROFILE_TOP_K]
        weights = np.asarray(
            [max(0.1, quality) for _similarity, quality in candidates],
            dtype=np.float32,
        )
        similarities = np.asarray(
            [similarity for similarity, _quality in candidates],
            dtype=np.float32,
        )
        profile_similarity = float(np.average(similarities, weights=weights))
        best_similarity = float(similarities[0])
        combined_similarity = 0.7 * best_similarity + 0.3 * profile_similarity
        person_scores.append(
            (combined_similarity, best_similarity, person_id, profile["name"])
        )

    person_scores.sort(key=lambda item: item[0], reverse=True)
    best_score, best_sample_similarity, best_person_id, best_person_name = person_scores[0]

    second_similarity = person_scores[1][0] if len(person_scores) > 1 else -1.0

    if best_score < threshold:
        return None
    if second_similarity >= 0.0 and (best_score - second_similarity) < ambiguity_margin:
        return None

    return {
        "person_id": best_person_id,
        "person_name": best_person_name,
        "similarity": best_score,
        "best_sample_similarity": best_sample_similarity,
        "margin": best_score - second_similarity if second_similarity >= 0.0 else None,
    }


from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from audio_core import read_wav_pcm, SAMPLE_RATE, SAMPLE_WIDTH
from database import (
    add_unknown_voice_sample,
    create_unknown_voice,
    delete_unknown_voice,
    list_unknown_voice_samples_with_embeddings,
    list_unknown_voices,
    update_unknown_voice,
)

_PENDING_UNKNOWN_VOICE_LABEL = "Unknown Speaker (pending)"


def _slice_pcm(pcm, start_seconds, end_seconds):
    start_byte = int(start_seconds * SAMPLE_RATE) * SAMPLE_WIDTH
    end_byte = int(end_seconds * SAMPLE_RATE) * SAMPLE_WIDTH
    return pcm[start_byte:end_byte]


def _read_region_pcm(wav_path, start_seconds, end_seconds):
    pcm = read_wav_pcm(wav_path)
    return _slice_pcm(pcm, start_seconds, end_seconds)


def find_matching_unknown_voice(query_embedding, threshold=None, candidate_ids=None):
    """candidate_ids, if given, restricts matching to those unknown_voice_ids."""
    threshold = config.VOICE_MATCH_THRESHOLD if threshold is None else threshold
    query = _normalize(query_embedding)
    if query is None:
        return None

    rows = list_unknown_voice_samples_with_embeddings()
    if candidate_ids is not None:
        rows = [row for row in rows if row[1] in candidate_ids]
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


def _save_unknown_voice_embedding(unknown_voice_id, embedding):
    config.UNKNOWN_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    embedding_path = (
        config.UNKNOWN_VOICES_DIR
        / f"unknown_voice_{int(unknown_voice_id)}_{uuid.uuid4().hex}.npy"
    )
    np.save(embedding_path, _normalize(embedding))
    return embedding_path


def _register_unknown_voice(embedding, label=None):
    # The row is created first so its auto-increment id can name the label/file.
    unknown_voice_id = create_unknown_voice(label or _PENDING_UNKNOWN_VOICE_LABEL, None)
    label = label or f"Unknown Speaker {unknown_voice_id}"
    try:
        embedding_path = _save_unknown_voice_embedding(unknown_voice_id, embedding)
        update_unknown_voice(unknown_voice_id, label, embedding_path)
        add_unknown_voice_sample(unknown_voice_id, embedding_path, quality=1.0)
    except Exception:
        delete_unknown_voice(unknown_voice_id)
        raise
    return unknown_voice_id, label


def register_unknown_voice(label, embedding):
    return _register_unknown_voice(embedding, label)[0]


def register_new_unknown_voice(embedding):
    """Register an unknown voice labeled "Unknown Speaker {unknown_voice_id}".

    The id-based label is unique across the whole database, so resolving it
    by label can never relabel another meeting's speaker.
    Returns (unknown_voice_id, label).
    """
    return _register_unknown_voice(embedding)


def add_unknown_voice_embedding_sample(unknown_voice_id, embedding, quality=1.0):
    embedding_path = _save_unknown_voice_embedding(unknown_voice_id, embedding)
    return add_unknown_voice_sample(unknown_voice_id, embedding_path, quality=quality)


def _unknown_voice_label(unknown_voice_id):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT label FROM unknown_voices WHERE unknown_voice_id=?",
            (int(unknown_voice_id),),
        ).fetchone()
    return row[0] if row else None


def _label_unmatched_cluster(centroid, candidate_ids, register_if_unmatched=True):
    """Reuse a stored unknown label, optionally registering a new identity."""
    existing = find_matching_unknown_voice(centroid, candidate_ids=candidate_ids)
    if existing is not None:
        label = _unknown_voice_label(existing["unknown_voice_id"])
        if label is not None:
            add_unknown_voice_embedding_sample(
                existing["unknown_voice_id"],
                centroid,
                quality=existing["similarity"],
            )
            return label
    if not register_if_unmatched:
        return None
    _unknown_voice_id, label = register_new_unknown_voice(centroid)
    return label


def _match_in_meeting_identity(centroid, identity_profiles):
    """Match a split cluster to an identity already supported in this meeting."""
    query = _normalize(centroid)
    if query is None or not identity_profiles:
        return None

    scores_by_label = {}
    for label, profile_embedding in identity_profiles:
        profile = _normalize(profile_embedding)
        if profile is None:
            continue
        scores_by_label[label] = max(
            scores_by_label.get(label, -1.0),
            float(np.dot(query, profile)),
        )

    if not scores_by_label:
        return None

    ranked = sorted(scores_by_label.items(), key=lambda item: item[1], reverse=True)
    best_label, best_similarity = ranked[0]
    second_similarity = ranked[1][1] if len(ranked) > 1 else -1.0
    margin = best_similarity - second_similarity
    if best_similarity < config.VOICE_IN_MEETING_IDENTITY_THRESHOLD:
        return None
    if second_similarity >= 0.0 and margin < config.VOICE_IN_MEETING_IDENTITY_MARGIN:
        return None
    return {
        "label": best_label,
        "similarity": best_similarity,
        "margin": margin if second_similarity >= 0.0 else None,
    }


def _sliding_windows(start, end, window_seconds, step_seconds):
    """Split [start, end) into overlapping windows; short regions stay whole."""
    if end - start <= window_seconds:
        return [(start, end)]
    windows = []
    window_start = start
    while window_start + window_seconds < end:
        windows.append((window_start, window_start + window_seconds))
        window_start += step_seconds
    # Final window aligned to the region end so the tail is always covered.
    windows.append((end - window_seconds, end))
    return windows


def diarize_meeting_audio(wav_path, speech_regions):
    """Cluster sliding windows of VAD speech regions into speakers.

    Returns [(speaker_label, start_seconds, end_seconds), ...] (one entry per
    window) suitable for transcription_core.transcribe_with_diarization's
    `diarize` parameter. Unmatched clusters are persisted as unknown voices.
    """
    embeddings = []
    valid_windows = []
    window_qualities = []
    diagnostics = []
    pcm = read_wav_pcm(wav_path)
    candidate_count = 0
    for region_start, region_end in speech_regions:
        for start, end in _sliding_windows(
            region_start,
            region_end,
            config.VOICE_DIARIZATION_WINDOW_SECONDS,
            config.VOICE_DIARIZATION_STEP_SECONDS,
        ):
            candidate_count += 1
            window_pcm = _slice_pcm(pcm, start, end)
            quality = _embedding_window_quality(window_pcm)
            if quality is None:
                diagnostics.append({
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "accepted": False,
                    "rejection_reason": "empty_audio",
                })
                continue
            if (
                quality["speech_ratio"] < config.VOICE_MIN_SPEECH_RATIO
                or quality["rms"] < config.VOICE_MIN_RMS
                or quality["clipping_ratio"] > config.VOICE_MAX_CLIPPING_RATIO
            ):
                logger.info(
                    "Rejected speaker window %.2f-%.2f: quality=%.3f speech_ratio=%.3f rms=%.4f clipping=%.3f",
                    start,
                    end,
                    quality["quality"],
                    quality["speech_ratio"],
                    quality["rms"],
                    quality["clipping_ratio"],
                )
                diagnostics.append({
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    **quality,
                    "accepted": False,
                    "rejection_reason": "quality_gate",
                })
                continue
            embedding = extract_voice_embedding(window_pcm)
            if embedding is None:
                diagnostics.append({
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    **quality,
                    "accepted": False,
                    "rejection_reason": "embedding_failed",
                })
                continue
            embeddings.append(embedding)
            valid_windows.append((start, end))
            window_qualities.append(quality["quality"])
            diagnostics.append({
                "start": start,
                "end": end,
                "duration": end - start,
                **quality,
                "accepted": True,
            })

    if not embeddings:
        _write_diagnostics(
            diagnostics,
            {
                "audio_path": str(wav_path),
                "speech_region_count": len(speech_regions),
                "candidate_windows": candidate_count,
                "accepted_embeddings": 0,
                "rejected_embeddings": candidate_count,
                "cluster_count": 0,
            },
        )
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
    for cluster_id, embedding, quality in zip(cluster_ids, embeddings, window_qualities):
        cluster_centroids.setdefault(cluster_id, []).append((embedding, quality))
    cluster_centroids = {
        cluster_id: _normalize(
            np.average(
                np.vstack([embedding for embedding, _quality in vectors]),
                axis=0,
                weights=np.asarray([quality for _embedding, quality in vectors]),
            )
        )
        for cluster_id, vectors in cluster_centroids.items()
    }

    # Keep existing unknown matching scoped to identities that predate this run,
    # then separately allow strong, unambiguous reuse among this meeting's clusters.
    preexisting_unknown_ids = None
    cluster_labels = {}
    meeting_identity_profiles = []
    for cluster_id, centroid in cluster_centroids.items():
        if centroid is None:
            cluster_labels[cluster_id] = "Unknown Speaker"
            continue
        match = match_voice_embedding(centroid)
        if match is not None:
            cluster_labels[cluster_id] = match["person_name"]
            meeting_identity_profiles.append((match["person_name"], centroid))
            continue
        if preexisting_unknown_ids is None:
            preexisting_unknown_ids = {row[0] for row in list_unknown_voices()}
        in_meeting_match = _match_in_meeting_identity(
            centroid,
            meeting_identity_profiles,
        )
        label = None
        if in_meeting_match is not None:
            label = in_meeting_match["label"]
            if label.startswith("Unknown Speaker"):
                unknown_id = int(label.rsplit(" ", 1)[-1])
                add_unknown_voice_embedding_sample(
                    unknown_id,
                    centroid,
                    quality=in_meeting_match["similarity"],
                )
            logger.info(
                "Reused in-meeting speaker identity %s for cluster %s (similarity=%.3f margin=%s)",
                label,
                cluster_id,
                in_meeting_match["similarity"],
                in_meeting_match["margin"],
            )
        else:
            label = _label_unmatched_cluster(
                centroid,
                preexisting_unknown_ids,
                register_if_unmatched=False,
            )
        if label is None:
            _unknown_voice_id, label = register_new_unknown_voice(centroid)
        cluster_labels[cluster_id] = label
        meeting_identity_profiles.append((label, centroid))

    for diagnostic, cluster_id in zip(
        (item for item in diagnostics if item.get("accepted")),
        cluster_ids,
    ):
        diagnostic["cluster_id"] = int(cluster_id)
        diagnostic["final_speaker"] = cluster_labels[cluster_id]

    _write_diagnostics(
        diagnostics,
        {
            "audio_path": str(wav_path),
            "speech_region_count": len(speech_regions),
            "candidate_windows": candidate_count,
            "accepted_embeddings": len(embeddings),
            "rejected_embeddings": candidate_count - len(embeddings),
            "average_embedding_quality": float(np.mean(window_qualities)),
            "cluster_count": len(cluster_centroids),
            "cluster_sizes": {
                str(cluster_id): int(sum(item == cluster_id for item in cluster_ids))
                for cluster_id in set(cluster_ids)
            },
        },
    )

    intervals = []
    for index, (cluster_id, (start, end)) in enumerate(zip(cluster_ids, valid_windows)):
        next_start = (
            valid_windows[index + 1][0]
            if index + 1 < len(valid_windows)
            and valid_windows[index + 1][0] < end
            else end
        )
        interval_end = max(start, next_start)
        if interval_end > start:
            intervals.append((cluster_labels[cluster_id], start, interval_end))

    merged = []
    for label, start, end in intervals:
        if merged and merged[-1][0] == label and merged[-1][2] >= start:
            merged[-1] = (label, merged[-1][1], max(merged[-1][2], end))
        else:
            merged.append((label, start, end))
    return merged
