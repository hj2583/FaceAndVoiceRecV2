"""Prepare and apply repeatable backfills for enrolled identity embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from face_voice_rec import config
from face_voice_rec.storage.safe_numpy import load_embedding_array
from face_voice_rec.storage import database
from face_voice_rec.storage.qdrant_store import (
    QdrantVectorStore,
    embedding_point_id,
    unknown_point_id,
)


@dataclass(frozen=True)
class PreparedBackfill:
    face_points: tuple[tuple[str, np.ndarray, dict[str, Any]], ...]
    voice_points: tuple[tuple[str, np.ndarray, dict[str, Any]], ...]
    unknown_face_points: tuple[tuple[str, np.ndarray, dict[str, Any]], ...]
    unknown_voice_points: tuple[tuple[str, np.ndarray, dict[str, Any]], ...]
    face_skips: tuple[tuple[int, str], ...]
    voice_skips: tuple[tuple[int, str], ...]
    unknown_face_skips: tuple[tuple[int, str], ...]
    unknown_voice_skips: tuple[tuple[int, str], ...]

    @property
    def face_skipped(self) -> int:
        return len(self.face_skips)

    @property
    def voice_skipped(self) -> int:
        return len(self.voice_skips)

    @property
    def skipped(self) -> int:
        return (
            self.face_skipped
            + self.voice_skipped
            + len(self.unknown_face_skips)
            + len(self.unknown_voice_skips)
        )


def _load_vector(path: str, dimension: int) -> np.ndarray:
    loaded = load_embedding_array(path)
    try:
        value = loaded.tolist() if isinstance(loaded, np.ndarray) else loaded
        if isinstance(value, np.ndarray):
            value = value.tolist()
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    finally:
        if isinstance(loaded, np.lib.npyio.NpzFile):
            loaded.close()

    if vector.size != dimension:
        raise ValueError(f"wrong dimension: expected {dimension}, got {vector.size}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("vector contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("vector has zero length")
    return vector / norm


def _prepare_rows(rows, modality: str, dimension: int):
    points = []
    skips = []
    for row in rows:
        embedding_id, person_id, _person_name, embedding_path, *extra = row
        try:
            vector = _load_vector(embedding_path, dimension)
            quality = float(extra[-1] or 0.0)
            if not np.isfinite(quality):
                raise ValueError("quality is non-finite")
        except FileNotFoundError:
            skips.append((int(embedding_id), "file not found"))
            continue
        except PermissionError:
            skips.append((int(embedding_id), "permission denied"))
            continue
        except EOFError:
            skips.append((int(embedding_id), "file is truncated"))
            continue
        except OSError:
            skips.append((int(embedding_id), "file could not be read"))
            continue
        except (TypeError, ValueError) as error:
            skips.append((int(embedding_id), str(error)))
            continue
        points.append(
            (
                embedding_point_id(modality, int(embedding_id)),
                vector,
                {
                    "record_type": "enrolled_embedding",
                    "modality": modality,
                    "embedding_id": int(embedding_id),
                    "person_id": int(person_id),
                    "quality": quality,
                },
            )
        )
    return tuple(points), tuple(skips)


def _prepare_unknown_rows(rows, modality: str, dimension: int):
    points = []
    skips = []
    for row in rows:
        sample_id, unknown_id, embedding_path, quality = row
        try:
            vector = _load_vector(embedding_path, dimension)
            normalized_quality = float(quality or 0.0)
            if not np.isfinite(normalized_quality):
                raise ValueError("quality is non-finite")
        except FileNotFoundError:
            skips.append((int(sample_id), "file not found"))
            continue
        except PermissionError:
            skips.append((int(sample_id), "permission denied"))
            continue
        except EOFError:
            skips.append((int(sample_id), "file is truncated"))
            continue
        except OSError:
            skips.append((int(sample_id), "file could not be read"))
            continue
        except (TypeError, ValueError) as error:
            skips.append((int(sample_id), str(error)))
            continue
        points.append(
            (
                unknown_point_id(modality, int(sample_id)),
                vector,
                {
                    "record_type": "unknown_sample",
                    "modality": modality,
                    "sample_id": int(sample_id),
                    "unknown_id": int(unknown_id),
                    "quality": normalized_quality,
                },
            )
        )
    return tuple(points), tuple(skips)


def prepare_known_embedding_backfill() -> PreparedBackfill:
    """Read and validate all identity vectors without changing Qdrant."""
    face_points, face_skips = _prepare_rows(
        database.list_embeddings(), "face", config.EMBEDDING_DIM
    )
    voice_points, voice_skips = _prepare_rows(
        database.list_voice_embeddings(), "voice", config.VOICE_EMBEDDING_DIM
    )
    unknown_face_points, unknown_face_skips = _prepare_unknown_rows(
        database.list_unknown_samples_with_embeddings(),
        "face",
        config.EMBEDDING_DIM,
    )
    unknown_voice_points, unknown_voice_skips = _prepare_unknown_rows(
        database.list_unknown_voice_samples_with_embeddings(),
        "voice",
        config.VOICE_EMBEDDING_DIM,
    )
    return PreparedBackfill(
        face_points,
        voice_points,
        unknown_face_points,
        unknown_voice_points,
        face_skips,
        voice_skips,
        unknown_face_skips,
        unknown_voice_skips,
    )


def apply_known_embedding_backfill(
    store: QdrantVectorStore,
    prepared: PreparedBackfill,
) -> None:
    """Upsert a validated backfill; deterministic IDs make retries idempotent."""
    if prepared.skipped:
        raise ValueError(
            f"Refusing incomplete backfill: {prepared.skipped} source embeddings "
            "could not be read or validated"
        )
    store.upsert_many(config.QDRANT_FACE_COLLECTION, prepared.face_points)
    store.upsert_many(config.QDRANT_VOICE_COLLECTION, prepared.voice_points)
    store.upsert_many(config.QDRANT_FACE_COLLECTION, prepared.unknown_face_points)
    store.upsert_many(config.QDRANT_VOICE_COLLECTION, prepared.unknown_voice_points)