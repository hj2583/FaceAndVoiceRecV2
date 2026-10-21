from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient, models

from face_voice_rec import config


_UPSERT_BATCH_SIZE = 256


def embedding_point_id(modality: str, embedding_id: int) -> str:
    """Return the stable Qdrant point ID for a relational embedding row."""
    return str(uuid5(NAMESPACE_URL, f"face-voice-rec:{modality}:enrolled:{embedding_id}"))


def unknown_point_id(modality: str, sample_id: int) -> str:
    """Return the stable Qdrant point ID for an unresolved sample row."""
    return str(uuid5(NAMESPACE_URL, f"face-voice-rec:{modality}:unknown:{sample_id}"))


class QdrantVectorStore:
    """Small Qdrant boundary for identity embeddings."""

    def __init__(self, client: QdrantClient):
        self._client = client
        self._dimensions: dict[str, int] = {}
        self._lock = threading.RLock()

    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        if not collection_name.strip():
            raise ValueError("Collection name cannot be empty")
        if vector_size <= 0:
            raise ValueError("Vector size must be positive")

        with self._lock:
            if self._client.collection_exists(collection_name):
                collection = self._client.get_collection(collection_name)
                config = collection.config.params.vectors
                if not isinstance(config, models.VectorParams):
                    raise ValueError(
                        f"Collection {collection_name!r} must use unnamed vectors"
                    )
                if config.size != vector_size:
                    raise ValueError(
                        f"Collection {collection_name!r} has vector size "
                        f"{config.size}, expected {vector_size}"
                    )
                if config.distance != models.Distance.COSINE:
                    raise ValueError(
                        f"Collection {collection_name!r} must use cosine distance"
                    )
            else:
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
            for field_name, field_schema in (
                ("person_id", models.PayloadSchemaType.INTEGER),
                ("unknown_id", models.PayloadSchemaType.INTEGER),
                ("record_type", models.PayloadSchemaType.KEYWORD),
                ("modality", models.PayloadSchemaType.KEYWORD),
            ):
                self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                    wait=True,
                )
            self._dimensions[collection_name] = vector_size

    def upsert(
        self,
        collection_name: str,
        point_id: int | str,
        vector: Sequence[float],
        payload: Mapping[str, Any],
    ) -> None:
        self.upsert_many(collection_name, [(point_id, vector, payload)])

    def upsert_many(
        self,
        collection_name: str,
        points: Sequence[tuple[int | str, Sequence[float], Mapping[str, Any]]],
    ) -> None:
        if not points:
            return
        qdrant_points = []
        for point_id, vector, payload in points:
            normalized = self._validated_vector(collection_name, vector)
            qdrant_points.append(
                models.PointStruct(
                    id=point_id,
                    vector=normalized.tolist(),
                    payload=dict(payload),
                )
            )
        for start in range(0, len(qdrant_points), _UPSERT_BATCH_SIZE):
            self._client.upsert(
                collection_name=collection_name,
                points=qdrant_points[start : start + _UPSERT_BATCH_SIZE],
                wait=True,
            )

    def search(
        self,
        collection_name: str,
        query_vector: Sequence[float],
        limit: int = 10,
        score_threshold: float | None = None,
        query_filter: models.Filter | None = None,
    ) -> list[models.ScoredPoint]:
        if limit <= 0:
            raise ValueError("Search limit must be positive")
        query = self._validated_vector(collection_name, query_vector)
        response = self._client.query_points(
            collection_name=collection_name,
            query=query.tolist(),
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True,
        )
        return response.points

    def search_grouped(
        self,
        collection_name: str,
        query_vector: Sequence[float],
        group_by: str,
        limit: int,
        group_size: int = 1,
        score_threshold: float | None = None,
        query_filter: models.Filter | None = None,
    ):
        if limit <= 0 or group_size <= 0:
            raise ValueError("Search limit and group size must be positive")
        query = self._validated_vector(collection_name, query_vector)
        response = self._client.query_points_groups(
            collection_name=collection_name,
            query=query.tolist(),
            group_by=group_by,
            group_size=group_size,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True,
        )
        return response.groups

    def retrieve(
        self,
        collection_name: str,
        point_ids: Sequence[int | str],
    ):
        if collection_name not in self._dimensions:
            raise ValueError(f"Collection {collection_name!r} has not been initialized")
        if not point_ids:
            return []
        return self._client.retrieve(
            collection_name=collection_name,
            ids=list(point_ids),
            with_payload=True,
            with_vectors=True,
        )

    def count(
        self,
        collection_name: str,
        query_filter: models.Filter | None = None,
    ) -> int:
        if collection_name not in self._dimensions:
            raise ValueError(f"Collection {collection_name!r} has not been initialized")
        return int(
            self._client.count(
                collection_name=collection_name,
                count_filter=query_filter,
                exact=True,
            ).count
        )

    def delete(self, collection_name: str, point_ids: Sequence[int | str]) -> None:
        if collection_name not in self._dimensions:
            raise ValueError(f"Collection {collection_name!r} has not been initialized")
        if not point_ids:
            return
        self._client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=list(point_ids)),
            wait=True,
        )

    def _validated_vector(
        self,
        collection_name: str,
        vector: Sequence[float],
    ) -> np.ndarray:
        expected_size = self._dimensions.get(collection_name)
        if expected_size is None:
            raise ValueError(f"Collection {collection_name!r} has not been initialized")
        normalized = np.asarray(vector, dtype=np.float32).reshape(-1)
        if normalized.size != expected_size:
            raise ValueError(
                f"Vector has size {normalized.size}, expected {expected_size}"
            )
        if not np.all(np.isfinite(normalized)):
            raise ValueError("Vector values must be finite")
        return normalized


def create_qdrant_vector_store(
    client: QdrantClient | None = None,
) -> QdrantVectorStore:
    """Connect to Qdrant and ensure the application's embedding collections exist."""
    if client is None:
        client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            timeout=config.QDRANT_TIMEOUT_SECONDS,
        )

    store = QdrantVectorStore(client)
    store.ensure_collection(config.QDRANT_FACE_COLLECTION, config.EMBEDDING_DIM)
    store.ensure_collection(
        config.QDRANT_VOICE_COLLECTION,
        config.VOICE_EMBEDDING_DIM,
    )
    return store


@lru_cache(maxsize=1)
def get_qdrant_vector_store() -> QdrantVectorStore:
    """Return the process-wide Qdrant connection used by runtime workflows."""
    return create_qdrant_vector_store()