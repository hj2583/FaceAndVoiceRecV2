import numpy as np
from qdrant_client import QdrantClient

from face_voice_rec import config
from face_voice_rec.domain.face import face_core
from face_voice_rec.domain.face.face_core import FaceIndex
from face_voice_rec.storage.qdrant_store import (
    QdrantVectorStore,
    embedding_point_id,
    unknown_point_id,
)


def _vector(first: float, second: float = 0.0) -> np.ndarray:
    value = np.zeros(config.EMBEDDING_DIM, dtype=np.float32)
    value[0] = first
    value[1] = second
    return value


def _store_with_face_vectors(vectors):
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(client)
    store.ensure_collection(config.QDRANT_FACE_COLLECTION, config.EMBEDDING_DIM)
    store.upsert_many(
        config.QDRANT_FACE_COLLECTION,
        [
            (
                embedding_point_id("face", embedding_id),
                vector,
                {
                    "record_type": "enrolled_embedding",
                    "modality": "face",
                    "embedding_id": embedding_id,
                    "person_id": person_id,
                    "quality": 1.0,
                },
            )
            for embedding_id, person_id, vector in vectors
        ],
    )
    return client, store


def test_face_index_returns_best_match_when_people_are_separated():
    client, store = _store_with_face_vectors(
        [
            (17, 4, _vector(1.0)),
            (18, 4, _vector(0.99, 0.01)),
            (19, 5, _vector(0.7, 0.7)),
        ]
    )

    result = FaceIndex(store).search(_vector(1.0))

    assert result["embedding_id"] == 17
    assert result["similarity"] > 0.99
    assert result["second_similarity"] < 0.95
    client.close()


def test_face_index_rejects_ambiguous_person_match():
    client, store = _store_with_face_vectors(
        [
            (17, 4, _vector(1.0)),
            (18, 5, _vector(0.99, 0.01)),
        ]
    )

    assert FaceIndex(store).search(_vector(1.0)) is None
    client.close()


def test_unknown_face_search_uses_qdrant_and_preserves_threshold(monkeypatch):
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(client)
    store.ensure_collection(config.QDRANT_FACE_COLLECTION, config.EMBEDDING_DIM)
    store.upsert(
        config.QDRANT_FACE_COLLECTION,
        unknown_point_id("face", 31),
        _vector(1.0),
        {
            "record_type": "unknown_sample",
            "modality": "face",
            "sample_id": 31,
            "unknown_id": 9,
            "quality": 0.9,
        },
    )
    monkeypatch.setattr(face_core, "get_qdrant_vector_store", lambda: store)
    monkeypatch.setattr(
        face_core,
        "list_unknown_samples_with_embeddings",
        lambda: [(31, 9, "unused.npy", 0.9)],
    )

    result = face_core.find_matching_unknown(_vector(1.0))

    assert result == {"unknown_id": 9, "similarity": 1.0, "sample_id": 31}
    client.close()