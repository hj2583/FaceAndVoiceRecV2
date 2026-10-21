import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from face_voice_rec import config
from face_voice_rec.storage.qdrant_store import (
    QdrantVectorStore,
    create_qdrant_vector_store,
    embedding_point_id,
    unknown_point_id,
)


def test_qdrant_vector_store_upserts_and_searches_points():
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(client)
    store.ensure_collection("face_embeddings", vector_size=3)

    store.upsert(
        "face_embeddings",
        point_id=17,
        vector=[1.0, 0.0, 0.0],
        payload={"embedding_id": 17, "person_id": 4, "quality": 0.9},
    )

    points = store.search("face_embeddings", [1.0, 0.0, 0.0], limit=1)

    assert len(points) == 1
    assert points[0].id == 17
    assert points[0].payload == {"embedding_id": 17, "person_id": 4, "quality": 0.9}
    assert points[0].score == pytest.approx(1.0)
    client.close()


def test_qdrant_vector_store_rejects_wrong_dimensions_and_nonfinite_values():
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(client)
    store.ensure_collection("voice_embeddings", vector_size=3)

    with pytest.raises(ValueError, match="expected 3"):
        store.upsert("voice_embeddings", 1, [1.0, 0.0], {})
    with pytest.raises(ValueError, match="finite"):
        store.search("voice_embeddings", [1.0, np.nan, 0.0])
    client.close()


def test_qdrant_vector_store_rejects_collection_dimension_mismatch():
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(client)
    store.ensure_collection("face_embeddings", vector_size=3)

    with pytest.raises(ValueError, match="expected 4"):
        store.ensure_collection("face_embeddings", vector_size=4)
    client.close()


def test_factory_initializes_face_and_voice_collections():
    client = QdrantClient(":memory:")

    store = create_qdrant_vector_store(client)

    assert client.collection_exists(config.QDRANT_FACE_COLLECTION)
    assert client.collection_exists(config.QDRANT_VOICE_COLLECTION)
    assert store._dimensions == {
        config.QDRANT_FACE_COLLECTION: config.EMBEDDING_DIM,
        config.QDRANT_VOICE_COLLECTION: config.VOICE_EMBEDDING_DIM,
    }
    client.close()


def test_grouped_search_returns_best_point_per_person_and_retrieves_vectors():
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(client)
    store.ensure_collection("face_embeddings", vector_size=3)
    first_id = embedding_point_id("face", 17)
    store.upsert_many(
        "face_embeddings",
        [
            (first_id, [1.0, 0.0, 0.0], {"person_id": 4, "embedding_id": 17}),
            (embedding_point_id("face", 18), [0.99, 0.01, 0.0], {"person_id": 4, "embedding_id": 18}),
            (embedding_point_id("face", 19), [0.8, 0.6, 0.0], {"person_id": 5, "embedding_id": 19}),
        ],
    )

    groups = store.search_grouped(
        "face_embeddings",
        [1.0, 0.0, 0.0],
        group_by="person_id",
        limit=2,
    )
    retrieved = store.retrieve("face_embeddings", [first_id])

    assert [group.id for group in groups] == [4, 5]
    assert groups[0].hits[0].payload["embedding_id"] == 17
    assert len(retrieved) == 1
    assert retrieved[0].vector == pytest.approx([1.0, 0.0, 0.0])
    client.close()


def test_enrolled_and_unknown_point_ids_are_stable_and_distinct():
    assert embedding_point_id("face", 17) == embedding_point_id("face", 17)
    assert unknown_point_id("face", 17) == unknown_point_id("face", 17)
    assert embedding_point_id("face", 17) != unknown_point_id("face", 17)
    assert unknown_point_id("face", 17) != unknown_point_id("voice", 17)


def test_count_can_filter_enrolled_face_vectors():
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(client)
    store.ensure_collection("face_embeddings", vector_size=3)
    store.upsert_many(
        "face_embeddings",
        [
            (1, [1.0, 0.0, 0.0], {"record_type": "enrolled_embedding", "modality": "face"}),
            (2, [0.0, 1.0, 0.0], {"record_type": "unknown_sample", "modality": "face"}),
        ],
    )

    count = store.count(
        "face_embeddings",
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

    assert count == 1
    client.close()