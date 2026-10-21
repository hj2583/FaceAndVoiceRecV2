import numpy as np
import pytest
from qdrant_client import QdrantClient

from face_voice_rec import config
from face_voice_rec.storage import qdrant_backfill
from face_voice_rec.storage.qdrant_store import QdrantVectorStore


def test_backfill_is_idempotent_and_keeps_only_relational_references(
    tmp_path, monkeypatch
):
    face_path = tmp_path / "face.npy"
    voice_path = tmp_path / "voice.npy"
    np.save(face_path, np.ones(config.EMBEDDING_DIM, dtype=np.float32))
    np.save(voice_path, np.ones(config.VOICE_EMBEDDING_DIM, dtype=np.float32))
    monkeypatch.setattr(
        qdrant_backfill.database,
        "list_embeddings",
        lambda: [(17, 4, "Alice", str(face_path), "private/image.jpg", 0.9)],
    )
    monkeypatch.setattr(
        qdrant_backfill.database,
        "list_voice_embeddings",
        lambda: [(21, 4, "Alice", str(voice_path), 0.8)],
    )
    monkeypatch.setattr(
        qdrant_backfill.database,
        "list_unknown_samples_with_embeddings",
        lambda: [],
    )
    monkeypatch.setattr(
        qdrant_backfill.database,
        "list_unknown_voice_samples_with_embeddings",
        lambda: [],
    )

    prepared = qdrant_backfill.prepare_known_embedding_backfill()
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(client)
    store.ensure_collection(config.QDRANT_FACE_COLLECTION, config.EMBEDDING_DIM)
    store.ensure_collection(config.QDRANT_VOICE_COLLECTION, config.VOICE_EMBEDDING_DIM)

    qdrant_backfill.apply_known_embedding_backfill(store, prepared)
    qdrant_backfill.apply_known_embedding_backfill(store, prepared)

    face_points = store.search(
        config.QDRANT_FACE_COLLECTION,
        np.ones(config.EMBEDDING_DIM, dtype=np.float32),
    )
    voice_points = store.search(
        config.QDRANT_VOICE_COLLECTION,
        np.ones(config.VOICE_EMBEDDING_DIM, dtype=np.float32),
    )
    assert len(face_points) == len(voice_points) == 1
    assert face_points[0].payload == {
        "record_type": "enrolled_embedding",
        "modality": "face",
        "embedding_id": 17,
        "person_id": 4,
        "quality": 0.9,
    }
    assert "image_path" not in face_points[0].payload
    client.close()


def test_backfill_refuses_to_write_when_any_source_embedding_is_invalid(
    tmp_path, monkeypatch
):
    valid_path = tmp_path / "face.npy"
    np.save(valid_path, np.ones(config.EMBEDDING_DIM, dtype=np.float32))
    monkeypatch.setattr(
        qdrant_backfill.database,
        "list_embeddings",
        lambda: [
            (17, 4, "Alice", str(valid_path), None, 0.9),
            (18, 4, "Alice", str(tmp_path / "missing.npy"), None, 0.9),
        ],
    )
    monkeypatch.setattr(qdrant_backfill.database, "list_voice_embeddings", lambda: [])
    monkeypatch.setattr(
        qdrant_backfill.database,
        "list_unknown_samples_with_embeddings",
        lambda: [],
    )
    monkeypatch.setattr(
        qdrant_backfill.database,
        "list_unknown_voice_samples_with_embeddings",
        lambda: [],
    )
    prepared = qdrant_backfill.prepare_known_embedding_backfill()
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(client)
    store.ensure_collection(config.QDRANT_FACE_COLLECTION, config.EMBEDDING_DIM)
    store.ensure_collection(config.QDRANT_VOICE_COLLECTION, config.VOICE_EMBEDDING_DIM)

    with pytest.raises(ValueError, match="Refusing incomplete backfill"):
        qdrant_backfill.apply_known_embedding_backfill(store, prepared)

    assert prepared.face_skips == ((18, "file not found"),)
    assert client.count(config.QDRANT_FACE_COLLECTION).count == 0
    client.close()


def test_backfill_includes_unknown_face_and_voice_samples(tmp_path, monkeypatch):
    face_path = tmp_path / "unknown-face.npy"
    voice_path = tmp_path / "unknown-voice.npy"
    np.save(face_path, np.ones(config.EMBEDDING_DIM, dtype=np.float32))
    np.save(voice_path, np.ones(config.VOICE_EMBEDDING_DIM, dtype=np.float32))
    monkeypatch.setattr(qdrant_backfill.database, "list_embeddings", lambda: [])
    monkeypatch.setattr(qdrant_backfill.database, "list_voice_embeddings", lambda: [])
    monkeypatch.setattr(
        qdrant_backfill.database,
        "list_unknown_samples_with_embeddings",
        lambda: [(31, 9, str(face_path), 0.7)],
    )
    monkeypatch.setattr(
        qdrant_backfill.database,
        "list_unknown_voice_samples_with_embeddings",
        lambda: [(41, 11, str(voice_path), 0.8)],
    )

    prepared = qdrant_backfill.prepare_known_embedding_backfill()
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(client)
    store.ensure_collection(config.QDRANT_FACE_COLLECTION, config.EMBEDDING_DIM)
    store.ensure_collection(config.QDRANT_VOICE_COLLECTION, config.VOICE_EMBEDDING_DIM)

    qdrant_backfill.apply_known_embedding_backfill(store, prepared)

    face_points = store.search(
        config.QDRANT_FACE_COLLECTION,
        np.ones(config.EMBEDDING_DIM, dtype=np.float32),
    )
    voice_points = store.search(
        config.QDRANT_VOICE_COLLECTION,
        np.ones(config.VOICE_EMBEDDING_DIM, dtype=np.float32),
    )
    assert face_points[0].payload["record_type"] == "unknown_sample"
    assert face_points[0].payload["unknown_id"] == 9
    assert voice_points[0].payload["record_type"] == "unknown_sample"
    assert voice_points[0].payload["unknown_id"] == 11
    client.close()