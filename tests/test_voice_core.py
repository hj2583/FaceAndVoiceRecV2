import numpy as np

import database
import voice_core


def _setup_database(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "faces.db")
    database.init_db()


def _fake_model(vector):
    class FakeModel:
        def encode_batch(self, tensor):
            import torch
            return torch.from_numpy(np.asarray([vector], dtype=np.float32)).unsqueeze(0)
    return FakeModel()


def test_extract_voice_embedding_normalizes_and_matches_dimension(monkeypatch):
    raw_vector = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    raw_vector[3] = 5.0  # not unit-norm on purpose; extract_voice_embedding must normalize it
    monkeypatch.setattr(voice_core, "_load_model", lambda: _fake_model(raw_vector))

    pcm = (b"\x00\x01" * 8000)  # 16000 samples of fake PCM16 audio, above _MIN_SAMPLES

    embedding = voice_core.extract_voice_embedding(pcm)

    assert embedding is not None
    assert embedding.shape == (voice_core.EMBEDDING_DIM,)
    assert abs(float(np.linalg.norm(embedding)) - 1.0) < 1e-5


def test_extract_voice_embedding_returns_none_for_empty_audio(monkeypatch):
    assert voice_core.extract_voice_embedding(b"") is None


def test_match_voice_embedding_returns_best_person_above_threshold(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Dana")
    embedding = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    embedding[0] = 1.0
    embedding_path = tmp_path / "dana.npy"
    np.save(embedding_path, embedding)
    database.add_voice_embedding(person_id, embedding_path, quality=1.0)

    query = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    query[0] = 1.0

    match = voice_core.match_voice_embedding(query)

    assert match is not None
    assert match["person_id"] == person_id
    assert match["person_name"] == "Dana"
    assert match["similarity"] > 0.99


def test_match_voice_embedding_returns_none_below_threshold(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Eve")
    embedding = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    embedding[0] = 1.0
    embedding_path = tmp_path / "eve.npy"
    np.save(embedding_path, embedding)
    database.add_voice_embedding(person_id, embedding_path, quality=1.0)

    query = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    query[-1] = 1.0  # orthogonal -> similarity 0.0

    assert voice_core.match_voice_embedding(query) is None


def test_match_voice_embedding_returns_none_when_no_voiceprints(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    query = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    query[0] = 1.0
    assert voice_core.match_voice_embedding(query) is None
