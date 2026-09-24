from pathlib import Path

import database


def _setup_database(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "faces.db")
    database.init_db()


def test_add_and_list_voice_embeddings(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alice")
    embedding_path = tmp_path / "alice.npy"
    embedding_path.write_bytes(b"embedding")

    embedding_id = database.add_voice_embedding(person_id, embedding_path, quality=0.9)

    rows = database.list_voice_embeddings()
    assert rows == [(embedding_id, person_id, "Alice", str(embedding_path), 0.9)]


def test_create_unknown_voice_and_add_samples(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    embedding_path = tmp_path / "cluster.npy"
    embedding_path.write_bytes(b"embedding")

    unknown_voice_id = database.create_unknown_voice("Speaker 1", embedding_path)
    sample_path = tmp_path / "sample.npy"
    sample_path.write_bytes(b"sample")
    database.add_unknown_voice_sample(unknown_voice_id, sample_path, quality=0.5)

    samples = database.list_unknown_voice_samples(unknown_voice_id)
    assert len(samples) == 1
    assert samples[0][1] == str(sample_path)

    unresolved = database.list_unknown_voices()
    assert unresolved[0][0] == unknown_voice_id
    assert unresolved[0][4] is None


def test_resolve_unknown_voice_sets_resolved_person_id(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Bob")
    embedding_path = tmp_path / "cluster.npy"
    embedding_path.write_bytes(b"embedding")
    unknown_voice_id = database.create_unknown_voice("Speaker 1", embedding_path)

    database.resolve_unknown_voice(unknown_voice_id, person_id)

    rows = database.list_unknown_voices(include_resolved=True)
    assert rows[0][4] == person_id
    assert database.list_unknown_voices() == []


def test_delete_unknown_voice_removes_record_samples_and_files(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    embedding_path = tmp_path / "cluster.npy"
    embedding_path.write_bytes(b"embedding")
    unknown_voice_id = database.create_unknown_voice("Speaker 1", embedding_path)
    sample_path = tmp_path / "sample.npy"
    sample_path.write_bytes(b"sample")
    database.add_unknown_voice_sample(unknown_voice_id, sample_path)

    assert database.delete_unknown_voice(unknown_voice_id) is True

    assert not embedding_path.exists()
    assert not sample_path.exists()
    assert database.list_unknown_voice_samples(unknown_voice_id) == []
    assert database.list_unknown_voices(include_resolved=True) == []


def test_delete_unknown_voice_refuses_resolved_records(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Carol")
    embedding_path = tmp_path / "cluster.npy"
    embedding_path.write_bytes(b"embedding")
    unknown_voice_id = database.create_unknown_voice("Speaker 1", embedding_path)
    database.resolve_unknown_voice(unknown_voice_id, person_id)

    assert database.delete_unknown_voice(unknown_voice_id) is False
    assert embedding_path.exists()
