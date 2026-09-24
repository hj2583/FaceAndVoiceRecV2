import numpy as np

import config
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


def test_diarize_meeting_audio_clusters_segments_by_similarity(tmp_path, monkeypatch):
    speaker_a = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    speaker_a[0] = 1.0
    speaker_b = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    speaker_b[1] = 1.0

    embeddings_by_region = {
        (0.0, 1.0): speaker_a,
        (1.0, 2.0): speaker_b,
        (2.0, 3.0): speaker_a,
    }

    def fake_extract(_pcm, sample_rate=16000):
        return None  # overridden per-call below

    calls = iter(embeddings_by_region.values())
    monkeypatch.setattr(voice_core, "extract_voice_embedding", lambda *_a, **_k: next(calls))
    monkeypatch.setattr(voice_core, "read_wav_pcm", lambda *_a, **_k: b"\x00\x00" * 8000)
    monkeypatch.setattr(voice_core, "match_voice_embedding", lambda *_a, **_k: None)

    result = voice_core.diarize_meeting_audio(
        tmp_path / "audio.wav",
        list(embeddings_by_region.keys()),
    )

    labels = [label for label, _start, _end in result]
    assert labels[0] == labels[2]
    assert labels[0] != labels[1]
    assert all(label.startswith("Unknown Speaker") for label in labels)


def test_diarize_meeting_audio_reads_wav_file_only_once(tmp_path, monkeypatch):
    speaker_a = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    speaker_a[0] = 1.0
    speaker_b = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    speaker_b[1] = 1.0

    regions = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
    calls = iter([speaker_a, speaker_b, speaker_a, speaker_b])

    read_wav_pcm_call_count = 0

    def fake_read_wav_pcm(_wav_path):
        nonlocal read_wav_pcm_call_count
        read_wav_pcm_call_count += 1
        return b"\x00\x00" * (4 * 16000)

    monkeypatch.setattr(voice_core, "extract_voice_embedding", lambda *_a, **_k: next(calls))
    monkeypatch.setattr(voice_core, "read_wav_pcm", fake_read_wav_pcm)
    monkeypatch.setattr(voice_core, "match_voice_embedding", lambda *_a, **_k: None)

    result = voice_core.diarize_meeting_audio(tmp_path / "audio.wav", regions)

    assert len(result) == len(regions)
    assert read_wav_pcm_call_count == 1


def test_diarize_meeting_audio_uses_enrolled_person_name(tmp_path, monkeypatch):
    embedding = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    embedding[0] = 1.0

    monkeypatch.setattr(voice_core, "extract_voice_embedding", lambda *_a, **_k: embedding)
    monkeypatch.setattr(voice_core, "read_wav_pcm", lambda *_a, **_k: b"\x00\x00" * 8000)
    monkeypatch.setattr(
        voice_core,
        "match_voice_embedding",
        lambda *_a, **_k: {"person_id": 7, "person_name": "Frank", "similarity": 0.9},
    )

    result = voice_core.diarize_meeting_audio(tmp_path / "audio.wav", [(0.0, 1.0)])

    assert result == [("Frank", 0.0, 1.0)]


def test_find_matching_unknown_voice_and_register(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "faces.db")
    monkeypatch.setattr(config, "UNKNOWN_VOICES_DIR", tmp_path)
    database.init_db()

    embedding = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    embedding[0] = 1.0

    unknown_voice_id = voice_core.register_unknown_voice("Speaker 1", embedding)

    match = voice_core.find_matching_unknown_voice(embedding)
    assert match is not None
    assert match["unknown_voice_id"] == unknown_voice_id
    assert match["similarity"] > 0.99


def test_match_voice_embedding_returns_none_when_no_voiceprints(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    query = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    query[0] = 1.0
    assert voice_core.match_voice_embedding(query) is None
