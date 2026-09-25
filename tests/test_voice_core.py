import numpy as np
from pathlib import Path

import config
import database
import voice_core


def _setup_database(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "faces.db")
    database.init_db()


def _unit(index):
    vector = np.zeros(voice_core.EMBEDDING_DIM, dtype=np.float32)
    vector[index] = 1.0
    return vector


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
    _setup_database(tmp_path, monkeypatch)
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
    _setup_database(tmp_path, monkeypatch)
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


def test_has_enrolled_voiceprints_reflects_voice_embeddings_table(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    assert voice_core.has_enrolled_voiceprints() is False

    person_id = database.create_person("Jill")
    embedding_path = tmp_path / "jill.npy"
    np.save(embedding_path, _unit(0))
    database.add_voice_embedding(person_id, embedding_path, quality=1.0)

    assert voice_core.has_enrolled_voiceprints() is True


def test_register_new_unknown_voice_labels_and_names_file_by_id(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)

    first_id, first_label = voice_core.register_new_unknown_voice(_unit(0))
    second_id, second_label = voice_core.register_new_unknown_voice(_unit(1))

    assert first_label == f"Unknown Speaker {first_id}"
    assert second_label == f"Unknown Speaker {second_id}"
    assert first_label != second_label
    rows = {row[0]: row for row in database.list_unknown_voices()}
    assert rows[first_id][1] == first_label
    embedding_path = Path(rows[first_id][2])
    assert embedding_path.exists()
    assert embedding_path.name.startswith(f"unknown_voice_{first_id}_")
    assert len(database.list_unknown_voice_samples(first_id)) == 1


def test_sliding_windows_keeps_short_region_whole_and_covers_long_region_tail():
    assert voice_core._sliding_windows(0.0, 1.0, 1.5, 0.75) == [(0.0, 1.0)]
    assert voice_core._sliding_windows(0.0, 3.0, 1.5, 0.75) == [
        (0.0, 1.5),
        (0.75, 2.25),
        (1.5, 3.0),
    ]
    assert voice_core._sliding_windows(0.0, 2.0, 1.5, 0.75) == [(0.0, 1.5), (0.5, 2.0)]


def test_diarize_meeting_audio_splits_speaker_change_inside_one_region(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "VOICE_DIARIZATION_WINDOW_SECONDS", 1.5)
    monkeypatch.setattr(config, "VOICE_DIARIZATION_STEP_SECONDS", 0.75)
    embeddings = iter([_unit(0), _unit(0), _unit(1)])
    monkeypatch.setattr(voice_core, "extract_voice_embedding", lambda *_a, **_k: next(embeddings))
    monkeypatch.setattr(voice_core, "read_wav_pcm", lambda *_a, **_k: b"\x00\x00" * (16000 * 3))

    result = voice_core.diarize_meeting_audio(tmp_path / "audio.wav", [(0.0, 3.0)])

    assert [(start, end) for _label, start, end in result] == [
        (0.0, 1.5),
        (0.75, 2.25),
        (1.5, 3.0),
    ]
    labels = [label for label, _start, _end in result]
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]


def test_diarize_meeting_audio_keeps_distinct_unknowns_apart_within_one_call(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    # e = sqrt(.4)*common + sqrt(.2)*speaker + sqrt(.4)*noise_i: intra-cluster
    # similarity 0.6, cross-cluster 0.4 (clustered apart), but the normalized
    # centroids of 4 windows each have similarity 0.4/0.7 ~= 0.57 >= match threshold.
    def window(speaker_dim, noise_dim):
        return (
            np.sqrt(0.4) * _unit(0) + np.sqrt(0.2) * _unit(speaker_dim) + np.sqrt(0.4) * _unit(noise_dim)
        ).astype(np.float32)

    embeddings = [window(1, 3 + i) for i in range(4)] + [window(2, 7 + i) for i in range(4)]
    calls = iter(embeddings)
    monkeypatch.setattr(voice_core, "extract_voice_embedding", lambda *_a, **_k: next(calls))
    monkeypatch.setattr(voice_core, "read_wav_pcm", lambda *_a, **_k: b"\x00\x00" * (16000 * 8))

    result = voice_core.diarize_meeting_audio(
        tmp_path / "audio.wav", [(float(i), float(i + 1)) for i in range(8)]
    )

    labels = [label for label, _start, _end in result]
    assert len(set(labels[:4])) == 1
    assert len(set(labels[4:])) == 1
    assert labels[0] != labels[4]
    assert len(database.list_unknown_voices()) == 2


def _insert_meeting(video_path):
    with database.get_conn() as conn:
        conn.execute(
            "INSERT INTO meetings(video_path, created_at, transcription_status) VALUES (?, ?, 'completed')",
            (video_path, database.utc_now()),
        )
        return conn.execute(
            "SELECT meeting_id FROM meetings WHERE video_path=?",
            (video_path,),
        ).fetchone()[0]


def test_unknown_voice_lifecycle_is_scoped_by_globally_unique_labels(tmp_path, monkeypatch):
    import transcription_core

    db_path = tmp_path / "faces.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    # save_transcripts writes segments through config.DB_PATH.
    monkeypatch.setattr(config, "DB_PATH", db_path)
    database.init_db()
    monkeypatch.setattr(voice_core, "read_wav_pcm", lambda *_a, **_k: b"\x00\x00" * (16000 * 2))
    voice_x, voice_y = _unit(0), _unit(1)

    def diarize(embeddings, regions):
        calls = iter(embeddings)
        monkeypatch.setattr(voice_core, "extract_voice_embedding", lambda *_a, **_k: next(calls))
        return voice_core.diarize_meeting_audio(tmp_path / "audio.wav", regions)

    def save(meeting_id, diarization, texts):
        segments = [
            transcription_core.TranscriptionSegment(label, int(start * 1000), int(end * 1000), text)
            for (label, start, end), text in zip(diarization, texts)
        ]
        transcription_core.save_transcripts(
            segments, config.TRANSCRIPTS_DIR / str(meeting_id), meeting_id=meeting_id
        )

    # Meeting 1: voice X is unmatched -> a new unknown voice is registered.
    meeting_1 = _insert_meeting("m1.mp4")
    diarization_1 = diarize([voice_x], [(0.0, 1.0)])
    (x_id, x_label, _path, _created, _resolved), = database.list_unknown_voices()
    assert x_label == f"Unknown Speaker {x_id}"
    assert diarization_1 == [(x_label, 0.0, 1.0)]
    assert len(database.list_unknown_voice_samples(x_id)) == 1
    save(meeting_1, diarization_1, ["x in meeting one"])

    # Meeting 2: new voice Y comes first (per-meeting numbering would have
    # reused meeting 1's label for it), then X again.
    meeting_2 = _insert_meeting("m2.mp4")
    diarization_2 = diarize([voice_y, voice_x], [(0.0, 1.0), (1.0, 2.0)])
    unknowns = database.list_unknown_voices()
    assert len(unknowns) == 2
    y_id, y_label = next((row[0], row[1]) for row in unknowns if row[0] != x_id)
    assert y_label == f"Unknown Speaker {y_id}"
    assert y_label != x_label
    assert diarization_2 == [(y_label, 0.0, 1.0), (x_label, 1.0, 2.0)]
    assert len(database.list_unknown_voice_samples(x_id)) == 2
    assert len(database.list_unknown_voice_samples(y_id)) == 1
    save(meeting_2, diarization_2, ["y in meeting two", "x in meeting two"])

    person_id = database.create_person("Zed")
    database.resolve_unknown_voice(y_id, person_id)

    with database.get_conn() as conn:
        rows = conn.execute(
            "SELECT meeting_id, speaker_label, person_id FROM transcription_segments "
            "ORDER BY meeting_id, start_ms"
        ).fetchall()
    assert rows == [
        (meeting_1, x_label, None),
        (meeting_2, "Zed", person_id),
        (meeting_2, x_label, None),
    ]
    assert [row[0] for row in database.list_unknown_voices()] == [x_id]

    meeting_2_dir = config.TRANSCRIPTS_DIR / str(meeting_2)
    assert sorted(path.name for path in meeting_2_dir.glob("*.txt")) == sorted(
        ["zed.txt", f"unknown_speaker_{x_id}.txt"]
    )
    assert (meeting_2_dir / "zed.txt").read_text(encoding="utf-8") == "[00:00] y in meeting two\n"
    meeting_1_dir = config.TRANSCRIPTS_DIR / str(meeting_1)
    assert [path.name for path in meeting_1_dir.glob("*.txt")] == [f"unknown_speaker_{x_id}.txt"]
