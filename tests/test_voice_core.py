import json
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


def _speech_pcm(seconds):
    # Non-silent deterministic PCM for tests that exercise quality gates.
    return (b"\x00\x10" * (16000 * seconds))


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


def test_reprocessed_diarization_uses_reassigned_source_identity(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "DB_PATH", database.DB_PATH)
    transcripts_dir = tmp_path / "transcripts"
    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", transcripts_dir)
    previous_person_id = database.create_person("Person 1")
    replacement_person_id = database.create_person("The Centre 1")
    audio_dir = transcripts_dir / "1"
    audio_dir.mkdir(parents=True)
    audio_path = audio_dir / "audio.wav"
    audio_path.write_bytes(b"audio")
    with database.get_conn() as conn:
        conn.execute(
            """
            INSERT INTO meetings(video_path, created_at, transcription_status)
            VALUES ('meeting.mp4', ?, 'completed')
            """,
            (database.utc_now(),),
        )

    embedding_path = tmp_path / "sample-3.npy"
    cluster_path = tmp_path / "cluster.npy"
    np.save(embedding_path, _unit(0))
    np.save(cluster_path, _unit(0))
    unknown_voice_id = database.create_unknown_voice(
        "Unknown Speaker 2", cluster_path
    )
    database.add_unknown_voice_sample(
        unknown_voice_id,
        embedding_path,
        quality=0.9,
        audio_path=audio_path,
        source_ref=audio_path,
        start_ms=17_328,
        end_ms=18_828,
    )
    database.resolve_unknown_voice(unknown_voice_id, previous_person_id)
    embedding_id = next(
        row[0]
        for row in database.list_voice_enrollments()
        if row[3] == str(embedding_path)
    )

    assert database.reassign_voice_embedding(embedding_id, replacement_person_id)
    monkeypatch.setattr(
        voice_core,
        "read_wav_pcm",
        lambda *_args, **_kwargs: _speech_pcm(25),
    )
    monkeypatch.setattr(
        voice_core,
        "extract_voice_embedding",
        lambda *_args, **_kwargs: _unit(0),
    )

    diarized = voice_core.diarize_meeting_audio(
        audio_path,
        [(17.328, 18.828)],
    )
    reprocessed = voice_core.diarize_meeting_audio(
        audio_path,
        [(17.328, 18.828)],
    )

    assert diarized == [("The Centre 1", 17.328, 18.828)]
    assert reprocessed == diarized
    assert database.list_unknown_voices() == []
    resolved = database.list_unknown_voices(include_resolved=True)
    assert len(resolved) == 1
    assert resolved[0][4] == replacement_person_id


def test_voice_match_rejects_close_candidate_profiles(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    query = _unit(0)
    for name in ("Centre 3", "Centre 5"):
        person_id = database.create_person(name)
        for suffix, embedding in (
            ("a", query),
            ("b", (0.99 * query + 0.01 * _unit(1)).astype(np.float32)),
        ):
            path = tmp_path / f"{name.replace(' ', '_')}_{suffix}.npy"
            np.save(path, embedding)
            database.add_voice_embedding(person_id, path, quality=0.9)

    diagnostic = voice_core.diagnose_voice_embedding_match(query)

    assert diagnostic["decision"] is None
    assert diagnostic["reason"] == "ambiguous_candidate_margin"
    assert [candidate["person_name"] for candidate in diagnostic["candidates"]] == [
        "Centre 3",
        "Centre 5",
    ]
    assert diagnostic["margin"] == 0.0


def test_duplicate_enrollment_embeddings_count_as_one_reference(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Centre 3")
    for suffix in ("first", "duplicate"):
        path = tmp_path / f"{suffix}.npy"
        np.save(path, _unit(0))
        database.add_voice_embedding(person_id, path, quality=0.9)

    diagnostic = voice_core.diagnose_voice_embedding_match(_unit(0))

    assert diagnostic["decision"] is not None
    assert diagnostic["decision"]["usable_samples"] == 1
    assert diagnostic["decision"]["supporting_samples"] == 1


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
    monkeypatch.setattr(voice_core, "read_wav_pcm", lambda *_a, **_k: _speech_pcm(3))
    monkeypatch.setattr(voice_core, "match_voice_embedding", lambda *_a, **_k: None)

    result = voice_core.diarize_meeting_audio(
        tmp_path / "audio.wav",
        list(embeddings_by_region.keys()),
    )

    labels = [label for label, _start, _end in result]
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
        return _speech_pcm(4)

    monkeypatch.setattr(voice_core, "extract_voice_embedding", lambda *_a, **_k: next(calls))
    monkeypatch.setattr(voice_core, "read_wav_pcm", fake_read_wav_pcm)
    monkeypatch.setattr(voice_core, "match_voice_embedding", lambda *_a, **_k: None)

    result = voice_core.diarize_meeting_audio(tmp_path / "audio.wav", regions)

    assert len(result) == len(regions)
    assert read_wav_pcm_call_count == 1


def test_diarize_persists_playable_unknown_sample_once_per_source_window(
    tmp_path, monkeypatch
):
    _setup_database(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "UNKNOWN_VOICES_DIR", tmp_path / "unknown_voices")
    monkeypatch.setattr(voice_core, "read_wav_pcm", lambda *_a, **_k: _speech_pcm(1))
    monkeypatch.setattr(voice_core, "extract_voice_embedding", lambda *_a, **_k: _unit(0))
    monkeypatch.setattr(voice_core, "match_voice_embedding", lambda *_a, **_k: None)
    wav_path = tmp_path / "meeting" / "audio.wav"

    first_result = voice_core.diarize_meeting_audio(wav_path, [(0.0, 1.0)])
    unknown_voice_id = database.list_unknown_voices()[0][0]
    first_samples = database.list_unknown_voice_samples(unknown_voice_id)
    playable_samples = [row for row in first_samples if row[4]]

    assert first_result[0][0] == f"Unknown Speaker {unknown_voice_id}"
    assert len(playable_samples) == 1
    audio_path = Path(playable_samples[0][4])
    assert audio_path.is_file()
    with open(audio_path, "rb") as audio_file:
        import wave

        with wave.open(audio_file, "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getframerate() == 16000
            assert wav.getnframes() > 0

    second_result = voice_core.diarize_meeting_audio(wav_path, [(0.0, 1.0)])

    assert second_result == first_result
    assert len(database.list_unknown_voices()) == 1
    assert len(database.list_unknown_voice_samples(unknown_voice_id)) == len(first_samples)


def test_reprocessing_resolved_source_reuses_confirmed_voice_identity(
    tmp_path, monkeypatch
):
    _setup_database(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "UNKNOWN_VOICES_DIR", tmp_path / "unknown_voices")
    monkeypatch.setattr(voice_core, "read_wav_pcm", lambda *_a, **_k: _speech_pcm(1))
    monkeypatch.setattr(voice_core, "extract_voice_embedding", lambda *_a, **_k: _unit(0))
    monkeypatch.setattr(voice_core, "match_voice_embedding", lambda *_a, **_k: None)
    wav_path = tmp_path / "transcripts" / "meeting_1" / "audio.wav"

    first_result = voice_core.diarize_meeting_audio(wav_path, [(0.0, 1.0)])
    unknown_voice_id = database.list_unknown_voices()[0][0]
    person_id = database.create_person("Morgan")
    database.resolve_unknown_voice(unknown_voice_id, person_id=person_id)

    repeated_result = voice_core.diarize_meeting_audio(wav_path, [(0.0, 1.0)])

    assert first_result == [(f"Unknown Speaker {unknown_voice_id}", 0.0, 1.0)]
    assert repeated_result == [("Morgan", 0.0, 1.0)]
    assert database.list_unknown_voices() == []


def test_diarize_uses_confident_known_face_identity_without_registering_unknown(
    tmp_path, monkeypatch
):
    _setup_database(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "UNKNOWN_VOICES_DIR", tmp_path / "unknown_voices")
    monkeypatch.setattr(voice_core, "read_wav_pcm", lambda *_a, **_k: _speech_pcm(1))
    monkeypatch.setattr(voice_core, "extract_voice_embedding", lambda *_a, **_k: _unit(0))
    monkeypatch.setattr(voice_core, "match_voice_embedding", lambda *_a, **_k: None)

    result = voice_core.diarize_meeting_audio(
        tmp_path / "audio.wav",
        [(0.0, 1.0)],
        known_speaker_events=[{
            "person_id": 5,
            "speaker": "Alice",
            "start_time": 0.0,
            "end_time": 1.0,
        }],
    )

    assert result == [("Alice", 0.0, 1.0)]
    assert database.list_unknown_voices() == []


def test_diarize_meeting_audio_uses_enrolled_person_name(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    embedding = _unit(0)
    person_id = database.create_person("Frank")
    embedding_path = tmp_path / "frank.npy"
    np.save(embedding_path, embedding)
    database.add_voice_embedding(person_id, embedding_path, quality=1.0)
    monkeypatch.setattr(config, "VOICE_MIN_IDENTITY_DURATION_SECONDS", 0.0)
    monkeypatch.setattr(voice_core, "extract_voice_embedding", lambda *_a, **_k: embedding)
    monkeypatch.setattr(voice_core, "read_wav_pcm", lambda *_a, **_k: _speech_pcm(1))

    result = voice_core.diarize_meeting_audio(tmp_path / "audio.wav", [(0.0, 1.0)])

    assert result == [("Frank", 0.0, 1.0)]


def test_short_speech_cluster_does_not_receive_known_identity(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "UNKNOWN_VOICES_DIR", tmp_path / "unknown_voices")
    monkeypatch.setattr(config, "VOICE_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(config, "VOICE_DIAGNOSTICS_PATH", tmp_path / "diagnostics.json")
    person_id = database.create_person("Frank")
    path = tmp_path / "frank.npy"
    np.save(path, _unit(0))
    database.add_voice_embedding(person_id, path, quality=1.0)
    monkeypatch.setattr(voice_core, "read_wav_pcm", lambda *_a, **_k: _speech_pcm(1))
    monkeypatch.setattr(voice_core, "extract_voice_embedding", lambda *_a, **_k: _unit(0))

    result = voice_core.diarize_meeting_audio(tmp_path / "audio.wav", [(0.0, 1.0)])

    assert result[0][0].startswith("Unknown Speaker ")
    report = json.loads((tmp_path / "diagnostics.json").read_text(encoding="utf-8"))
    assert report["summary"]["speaker_decisions"][0]["decision_reason"] == (
        "insufficient_speech_duration"
    )


def test_voice_face_conflict_remains_unresolved(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "UNKNOWN_VOICES_DIR", tmp_path / "unknown_voices")
    monkeypatch.setattr(config, "VOICE_MIN_IDENTITY_DURATION_SECONDS", 0.0)
    monkeypatch.setattr(config, "VOICE_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(config, "VOICE_DIAGNOSTICS_PATH", tmp_path / "diagnostics.json")
    person_id = database.create_person("Centre 3")
    path = tmp_path / "centre3.npy"
    np.save(path, _unit(0))
    database.add_voice_embedding(person_id, path, quality=1.0)
    monkeypatch.setattr(voice_core, "read_wav_pcm", lambda *_a, **_k: _speech_pcm(1))
    monkeypatch.setattr(voice_core, "extract_voice_embedding", lambda *_a, **_k: _unit(0))

    result = voice_core.diarize_meeting_audio(
        tmp_path / "audio.wav",
        [(0.0, 1.0)],
        known_speaker_events=[{
            "person_id": 32,
            "speaker": "Centre 5",
            "start_time": 0.0,
            "end_time": 1.0,
        }],
    )

    assert result[0][0].startswith("Unknown Speaker ")
    report = json.loads((tmp_path / "diagnostics.json").read_text(encoding="utf-8"))
    decision = report["summary"]["speaker_decisions"][0]
    assert decision["decision_reason"] == "audio_video_identity_conflict"
    assert decision["face_candidate"]["person_id"] == 32
    assert decision["voice_match"]["decision"]["person_id"] == person_id
    assert decision["voice_match"]["threshold"] == config.VOICE_MATCH_THRESHOLD
    assert "audio_path" not in report["summary"]


def test_diarize_meeting_audio_writes_quality_diagnostics(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "VOICE_DIAGNOSTICS_ENABLED", True)
    monkeypatch.setattr(config, "VOICE_DIAGNOSTICS_PATH", tmp_path / "diagnostics.json")
    monkeypatch.setattr(voice_core, "read_wav_pcm", lambda *_a, **_k: _speech_pcm(2))
    monkeypatch.setattr(voice_core, "extract_voice_embedding", lambda *_a, **_k: _unit(0))
    monkeypatch.setattr(voice_core, "match_voice_embedding", lambda *_a, **_k: None)

    voice_core.diarize_meeting_audio(
        tmp_path / "audio.wav",
        [(0.0, 1.0), (1.0, 2.0)],
    )

    report = json.loads((tmp_path / "diagnostics.json").read_text(encoding="utf-8"))
    assert report["summary"]["candidate_windows"] == 2
    assert report["summary"]["accepted_embeddings"] == 2
    assert all(window["accepted"] for window in report["windows"])


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
    monkeypatch.setattr(voice_core, "read_wav_pcm", lambda *_a, **_k: _speech_pcm(3))

    result = voice_core.diarize_meeting_audio(tmp_path / "audio.wav", [(0.0, 3.0)])

    assert [(start, end) for _label, start, end in result] == [
        (0.0, 1.5),
        (1.5, 3.0),
    ]
    labels = [label for label, _start, _end in result]
    assert labels[0] != labels[1]


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
    monkeypatch.setattr(voice_core, "read_wav_pcm", lambda *_a, **_k: _speech_pcm(8))

    result = voice_core.diarize_meeting_audio(
        tmp_path / "audio.wav", [(float(i), float(i + 1)) for i in range(8)]
    )

    labels = [label for label, _start, _end in result]
    assert len(labels) == 2
    assert labels[0] != labels[1]
    assert result[0][2] <= result[1][1]
    assert len(database.list_unknown_voices()) == 2


def test_diarize_meeting_audio_reuses_unknown_identity_for_split_same_voice(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "VOICE_CLUSTER_DISTANCE_THRESHOLD", 0.1)
    first_embedding = _unit(0)
    second_embedding = (0.8 * _unit(0) + 0.6 * _unit(1)).astype(np.float32)
    embeddings = iter([first_embedding, second_embedding])
    monkeypatch.setattr(voice_core, "extract_voice_embedding", lambda *_a, **_k: next(embeddings))
    monkeypatch.setattr(voice_core, "read_wav_pcm", lambda *_a, **_k: _speech_pcm(2))
    monkeypatch.setattr(voice_core, "match_voice_embedding", lambda *_a, **_k: None)

    result = voice_core.diarize_meeting_audio(
        tmp_path / "audio.wav",
        [(0.0, 1.0), (1.0, 2.0)],
    )

    assert len(database.list_unknown_voices()) == 1
    samples = database.list_unknown_voice_samples(database.list_unknown_voices()[0][0])
    assert len(samples) == 1
    assert Path(samples[0][4]).is_file()
    assert {label for label, _start, _end in result} == {
        database.list_unknown_voices()[0][1]
    }


def test_diarize_prefers_current_meeting_identity_over_duplicate_old_unknowns(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "faces.db")
    monkeypatch.setattr(config, "UNKNOWN_VOICES_DIR", tmp_path / "unknown")
    database.init_db()
    monkeypatch.setattr(config, "VOICE_CLUSTER_DISTANCE_THRESHOLD", 0.1)
    first_embedding = _unit(0)
    second_embedding = (0.8 * _unit(0) + 0.6 * _unit(1)).astype(np.float32)
    voice_core.register_unknown_voice("Unknown Speaker 1", first_embedding)
    voice_core.register_unknown_voice("Unknown Speaker 2", second_embedding)
    embeddings = iter([first_embedding, second_embedding])
    monkeypatch.setattr(voice_core, "extract_voice_embedding", lambda *_a, **_k: next(embeddings))
    monkeypatch.setattr(voice_core, "read_wav_pcm", lambda *_a, **_k: _speech_pcm(2))
    monkeypatch.setattr(voice_core, "match_voice_embedding", lambda *_a, **_k: None)

    result = voice_core.diarize_meeting_audio(
        tmp_path / "audio.wav",
        [(0.0, 1.0), (1.0, 2.0)],
    )

    assert len({label for label, _start, _end in result}) == 1


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
    monkeypatch.setattr(voice_core, "read_wav_pcm", lambda *_a, **_k: _speech_pcm(2))
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
