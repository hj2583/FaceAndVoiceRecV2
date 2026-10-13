import numpy as np

from face_voice_rec import config
from face_voice_rec.storage import database
from face_voice_rec.domain.transcription.speaker_introductions import (
    extract_self_introductions,
    process_meeting_introductions,
)


def _setup_database(tmp_path, monkeypatch):
    db_path = tmp_path / "faces.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    database.init_db()
    with database.get_conn() as conn:
        conn.execute(
            """
            INSERT INTO meetings(video_path, created_at, transcription_status)
            VALUES ('meeting.mp4', ?, 'completed')
            """,
            (database.utc_now(),),
        )
        return int(
            conn.execute(
                "SELECT meeting_id FROM meetings WHERE video_path='meeting.mp4'"
            ).fetchone()[0]
        )


def _add_introduction_segment(meeting_id, speaker, text):
    with database.get_conn() as conn:
        return int(
            conn.execute(
                """
                INSERT INTO transcription_segments(
                    meeting_id, speaker_label, start_ms, end_ms, text, created_at
                ) VALUES (?, ?, 27000, 29000, ?, ?)
                """,
                (meeting_id, speaker, text, database.utc_now()),
            ).lastrowid
        )


def test_extracts_common_and_multilingual_self_introductions():
    cases = (
        ("Hi, I'm Alex and I'm Head of Information.", "Alex", "Head of Information"),
        ("Hi, I'm Rachel and I work in administration.", "Rachel", "administration"),
        ("I'm John and I am Head of Finance.", "John", "Head of Finance"),
        ("I'm Matthew, Head of Research.", "Matthew", "Head of Research"),
        ("My name is Sarah.", "Sarah", None),
        ("You can call me David.", "David", None),
        ("Nama saya Aisyah dan saya bekerja sebagai pengurus.", "Aisyah", "pengurus"),
        ("Me llamo Ana y soy directora.", "Ana", "directora"),
        ("Je m'appelle Jean et je suis ingénieur.", "Jean", "ingénieur"),
    )

    for text, expected_name, expected_title in cases:
        results = extract_self_introductions(text)
        assert len(results) == 1
        assert results[0].name == expected_name
        assert results[0].job_title == expected_title
        assert 0.0 < results[0].confidence <= 1.0


def test_enrolls_unknown_voice_from_explicit_introduction_and_keeps_face_pending(
    tmp_path, monkeypatch
):
    meeting_id = _setup_database(tmp_path, monkeypatch)
    cluster_path = tmp_path / "cluster.npy"
    sample_path = tmp_path / "sample.npy"
    cluster_path.write_bytes(b"cluster")
    sample_path.write_bytes(b"sample")
    unknown_voice_id = database.create_unknown_voice(
        "Unknown Speaker 17", cluster_path
    )
    database.add_unknown_voice_sample(
        unknown_voice_id,
        sample_path,
        quality=0.9,
        source_ref="meeting.mp4",
        start_ms=27_000,
        end_ms=29_000,
    )
    segment_id = _add_introduction_segment(
        meeting_id,
        "Unknown Speaker 17",
        "Hi, I'm Alex and I'm Head of Information.",
    )

    updates = process_meeting_introductions(
        meeting_id,
        [{
            "start_time": 27.0,
            "end_time": 29.0,
            "track_id": 5,
            "person_id": None,
            "activity_score": 0.9,
        }],
    )

    person = next(row for row in database.list_persons() if row[1] == "Alex")
    introduction = database.list_meeting_introductions(meeting_id)[0]
    with database.get_conn() as conn:
        transcript = conn.execute(
            """
            SELECT speaker_label, person_id FROM transcription_segments
            WHERE meeting_id=? AND segment_id=?
            """,
            (meeting_id, segment_id),
        ).fetchone()
        enrollment_count = conn.execute(
            "SELECT COUNT(*) FROM voice_embeddings WHERE person_id=?",
            (person[0],),
        ).fetchone()[0]
        resolved_person_id = conn.execute(
            "SELECT resolved_person_id FROM unknown_voices WHERE unknown_voice_id=?",
            (unknown_voice_id,),
        ).fetchone()[0]

    assert transcript == ("Alex", person[0])
    assert enrollment_count == 2
    assert resolved_person_id == person[0]
    assert introduction[1:3] == ("Alex", "Head of Information")
    assert introduction[5] == "Unknown Speaker 17"
    assert introduction[8] == 5
    assert introduction[9] == person[0]
    assert introduction[12] == "voice_enrolled_face_pending"
    assert updates == [{
        "speaker_label": "Unknown Speaker 17",
        "person_name": "Alex",
        "start_ms": 27_000,
        "end_ms": 29_000,
        "global": True,
    }]

    again = process_meeting_introductions(meeting_id)
    assert len(again) == 1
    assert again[0]["speaker_label"] == "Alex"
    assert len(database.list_meeting_introductions(meeting_id)) == 1
    with database.get_conn() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM voice_embeddings WHERE person_id=?",
            (person[0],),
        ).fetchone()[0] == enrollment_count


def test_does_not_overwrite_conflicting_recognized_face_or_voice(tmp_path, monkeypatch):
    meeting_id = _setup_database(tmp_path, monkeypatch)
    recognized_person_id = database.create_person("Jordan")
    cluster_path = tmp_path / "cluster.npy"
    sample_path = tmp_path / "sample.npy"
    cluster_path.write_bytes(b"cluster")
    sample_path.write_bytes(b"sample")
    unknown_voice_id = database.create_unknown_voice(
        "Unknown Speaker 22", cluster_path
    )
    database.add_unknown_voice_sample(unknown_voice_id, sample_path, quality=0.8)
    _add_introduction_segment(
        meeting_id,
        "Unknown Speaker 22",
        "Hi, I'm Alex and I'm Head of Finance.",
    )

    process_meeting_introductions(
        meeting_id,
        [{
            "start_time": 27.0,
            "end_time": 29.0,
            "track_id": 8,
            "person_id": recognized_person_id,
            "activity_score": 0.95,
        }],
    )

    introduction = database.list_meeting_introductions(meeting_id)[0]
    assert introduction[1] == "Alex"
    assert introduction[9] is None
    assert introduction[12] == "pending_identity_conflict"
    assert database.get_unknown_voice_by_label("Unknown Speaker 22")[1] is None
    assert all(person[1] != "Alex" for person in database.list_persons())


def test_matches_introduction_to_existing_recognized_face(tmp_path, monkeypatch):
    meeting_id = _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Sarah")
    segment_id = _add_introduction_segment(
        meeting_id,
        "UNKNOWN",
        "My name is Sarah.",
    )

    process_meeting_introductions(
        meeting_id,
        [{
            "start_time": 27.0,
            "end_time": 29.0,
            "track_id": 11,
            "person_id": person_id,
            "activity_score": 0.92,
        }],
    )

    introduction = database.list_meeting_introductions(meeting_id)[0]
    with database.get_conn() as conn:
        transcript = conn.execute(
            """
            SELECT speaker_label, person_id FROM transcription_segments
            WHERE segment_id=?
            """,
            (segment_id,),
        ).fetchone()
    assert introduction[12] == "matched_face_and_voice"
    assert introduction[9] == person_id
    assert transcript == ("Sarah", person_id)


def test_enrolls_uniquely_aligned_unknown_face_and_reprocesses_idempotently(
    tmp_path, monkeypatch
):
    meeting_id = _setup_database(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "KNOWN_FACES_DIR", tmp_path / "known_faces")
    embedding_path = tmp_path / "unknown-face.npy"
    image_path = tmp_path / "unknown-face.jpg"
    np.save(embedding_path, np.ones(512, dtype=np.float32))
    face_unknown_id = database.create_unknown(
        "track_9_unique-run",
        image_path,
        embedding_path,
    )
    database.add_unknown_sample(
        face_unknown_id,
        image_path,
        embedding_path,
        quality=0.9,
    )
    segment_id = _add_introduction_segment(
        meeting_id,
        "UNKNOWN",
        "Hi, I'm Taylor and I'm Head of Research.",
    )
    events = [{
        "start_time": 27.0,
        "end_time": 29.0,
        "track_id": 9,
        "face_unknown_label": "track_9_unique-run",
        "person_id": None,
        "activity_score": 0.95,
    }]

    process_meeting_introductions(meeting_id, events)

    person_id = next(
        person[0] for person in database.list_persons()
        if person[1] == "Taylor"
    )
    introduction = database.list_meeting_introductions(meeting_id)[0]
    with database.get_conn() as conn:
        transcript = conn.execute(
            """
            SELECT speaker_label, person_id FROM transcription_segments
            WHERE segment_id=?
            """,
            (segment_id,),
        ).fetchone()
        face_embedding_count = conn.execute(
            "SELECT COUNT(*) FROM face_embeddings WHERE person_id=?",
            (person_id,),
        ).fetchone()[0]
        face_owner = conn.execute(
            "SELECT resolved_person_id FROM unknown_tracks WHERE unknown_id=?",
            (face_unknown_id,),
        ).fetchone()[0]

    assert introduction[12] == "matched_face_voice_pending"
    assert introduction[14] == face_unknown_id
    assert transcript == ("Taylor", person_id)
    assert face_embedding_count == 1
    assert face_owner == person_id

    process_meeting_introductions(meeting_id, events)
    with database.get_conn() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM face_embeddings WHERE person_id=?",
            (person_id,),
        ).fetchone()[0] == 1
    assert len(database.list_meeting_introductions(meeting_id)) == 1


def test_renames_a_generic_voice_identity_without_creating_a_duplicate(
    tmp_path, monkeypatch
):
    meeting_id = _setup_database(tmp_path, monkeypatch)
    existing_person_id = database.create_person("The Centre 3")
    segment_id = _add_introduction_segment(
        meeting_id,
        "The Centre 3",
        "Hi, I'm Rachel and I work in administration.",
    )

    process_meeting_introductions(meeting_id)

    people = database.list_persons()
    assert [(row[0], row[1]) for row in people] == [
        (existing_person_id, "Rachel")
    ]
    introduction = database.list_meeting_introductions(meeting_id)[0]
    with database.get_conn() as conn:
        transcript = conn.execute(
            """
            SELECT speaker_label, person_id FROM transcription_segments
            WHERE segment_id=?
            """,
            (segment_id,),
        ).fetchone()
    assert introduction[12] == "matched_voice_named"
    assert introduction[9] == existing_person_id
    assert transcript == ("Rachel", existing_person_id)


def test_multiple_introductions_in_one_utterance_remain_pending(tmp_path, monkeypatch):
    meeting_id = _setup_database(tmp_path, monkeypatch)
    _add_introduction_segment(
        meeting_id,
        "UNKNOWN",
        "Hi, I'm Alex. Hi, I'm Rachel.",
    )

    process_meeting_introductions(meeting_id)

    introductions = database.list_meeting_introductions(meeting_id)
    assert [row[1] for row in introductions] == ["Alex", "Rachel"]
    assert all(row[12] == "pending_multiple_introductions" for row in introductions)
    assert database.list_persons() == []


def test_missing_enrollment_file_leaves_introduction_pending(tmp_path, monkeypatch):
    meeting_id = _setup_database(tmp_path, monkeypatch)
    unknown_voice_id = database.create_unknown_voice(
        "Unknown Speaker 31",
        tmp_path / "missing-cluster.npy",
    )
    database.add_unknown_voice_sample(
        unknown_voice_id,
        tmp_path / "missing-sample.npy",
        quality=0.8,
    )
    _add_introduction_segment(
        meeting_id,
        "Unknown Speaker 31",
        "My name is Sam.",
    )

    process_meeting_introductions(meeting_id)

    introduction = database.list_meeting_introductions(meeting_id)[0]
    assert introduction[1] == "Sam"
    assert introduction[12] == "pending_voice_enrollment"
    assert "missing" in introduction[13].lower()
    assert database.list_persons() == []


def test_manual_confirmation_enrolls_pending_voice_and_updates_source_segments(
    tmp_path, monkeypatch
):
    meeting_id = _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alex")
    cluster_path = tmp_path / "manual-cluster.npy"
    sample_path = tmp_path / "manual-sample.npy"
    cluster_path.write_bytes(b"cluster")
    sample_path.write_bytes(b"sample")
    unknown_voice_id = database.create_unknown_voice(
        "Unknown Speaker 41",
        cluster_path,
    )
    database.add_unknown_voice_sample(
        unknown_voice_id,
        sample_path,
        quality=0.85,
    )
    segment_id = _add_introduction_segment(
        meeting_id,
        "Unknown Speaker 41",
        "My name is Alex.",
    )
    introduction_id = database.save_meeting_introduction(
        meeting_id,
        "Alex",
        None,
        27_000,
        29_000,
        "Unknown Speaker 41",
        [segment_id],
        0.85,
        "pending_voice_enrollment",
        voice_unknown_id=unknown_voice_id,
        reason="awaiting manual confirmation",
    )

    assert database.confirm_meeting_introduction(
        introduction_id,
        person_id,
    ) == person_id

    introduction = database.list_meeting_introductions(meeting_id)[0]
    with database.get_conn() as conn:
        transcript = conn.execute(
            """
            SELECT speaker_label, person_id FROM transcription_segments
            WHERE segment_id=?
            """,
            (segment_id,),
        ).fetchone()
        enrollment_ids = conn.execute(
            "SELECT person_id FROM voice_embeddings ORDER BY embedding_id"
        ).fetchall()
        resolved_person_id = conn.execute(
            "SELECT resolved_person_id FROM unknown_voices WHERE unknown_voice_id=?",
            (unknown_voice_id,),
        ).fetchone()[0]

    assert introduction[12] == "manually_confirmed"
    assert transcript == ("Alex", person_id)
    assert enrollment_ids == [(person_id,), (person_id,)]
    assert resolved_person_id == person_id


def test_manual_confirmation_enrolls_linked_unknown_face_track(
    tmp_path, monkeypatch
):
    meeting_id = _setup_database(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "KNOWN_FACES_DIR", tmp_path / "known_faces")
    person_id = database.create_person("Alex")
    image_path = tmp_path / "manual-face.jpg"
    embedding_path = tmp_path / "manual-face.npy"
    np.save(embedding_path, np.ones(512, dtype=np.float32))
    face_unknown_id = database.create_unknown(
        "track_14_manual-run",
        image_path,
        embedding_path,
    )
    database.add_unknown_sample(
        face_unknown_id,
        image_path,
        embedding_path,
        quality=0.9,
    )
    segment_id = _add_introduction_segment(
        meeting_id,
        "UNKNOWN",
        "My name is Alex.",
    )
    introduction_id = database.save_meeting_introduction(
        meeting_id,
        "Alex",
        None,
        27_000,
        29_000,
        "UNKNOWN",
        [segment_id],
        0.85,
        "pending_face_identity",
        face_track_id=14,
        face_unknown_id=face_unknown_id,
        reason="awaiting manual confirmation",
    )

    database.confirm_meeting_introduction(introduction_id, person_id)

    introduction = database.list_meeting_introductions(meeting_id)[0]
    with database.get_conn() as conn:
        face_owner = conn.execute(
            "SELECT resolved_person_id FROM unknown_tracks WHERE unknown_id=?",
            (face_unknown_id,),
        ).fetchone()[0]
        embedding_count = conn.execute(
            "SELECT COUNT(*) FROM face_embeddings WHERE person_id=?",
            (person_id,),
        ).fetchone()[0]
    assert introduction[12] == "manually_confirmed"
    assert introduction[14] == face_unknown_id
    assert face_owner == person_id
    assert embedding_count == 1

    process_meeting_introductions(meeting_id)
    assert database.list_meeting_introductions(meeting_id)[0][12] == (
        "manually_confirmed"
    )
