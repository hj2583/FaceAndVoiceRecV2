from pathlib import Path

import config
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


def test_resolve_unknown_voice_relabels_existing_transcript_segments(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Ivy")
    embedding_path = tmp_path / "cluster.npy"
    embedding_path.write_bytes(b"embedding")
    unknown_voice_id = database.create_unknown_voice("Unknown Speaker 1", embedding_path)

    with database.get_conn() as conn:
        conn.execute(
            "INSERT INTO meetings(video_path, created_at, transcription_status) VALUES (?, ?, 'completed')",
            ("video.mp4", database.utc_now()),
        )
        meeting_id = conn.execute("SELECT meeting_id FROM meetings").fetchone()[0]
        conn.execute(
            """
            INSERT INTO transcription_segments(
                meeting_id, speaker_label, start_ms, end_ms, text, created_at
            ) VALUES (?, 'Unknown Speaker 1', 0, 1000, 'hi', ?)
            """,
            (meeting_id, database.utc_now()),
        )

    database.resolve_unknown_voice(unknown_voice_id, person_id)

    with database.get_conn() as conn:
        row = conn.execute(
            "SELECT speaker_label, person_id FROM transcription_segments WHERE meeting_id=?",
            (meeting_id,),
        ).fetchone()

    assert row == ("Ivy", person_id)


def test_rename_person_relabels_existing_transcript_segments(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Old Name")

    with database.get_conn() as conn:
        conn.execute(
            "INSERT INTO meetings(video_path, created_at, transcription_status) VALUES (?, ?, 'completed')",
            ("video.mp4", database.utc_now()),
        )
        meeting_id = conn.execute("SELECT meeting_id FROM meetings").fetchone()[0]
        conn.execute(
            """
            INSERT INTO transcription_segments(
                meeting_id, speaker_label, person_id, start_ms, end_ms, text, created_at
            ) VALUES (?, 'Old Name', ?, 0, 1000, 'hi', ?)
            """,
            (meeting_id, person_id, database.utc_now()),
        )

    database.rename_person(person_id, "New Name")

    with database.get_conn() as conn:
        row = conn.execute(
            "SELECT speaker_label FROM transcription_segments WHERE meeting_id=?",
            (meeting_id,),
        ).fetchone()

    assert row[0] == "New Name"


def test_rename_person_regenerates_meeting_transcript_files(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Old Name")

    with database.get_conn() as conn:
        conn.execute(
            "INSERT INTO meetings(video_path, created_at, transcription_status) VALUES (?, ?, 'completed')",
            ("video.mp4", database.utc_now()),
        )
        meeting_id = conn.execute("SELECT meeting_id FROM meetings").fetchone()[0]
        conn.execute(
            """
            INSERT INTO transcription_segments(
                meeting_id, speaker_label, person_id, start_ms, end_ms, text, created_at
            ) VALUES (?, 'Old Name', ?, 61000, 62000, 'hi there', ?)
            """,
            (meeting_id, person_id, database.utc_now()),
        )

    meeting_dir = config.TRANSCRIPTS_DIR / str(meeting_id)
    meeting_dir.mkdir(parents=True)
    (meeting_dir / "old_name.txt").write_text("[01:01] hi there\n", encoding="utf-8")
    (meeting_dir / "audio.wav").write_bytes(b"audio")

    database.rename_person(person_id, "New Name")

    assert sorted(path.name for path in meeting_dir.glob("*.txt")) == ["new_name.txt"]
    assert (meeting_dir / "new_name.txt").read_text(encoding="utf-8") == "[01:01] hi there\n"
    assert (meeting_dir / "audio.wav").exists()
