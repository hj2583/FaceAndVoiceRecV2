import sqlite3

import pytest

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

    audio_path = tmp_path / "alice.wav"
    audio_path.write_bytes(b"original audio")
    audio_embedding_id = database.add_voice_embedding(
        person_id,
        tmp_path / "alice-audio.npy",
        quality=1.0,
        audio_path=audio_path,
    )
    enrollment = database.list_voice_enrollments()[0]
    assert enrollment[:5] == (
        audio_embedding_id,
        person_id,
        "Alice",
        str(tmp_path / "alice-audio.npy"),
        str(audio_path),
    )


def test_reassign_voice_embedding_changes_person_without_moving_sample_files(
    tmp_path, monkeypatch
):
    _setup_database(tmp_path, monkeypatch)
    original_person_id = database.create_person("Alice")
    replacement_person_id = database.create_person("Bob")
    embedding_path = tmp_path / "alice.npy"
    audio_path = tmp_path / "alice.wav"
    embedding_path.write_bytes(b"embedding")
    audio_path.write_bytes(b"audio")
    embedding_id = database.add_voice_embedding(
        original_person_id,
        embedding_path,
        quality=0.9,
        audio_path=audio_path,
    )

    assert database.reassign_voice_embedding(embedding_id, replacement_person_id)

    enrollment = database.list_voice_enrollments()[0]
    assert enrollment[:5] == (
        embedding_id,
        replacement_person_id,
        "Bob",
        str(embedding_path),
        str(audio_path),
    )
    assert enrollment[5] == pytest.approx(0.9)
    assert embedding_path.read_bytes() == b"embedding"
    assert audio_path.read_bytes() == b"audio"


def test_reassign_voice_embedding_reports_missing_records_and_people(
    tmp_path, monkeypatch
):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alice")
    embedding_path = tmp_path / "alice.npy"
    embedding_path.write_bytes(b"embedding")
    embedding_id = database.add_voice_embedding(person_id, embedding_path)

    assert database.reassign_voice_embedding(embedding_id + 1, person_id) is False
    with pytest.raises(ValueError, match="does not exist"):
        database.reassign_voice_embedding(embedding_id, person_id + 1)

    assert database.list_voice_enrollments()[0][1] == person_id


def test_reassign_voice_embedding_updates_only_its_resolved_source_cluster(
    tmp_path, monkeypatch
):
    _setup_database(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "DB_PATH", database.DB_PATH)
    transcripts_dir = tmp_path / "transcripts"
    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", transcripts_dir)
    original_person_id = database.create_person("Person 1")
    replacement_person_id = database.create_person("The Centre 1")
    unrelated_embedding = tmp_path / "unrelated.npy"
    unrelated_embedding.write_bytes(b"unrelated")
    unrelated_embedding_id = database.add_voice_embedding(
        original_person_id, unrelated_embedding
    )

    with database.get_conn() as conn:
        conn.execute(
            """
            INSERT INTO meetings(video_path, created_at, transcription_status)
            VALUES (?, ?, 'completed')
            """,
            ("meeting.mp4", database.utc_now()),
        )
        meeting_id = conn.execute(
            "SELECT meeting_id FROM meetings WHERE video_path='meeting.mp4'"
        ).fetchone()[0]
    meeting_dir = transcripts_dir / str(meeting_id)
    meeting_dir.mkdir(parents=True)
    audio_path = meeting_dir / "audio.wav"
    audio_path.write_bytes(b"meeting audio")

    sample_embedding = tmp_path / "sample-3.npy"
    sibling_embedding = tmp_path / "sibling.npy"
    cluster_embedding = tmp_path / "cluster.npy"
    for path in (sample_embedding, sibling_embedding, cluster_embedding):
        path.write_bytes(b"embedding")
    unknown_voice_id = database.create_unknown_voice(
        "Unknown Speaker 2", cluster_embedding
    )
    database.add_unknown_voice_sample(
        unknown_voice_id,
        sample_embedding,
        quality=0.9,
        audio_path=audio_path,
        source_ref=audio_path,
        start_ms=17_328,
        end_ms=18_828,
    )
    database.add_unknown_voice_sample(
        unknown_voice_id,
        sibling_embedding,
        quality=0.8,
        audio_path=audio_path,
        source_ref=audio_path,
        start_ms=19_000,
        end_ms=20_000,
    )
    database.resolve_unknown_voice(unknown_voice_id, original_person_id)

    unrelated_unknown_embedding = tmp_path / "other-cluster.npy"
    unrelated_unknown_sample = tmp_path / "other-sample.npy"
    unrelated_unknown_embedding.write_bytes(b"embedding")
    unrelated_unknown_sample.write_bytes(b"embedding")
    other_unknown_id = database.create_unknown_voice(
        "Unknown Speaker 3", unrelated_unknown_embedding
    )
    database.add_unknown_voice_sample(
        other_unknown_id,
        unrelated_unknown_sample,
        quality=0.8,
    )
    database.resolve_unknown_voice(other_unknown_id, original_person_id)

    with database.get_conn() as conn:
        conn.executemany(
            """
            INSERT INTO transcription_segments(
                meeting_id, speaker_label, person_id, start_ms, end_ms, text, created_at
            ) VALUES (?, 'Person 1', ?, ?, ?, 'sample speech', ?)
            """,
            [
                (
                    meeting_id,
                    original_person_id,
                    17_328,
                    18_828,
                    database.utc_now(),
                ),
                (
                    meeting_id,
                    original_person_id,
                    30_000,
                    31_000,
                    database.utc_now(),
                ),
            ],
        )

    sample_enrollment_id = next(
        row[0]
        for row in database.list_voice_enrollments()
        if row[3] == str(sample_embedding)
    )
    assert database.reassign_voice_embedding(
        sample_enrollment_id, replacement_person_id
    )

    with database.get_conn() as conn:
        transcript_rows = conn.execute(
            """
            SELECT speaker_label, person_id
            FROM transcription_segments
            WHERE meeting_id=?
            ORDER BY start_ms
            """,
            (meeting_id,),
        ).fetchall()
        sample_and_sibling_persons = conn.execute(
            """
            SELECT person_id FROM voice_embeddings
            WHERE embedding_path IN (?, ?)
            ORDER BY embedding_path
            """,
            (str(sample_embedding), str(sibling_embedding)),
        ).fetchall()
        unknown_person = conn.execute(
            "SELECT resolved_person_id FROM unknown_voices WHERE unknown_voice_id=?",
            (unknown_voice_id,),
        ).fetchone()[0]
        other_unknown_person = conn.execute(
            "SELECT resolved_person_id FROM unknown_voices WHERE unknown_voice_id=?",
            (other_unknown_id,),
        ).fetchone()[0]
        unrelated_person_id = conn.execute(
            "SELECT person_id FROM voice_embeddings WHERE embedding_id=?",
            (unrelated_embedding_id,),
        ).fetchone()[0]

    assert transcript_rows == [
        ("The Centre 1", replacement_person_id),
        ("Person 1", original_person_id),
    ]
    assert sample_and_sibling_persons == [
        (replacement_person_id,),
        (replacement_person_id,),
    ]
    assert unknown_person == replacement_person_id
    assert other_unknown_person == original_person_id
    assert unrelated_person_id == original_person_id
    assert database.list_resolved_unknown_voice_samples_for_source(
        str(audio_path)
    )[0][:2] == (replacement_person_id, "The Centre 1")
    import voice_core

    assert voice_core._match_resolved_source_identity(
        str(audio_path),
        [(17.328, 18.828)],
    ) == {
        "person_id": replacement_person_id,
        "person_name": "The Centre 1",
    }


def test_source_reprocessing_prefers_reassigned_sample_over_stale_cluster_person(
    tmp_path, monkeypatch
):
    _setup_database(tmp_path, monkeypatch)
    original_person_id = database.create_person("Person 1")
    replacement_person_id = database.create_person("The Centre 1")
    cluster_path = tmp_path / "unknown-cluster.npy"
    sample_path = tmp_path / "sample-3.npy"
    cluster_path.write_bytes(b"cluster embedding")
    sample_path.write_bytes(b"sample embedding")
    unknown_voice_id = database.create_unknown_voice(
        "Unknown Speaker 2", cluster_path
    )
    source_ref = str(tmp_path / "transcripts" / "3" / "audio.wav")
    database.add_unknown_voice_sample(
        unknown_voice_id,
        sample_path,
        source_ref=source_ref,
        start_ms=17_328,
        end_ms=18_828,
    )
    database.resolve_unknown_voice(unknown_voice_id, original_person_id)
    with database.get_conn() as conn:
        conn.execute(
            "UPDATE voice_embeddings SET person_id=? WHERE embedding_path=?",
            (replacement_person_id, str(sample_path)),
        )

    assert database.list_unknown_voices(include_resolved=True)[0][4] == original_person_id
    import voice_core

    assert voice_core._match_resolved_source_identity(
        source_ref,
        [(17.328, 18.828)],
    ) == {
        "person_id": replacement_person_id,
        "person_name": "The Centre 1",
    }


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


def test_voice_audio_schema_migrates_existing_database(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute("""
            CREATE TABLE voice_embeddings (
                embedding_id INTEGER PRIMARY KEY,
                person_id INTEGER NOT NULL,
                embedding_path TEXT NOT NULL,
                quality REAL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE unknown_voices (
                unknown_voice_id INTEGER PRIMARY KEY,
                label TEXT NOT NULL,
                embedding_path TEXT,
                created_at TEXT NOT NULL,
                resolved_person_id INTEGER
            )
        """)
        connection.execute("""
            CREATE TABLE unknown_voice_samples (
                sample_id INTEGER PRIMARY KEY,
                unknown_voice_id INTEGER NOT NULL,
                embedding_path TEXT NOT NULL,
                quality REAL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)

    database.init_db()

    with sqlite3.connect(db_path) as connection:
        voice_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(voice_embeddings)")
        }
        sample_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(unknown_voice_samples)")
        }
        indexes = {
            row[1]
            for row in connection.execute("PRAGMA index_list(unknown_voice_samples)")
        }
    assert "audio_path" in voice_columns
    assert {"audio_path", "source_ref", "start_ms", "end_ms"} <= sample_columns
    assert "idx_unknown_voice_sample_source_window" in indexes


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


def test_resolving_unknown_voice_enrolls_audio_samples_and_can_create_person(
    tmp_path, monkeypatch
):
    _setup_database(tmp_path, monkeypatch)
    embedding_path = tmp_path / "unknown.npy"
    sample_embedding_path = tmp_path / "sample.npy"
    sample_audio_path = tmp_path / "sample.wav"
    embedding_path.write_bytes(b"centroid")
    sample_embedding_path.write_bytes(b"sample embedding")
    sample_audio_path.write_bytes(b"original voice audio")
    unknown_voice_id = database.create_unknown_voice(
        "Unknown Speaker 12",
        embedding_path,
    )
    database.add_unknown_voice_sample(
        unknown_voice_id,
        sample_embedding_path,
        quality=0.8,
        audio_path=sample_audio_path,
    )

    person_id = database.resolve_unknown_voice(
        unknown_voice_id,
        new_person_name="New Speaker",
    )

    assert database.list_unknown_voices() == []
    enrollments = database.list_voice_enrollments()
    assert {row[3] for row in enrollments} == {
        str(embedding_path),
        str(sample_embedding_path),
    }
    assert {row[4] for row in enrollments} == {None, str(sample_audio_path)}
    assert {row[1] for row in enrollments} == {person_id}
    with pytest.raises(ValueError, match="already resolved"):
        database.resolve_unknown_voice(unknown_voice_id, person_id=person_id)


def test_delete_unknown_voice_removes_record_samples_and_files(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    embedding_path = tmp_path / "cluster.npy"
    embedding_path.write_bytes(b"embedding")
    unknown_voice_id = database.create_unknown_voice("Speaker 1", embedding_path)
    sample_path = tmp_path / "sample.npy"
    sample_path.write_bytes(b"sample")
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    database.add_unknown_voice_sample(
        unknown_voice_id,
        sample_path,
        audio_path=audio_path,
    )

    assert database.delete_unknown_voice(unknown_voice_id) is True

    assert not embedding_path.exists()
    assert not sample_path.exists()
    assert not audio_path.exists()
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
