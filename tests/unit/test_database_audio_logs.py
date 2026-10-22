from face_voice_rec.storage import database


def _setup_database(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "faces.db")
    database.init_db()


def test_log_audio_stores_and_returns_track_id(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alice")

    database.log_audio(1.0, 2.5, person_id, "Alice", 0.9, "video", track_id=3)

    rows = database.fetch_audio_logs("video")
    assert len(rows) == 1
    _log_id, start, end, name, confidence, source, transcript, track_id = rows[0]
    assert (start, end, name, confidence, source, transcript, track_id) == (
        1.0, 2.5, "Alice", 0.9, "video", None, 3,
    )


def test_log_audio_defaults_track_id_to_none(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)

    database.log_audio(0.0, 1.0, None, "UNKNOWN", 0.4, "realtime")

    rows = database.fetch_audio_logs()
    assert rows[0][-1] is None


def test_fetch_audio_logs_filters_by_source_and_person(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    alice_id = database.create_person("Alice")
    bob_id = database.create_person("Bob")

    database.log_audio(10.0, 15.0, alice_id, "Alice", 0.95, "realtime")
    database.log_audio(20.0, 25.0, bob_id, "Bob", 0.85, "realtime")
    database.log_audio(30.0, 35.0, None, "UNKNOWN", 0.50, "realtime")
    database.log_audio(1.0, 5.0, alice_id, "Alice", 0.90, "video")

    # Filter source
    realtime_rows = database.fetch_audio_logs(source="realtime")
    assert len(realtime_rows) == 3
    video_rows = database.fetch_audio_logs(source="video")
    assert len(video_rows) == 1

    # Filter person
    alice_realtime = database.fetch_audio_logs(source="realtime", person_id=alice_id)
    assert len(alice_realtime) == 1
    assert alice_realtime[0][3] == "Alice"

    # Filter unknown
    unknown_realtime = database.fetch_audio_logs(source="realtime", person_id="UNKNOWN")
    assert len(unknown_realtime) == 1
    assert unknown_realtime[0][3] == "UNKNOWN"

    # Filter time range
    time_filtered = database.fetch_audio_logs(source="realtime", from_time=15.0, to_time=25.0)
    assert len(time_filtered) == 1
    assert time_filtered[0][3] == "Bob"


def test_fetch_audio_logs_filters_and_lists_video_sources(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    first_video = tmp_path / "first.mp4"
    second_video = tmp_path / "second.mp4"

    database.log_audio(
        1.0, 2.0, None, "Alice", 0.9, "video",
        track_id=1, source_ref=first_video,
    )
    database.log_audio(
        3.0, 4.0, None, "Bob", 0.8, "video",
        track_id=2, source_ref=second_video,
    )
    database.log_audio(5.0, 6.0, None, "Live", 0.7, "realtime")

    assert database.list_audio_log_video_sources() == sorted([
        first_video.resolve().as_posix(),
        second_video.resolve().as_posix(),
    ])
    rows = database.fetch_audio_logs(
        source="video",
        source_ref=first_video,
    )
    assert len(rows) == 1
    assert rows[0][3] == "Alice"





