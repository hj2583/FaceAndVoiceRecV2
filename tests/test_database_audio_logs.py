import database


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
