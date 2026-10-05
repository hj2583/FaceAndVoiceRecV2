import sqlite3

import pytest

import database


def _setup_database(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "faces.db")
    database.init_db()


def test_init_db_creates_attendance_events_table_and_is_idempotent(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    database.init_db()

    with database.get_conn() as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "attendance_events" in names


@pytest.mark.parametrize(
    "payload",
    [
        {
            "person_id": 1,
            "person_name": "Alice",
            "source": "unknown",
            "source_ref": "x",
            "track_id": 1,
            "confidence": 0.9,
            "observed_at_utc": "2026-10-05T12:30:00+00:00",
            "media_offset_ms": None,
        },
        {
            "person_id": 1,
            "person_name": "Alice",
            "source": "realtime",
            "source_ref": "run-1",
            "track_id": 1,
            "confidence": 0.9,
            "observed_at_utc": None,
            "media_offset_ms": None,
        },
        {
            "person_id": 1,
            "person_name": "Alice",
            "source": "realtime",
            "source_ref": "run-1",
            "track_id": 1,
            "confidence": 0.9,
            "observed_at_utc": "2026-10-05T12:30:00+00:00",
            "media_offset_ms": 1000,
        },
        {
            "person_id": 1,
            "person_name": "Alice",
            "source": "video",
            "source_ref": "C:/videos/a.mp4",
            "track_id": 1,
            "confidence": 0.9,
            "observed_at_utc": "2026-10-05T12:30:00+00:00",
            "media_offset_ms": 1000,
        },
        {
            "person_id": 1,
            "person_name": "Alice",
            "source": "video",
            "source_ref": "C:/videos/a.mp4",
            "track_id": 1,
            "confidence": 0.9,
            "observed_at_utc": None,
            "media_offset_ms": None,
        },
    ],
)
def test_record_attendance_event_validates_source_and_timestamps(tmp_path, monkeypatch, payload):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alice")
    payload["person_id"] = person_id

    with pytest.raises(ValueError):
        database.record_attendance_event(**payload)


def test_attendance_event_is_idempotent_per_track(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alice")
    event = {
        "person_id": person_id,
        "person_name": "Alice",
        "source": "realtime",
        "source_ref": "run-1",
        "track_id": 4,
        "confidence": 0.91,
        "observed_at_utc": "2026-10-05T12:30:00+00:00",
        "media_offset_ms": None,
    }

    assert database.record_attendance_event(**event) is True
    assert database.record_attendance_event(**event) is False
    rows = database.fetch_attendance(source="realtime", person_id=person_id)
    assert len(rows) == 1
    assert rows[0]["track_id"] == 4
    assert rows[0]["observed_at_utc"] == event["observed_at_utc"]


def test_fetch_attendance_filters_and_orders_newest_first(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    alice = database.create_person("Alice")
    bob = database.create_person("Bob")

    database.record_attendance_event(
        person_id=alice,
        person_name="Alice",
        source="realtime",
        source_ref="run-1",
        track_id=1,
        confidence=0.8,
        observed_at_utc="2026-10-05T12:30:00+00:00",
    )
    database.record_attendance_event(
        person_id=bob,
        person_name="Bob",
        source="realtime",
        source_ref="run-1",
        track_id=2,
        confidence=0.85,
        observed_at_utc="2026-10-05T12:31:00+00:00",
    )

    all_rows = database.fetch_attendance(source="realtime")
    assert [row["person_name"] for row in all_rows] == ["Bob", "Alice"]

    alice_rows = database.fetch_attendance(source="realtime", person_id=alice)
    assert len(alice_rows) == 1
    assert alice_rows[0]["person_name"] == "Alice"


def test_replace_video_attendance_replaces_only_matching_video(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alice")
    first = [{
        "person_id": person_id,
        "person_name": "Alice",
        "track_id": 1,
        "confidence": 0.9,
        "media_offset_ms": 1200,
    }]
    second = [{
        "person_id": person_id,
        "person_name": "Alice",
        "track_id": 2,
        "confidence": 0.95,
        "media_offset_ms": 2400,
    }]

    database.replace_video_attendance("C:/videos/a.mp4", first)
    database.replace_video_attendance("C:/videos/b.mp4", first)
    assert database.replace_video_attendance("C:/videos/a.mp4", second) == 1

    rows = database.fetch_attendance(source="video")
    assert [(row["source_ref"], row["track_id"], row["media_offset_ms"]) for row in rows] == [
        ("C:/videos/a.mp4", 2, 2400),
        ("C:/videos/b.mp4", 1, 1200),
    ]


def test_replace_video_attendance_rolls_back_on_invalid_candidate(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alice")
    old = [{
        "person_id": person_id,
        "person_name": "Alice",
        "track_id": 1,
        "confidence": 0.9,
        "media_offset_ms": 1000,
    }]
    database.replace_video_attendance("C:/videos/a.mp4", old)

    candidates = [
        {
            "person_id": person_id,
            "person_name": "Alice",
            "track_id": 2,
            "confidence": 0.95,
            "media_offset_ms": 2000,
        },
        {
            "person_id": 999999,
            "person_name": "Missing",
            "track_id": 3,
            "confidence": 0.9,
            "media_offset_ms": 3000,
        },
    ]
    with pytest.raises(sqlite3.IntegrityError):
        database.replace_video_attendance("C:/videos/a.mp4", candidates)

    rows = database.fetch_attendance(source="video")
    assert [(row["track_id"], row["media_offset_ms"]) for row in rows] == [(1, 1000)]
