import sqlite3
from pathlib import Path

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


def test_record_attendance_event_normalizes_offset_aware_observed_time_to_utc(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alice")

    inserted = database.record_attendance_event(
        person_id=person_id,
        person_name="Alice",
        source="realtime",
        source_ref="run-1",
        track_id=4,
        confidence=0.91,
        observed_at_utc="2026-10-05T12:30:00+08:00",
    )

    assert inserted is True
    rows = database.fetch_attendance(source="realtime", person_id=person_id)
    assert rows[0]["observed_at_utc"] == "2026-10-05T04:30:00+00:00"


def test_record_attendance_event_rejects_naive_observed_time(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alice")

    with pytest.raises(ValueError):
        database.record_attendance_event(
            person_id=person_id,
            person_name="Alice",
            source="realtime",
            source_ref="run-1",
            track_id=4,
            confidence=0.91,
            observed_at_utc="2026-10-05T12:30:00",
        )


@pytest.mark.parametrize(
    "bad_confidence",
    [float("nan"), float("inf"), float("-inf"), -0.01, 1.01],
)
def test_record_attendance_event_rejects_non_finite_or_out_of_range_confidence(
    tmp_path,
    monkeypatch,
    bad_confidence,
):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alice")

    with pytest.raises(ValueError):
        database.record_attendance_event(
            person_id=person_id,
            person_name="Alice",
            source="realtime",
            source_ref="run-1",
            track_id=4,
            confidence=bad_confidence,
            observed_at_utc="2026-10-05T12:30:00+00:00",
        )


@pytest.mark.parametrize("bad_track_id", [True, False, -1, 1.25, "1.5"])
def test_record_attendance_event_rejects_invalid_track_id(tmp_path, monkeypatch, bad_track_id):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alice")

    with pytest.raises(ValueError):
        database.record_attendance_event(
            person_id=person_id,
            person_name="Alice",
            source="realtime",
            source_ref="run-1",
            track_id=bad_track_id,
            confidence=0.9,
            observed_at_utc="2026-10-05T12:30:00+00:00",
        )


@pytest.mark.parametrize("bad_offset", [True, False, -1, 1.25, "1.5"])
def test_record_attendance_event_rejects_invalid_video_offset(tmp_path, monkeypatch, bad_offset):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alice")

    with pytest.raises(ValueError):
        database.record_attendance_event(
            person_id=person_id,
            person_name="Alice",
            source="video",
            source_ref="video.mp4",
            track_id=1,
            confidence=0.9,
            media_offset_ms=bad_offset,
        )


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
    video_a = (tmp_path / "videos" / "a.mp4").resolve()
    video_b = (tmp_path / "videos" / "b.mp4").resolve()
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

    database.replace_video_attendance(str(video_a), first)
    database.replace_video_attendance(str(video_b), first)
    assert database.replace_video_attendance(str(video_a), second) == 1

    rows = database.fetch_attendance(source="video")
    assert [(row["source_ref"], row["track_id"], row["media_offset_ms"]) for row in rows] == [
        (video_a.as_posix(), 2, 2400),
        (video_b.as_posix(), 1, 1200),
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
    video_a = (tmp_path / "videos" / "a.mp4").resolve()
    database.replace_video_attendance(str(video_a), old)

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
        database.replace_video_attendance(str(video_a), candidates)

    rows = database.fetch_attendance(source="video")
    assert [(row["track_id"], row["media_offset_ms"]) for row in rows] == [(1, 1000)]


def test_fetch_attendance_orders_realtime_by_observed_time_not_insertion_order(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    first = database.create_person("First")
    second = database.create_person("Second")

    database.record_attendance_event(
        person_id=second,
        person_name="Second",
        source="realtime",
        source_ref="run",
        track_id=2,
        confidence=0.9,
        observed_at_utc="2026-10-05T12:31:00+00:00",
    )
    database.record_attendance_event(
        person_id=first,
        person_name="First",
        source="realtime",
        source_ref="run",
        track_id=1,
        confidence=0.9,
        observed_at_utc="2026-10-05T12:30:00+00:00",
    )

    assert [row["person_name"] for row in database.fetch_attendance()] == [
        "Second",
        "First",
    ]


def test_fetch_attendance_orders_realtime_by_observed_time_when_inserted_out_of_order(
    tmp_path,
    monkeypatch,
):
    _setup_database(tmp_path, monkeypatch)
    first = database.create_person("First")
    second = database.create_person("Second")

    database.record_attendance_event(
        person_id=second,
        person_name="Second",
        source="realtime",
        source_ref="run",
        track_id=2,
        confidence=0.9,
        observed_at_utc="2026-10-05T12:31:00+00:00",
    )
    database.record_attendance_event(
        person_id=first,
        person_name="First",
        source="realtime",
        source_ref="run",
        track_id=1,
        confidence=0.9,
        observed_at_utc="2026-10-05T12:30:00+00:00",
    )

    with database.get_conn() as conn:
        conn.execute("DELETE FROM attendance_events")

    database.record_attendance_event(
        person_id=first,
        person_name="First",
        source="realtime",
        source_ref="run",
        track_id=1,
        confidence=0.9,
        observed_at_utc="2026-10-05T12:30:00+00:00",
    )
    database.record_attendance_event(
        person_id=second,
        person_name="Second",
        source="realtime",
        source_ref="run",
        track_id=2,
        confidence=0.9,
        observed_at_utc="2026-10-05T12:31:00+00:00",
    )

    assert [row["person_name"] for row in database.fetch_attendance(source="realtime")] == [
        "Second",
        "First",
    ]


def test_empty_video_replacement_clears_only_that_video(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alice")
    video_a = str((tmp_path / "videos" / "a.mp4").resolve())
    video_b = str((tmp_path / "videos" / "b.mp4").resolve())
    event = {
        "person_id": person_id,
        "person_name": "Alice",
        "track_id": 1,
        "confidence": 0.9,
        "media_offset_ms": 1000,
    }

    database.replace_video_attendance(video_a, [event])
    database.replace_video_attendance(video_b, [event])

    assert database.replace_video_attendance(video_a, []) == 0
    assert [row["source_ref"] for row in database.fetch_attendance(source="video")] == [
        Path(video_b).as_posix(),
    ]


def test_duplicate_candidates_in_one_video_replacement_are_inserted_once(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alice")
    video_path = str((tmp_path / "videos" / "a.mp4").resolve())
    event = {
        "person_id": person_id,
        "person_name": "Alice",
        "track_id": 1,
        "confidence": 0.9,
        "media_offset_ms": 1000,
    }

    assert database.replace_video_attendance(video_path, [event, event]) == 1
    assert len(database.fetch_attendance(source="video")) == 1


def test_fetch_attendance_filters_inclusive_period_and_video_path(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    alice = database.create_person("Alice")
    bob = database.create_person("Bob")
    video_a = str((tmp_path / "videos" / "a.mp4").resolve())
    video_b = str((tmp_path / "videos" / "b.mp4").resolve())

    for minute in (9, 10, 11):
        database.record_attendance_event(
            person_id=alice,
            person_name="Alice",
            source="realtime",
            source_ref="run-1",
            track_id=minute,
            confidence=0.9,
            observed_at_utc=f"2026-10-06T10:{minute:02d}:00+00:00",
        )
    database.record_attendance_event(
        person_id=bob,
        person_name="Bob",
        source="realtime",
        source_ref="run-1",
        track_id=50,
        confidence=0.95,
        observed_at_utc="2026-10-06T10:10:00+00:00",
    )

    video_event = {
        "person_id": alice,
        "person_name": "Alice",
        "track_id": 1,
        "confidence": 0.9,
        "media_offset_ms": 1000,
    }
    database.replace_video_attendance(video_a, [video_event])
    database.replace_video_attendance(video_b, [video_event])

    realtime_rows = database.fetch_attendance(
        source="realtime",
        person_id=alice,
        observed_from_utc="2026-10-06T10:10:00+00:00",
        observed_to_utc="2026-10-06T10:11:00+00:00",
    )
    assert [row["track_id"] for row in realtime_rows] == [11, 10]

    only_video_a = database.fetch_attendance(source="video", source_ref=video_a)
    assert [row["source_ref"] for row in only_video_a] == [Path(video_a).as_posix()]

    with pytest.raises(ValueError, match="start.*end"):
        database.fetch_attendance(
            source="realtime",
            observed_from_utc="2026-10-06T11:00:00+00:00",
            observed_to_utc="2026-10-06T10:00:00+00:00",
        )
