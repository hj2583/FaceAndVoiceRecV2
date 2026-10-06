from datetime import datetime, timedelta, timezone
from pathlib import Path

import database
import pytest

from attendance_view import (
    attendance_frame_to_csv,
    build_attendance_frame,
    filter_visible_attendance,
    format_clip_offset,
    local_datetime_range_to_utc_iso,
    local_datetime_to_utc_iso,
    video_source_options,
)


def _setup_database(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "faces.db")
    database.init_db()


def test_filter_visible_attendance_excludes_unknown_empty_and_missing_confidence_without_threshold():
    rows = [
        {"person_id": 1, "person_name": "Alice", "confidence": 0.55},
        {"person_id": 2, "person_name": "Bob", "confidence": 0.10},
        {"person_id": 3, "person_name": "UNKNOWN", "confidence": 0.95},
        {"person_id": 4, "person_name": "", "confidence": 0.9},
        {"person_id": 5, "person_name": "Charlie", "confidence": None},
        {"person_id": None, "person_name": "Dana", "confidence": 0.9},
    ]

    result = filter_visible_attendance(rows)

    assert [row["person_name"] for row in result] == ["Alice", "Bob"]


def test_attendance_frame_keeps_realtime_dates_separate_from_video_offsets_and_has_source_ref():
    frame = build_attendance_frame([
        {
            "person_name": "Alice", "source": "realtime",
            "observed_at_utc": "2026-10-05T12:30:00+00:00",
            "media_offset_ms": None, "confidence": 0.91, "track_id": 4,
            "source_ref": "run-1",
        },
        {
            "person_name": "Alice", "source": "video",
            "observed_at_utc": None, "media_offset_ms": 60000,
            "confidence": 0.95, "track_id": 2,
            "source_ref": "C:/videos/a.mp4",
        },
    ])

    assert list(frame.columns) == [
        "Person", "Source", "Observed (UTC)", "Video", "Source Ref", "Clip Time",
        "Confidence", "Track ID",
    ]
    assert frame.iloc[0]["Observed (UTC)"] == "2026-10-05T12:30:00+00:00"
    assert frame.iloc[0]["Source Ref"] == "run-1"
    assert frame.iloc[0]["Clip Time"] == ""
    assert frame.iloc[1]["Observed (UTC)"] == ""
    assert frame.iloc[1]["Video"] == "a.mp4"
    assert frame.iloc[1]["Source Ref"] == "C:/videos/a.mp4"
    assert frame.iloc[1]["Clip Time"] == "01:00"


def test_clip_offset_formatting_handles_minute_and_hour_boundaries():
    assert format_clip_offset(0) == "00:00"
    assert format_clip_offset(59999) == "00:59"
    assert format_clip_offset(60000) == "01:00"
    assert format_clip_offset(3600000) == "1:00:00"


def test_attendance_csv_escapes_formula_like_text_cells_but_keeps_numeric_cells():
    frame = build_attendance_frame([
        {
            "person_name": "=Alice",
            "source": "video",
            "observed_at_utc": None,
            "media_offset_ms": 0,
            "confidence": 0.9,
            "track_id": -1,
            "source_ref": "@/tmp/+danger/-clip.mp4",
        }
    ])

    csv_text = attendance_frame_to_csv(frame).decode("utf-8-sig")
    assert csv_text.splitlines()[0] == (
        "Person,Source,Observed (UTC),Video,Source Ref,Clip Time,Confidence,Track ID"
    )
    assert "'=Alice" in csv_text
    assert "'@/tmp/+danger/-clip.mp4" in csv_text
    assert ",0.9,-1" in csv_text


def test_attendance_helpers_accept_sqlite_rows_from_fetch_and_export_expected_values(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)

    alice_id = database.create_person("Alice")
    bob_id = database.create_person("Bob")

    assert database.record_attendance_event(
        person_id=alice_id,
        person_name="Alice",
        source="realtime",
        source_ref="run-abc",
        track_id=10,
        confidence=0.52,
        observed_at_utc="2026-10-06T11:30:00+00:00",
    )

    source_video = str((tmp_path / "videos" / "clip.mp4").resolve())
    inserted = database.replace_video_attendance(
        source_video,
        [
            {
                "person_id": bob_id,
                "person_name": "Bob",
                "track_id": 99,
                "confidence": 0.88,
                "media_offset_ms": 65000,
            }
        ],
    )
    assert inserted == 1

    fetched_rows = database.fetch_attendance()
    assert fetched_rows
    assert type(fetched_rows[0]).__name__ == "Row"

    visible_rows = filter_visible_attendance(fetched_rows)
    frame = build_attendance_frame(visible_rows)
    csv_text = attendance_frame_to_csv(frame).decode("utf-8-sig")

    expected_video_ref = Path(source_video).as_posix()
    assert set(frame["Source"]) == {"realtime", "video"}
    assert "run-abc" in set(frame["Source Ref"])
    assert expected_video_ref in set(frame["Source Ref"])

    video_rows = frame[frame["Source"] == "video"]
    assert video_rows.iloc[0]["Video"] == "clip.mp4"
    assert video_rows.iloc[0]["Clip Time"] == "01:05"

    assert "run-abc" in csv_text
    assert expected_video_ref in csv_text
    assert "clip.mp4" in csv_text


def test_local_datetime_range_converts_to_utc_inclusively():
    start = datetime(2026, 10, 6, 9, 0, tzinfo=timezone(timedelta(hours=8)))
    end = datetime(2026, 10, 6, 10, 0, tzinfo=timezone(timedelta(hours=8)))

    assert local_datetime_to_utc_iso(start) == "2026-10-06T01:00:00+00:00"
    assert local_datetime_range_to_utc_iso(start, end) == (
        "2026-10-06T01:00:00+00:00",
        "2026-10-06T02:00:00+00:00",
    )


def test_local_datetime_naive_value_is_interpreted_in_host_timezone():
    value = datetime(2026, 10, 6, 11, 0)

    assert local_datetime_to_utc_iso(value) == value.astimezone(timezone.utc).isoformat()


def test_local_datetime_range_rejects_reversed_endpoints():
    start = datetime(2026, 10, 6, 11, 0)
    end = datetime(2026, 10, 6, 10, 0)

    with pytest.raises(ValueError, match="start.*end"):
        local_datetime_range_to_utc_iso(start, end)


def test_video_source_options_keep_equal_basenames_distinguishable():
    refs = [
        "C:/capture/day1/session/clip.mp4",
        "D:/archive/day2/session/clip.mp4",
    ]

    options, labels = video_source_options(refs)

    assert options == refs
    assert labels[refs[0]] != labels[refs[1]]
    assert "clip.mp4" in labels[refs[0]]
    assert "clip.mp4" in labels[refs[1]]
