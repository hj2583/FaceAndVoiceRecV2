from attendance_view import attendance_frame_to_csv, build_attendance_frame, format_clip_offset


def test_attendance_frame_keeps_realtime_dates_separate_from_video_offsets():
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
        "Person", "Source", "Observed (UTC)", "Video", "Clip Time",
        "Confidence", "Track ID",
    ]
    assert frame.iloc[0]["Observed (UTC)"] == "2026-10-05T12:30:00+00:00"
    assert frame.iloc[0]["Clip Time"] == ""
    assert frame.iloc[1]["Observed (UTC)"] == ""
    assert frame.iloc[1]["Video"] == "a.mp4"
    assert frame.iloc[1]["Clip Time"] == "01:00"


def test_clip_offset_formatting_handles_minute_and_hour_boundaries():
    assert format_clip_offset(0) == "00:00"
    assert format_clip_offset(59999) == "00:59"
    assert format_clip_offset(60000) == "01:00"
    assert format_clip_offset(3600000) == "1:00:00"


def test_attendance_csv_exports_exact_frame_columns():
    frame = build_attendance_frame([{
        "person_name": "Alice", "source": "video", "observed_at_utc": None,
        "media_offset_ms": 0, "confidence": 0.9, "track_id": 1,
        "source_ref": "C:/videos/a.mp4",
    }])

    csv_text = attendance_frame_to_csv(frame).decode("utf-8-sig")
    assert csv_text.splitlines()[0] == (
        "Person,Source,Observed (UTC),Video,Clip Time,Confidence,Track ID"
    )
    assert "Alice,video,,a.mp4,00:00,0.9,1" in csv_text
