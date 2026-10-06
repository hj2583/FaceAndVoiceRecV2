from attendance_view import attendance_frame_to_csv, build_attendance_frame, filter_visible_attendance, format_clip_offset


def test_filter_visible_attendance_uses_threshold_and_excludes_unknown():
    """Test that filter_visible_attendance correctly filters by threshold and excludes UNKNOWN/empty names."""
    rows = [
        {"person_name": "Alice", "confidence": 0.70},
        {"person_name": "Bob", "confidence": 0.699},
        {"person_name": "UNKNOWN", "confidence": 0.95},
        {"person_name": "", "confidence": 0.90},
        {"person_name": "Charlie", "confidence": None},
    ]

    result = filter_visible_attendance(rows, threshold=0.70)
    
    # Only Alice should pass: confidence >= threshold, known name, confidence not None
    assert len(result) == 1
    assert result[0]["person_name"] == "Alice"
    assert result[0]["confidence"] == 0.70


def test_filter_visible_attendance_with_various_thresholds():
    """Test threshold boundary behavior."""
    rows = [
        {"person_name": "Alice", "confidence": 0.75},
        {"person_name": "Bob", "confidence": 0.70},
        {"person_name": "Charlie", "confidence": 0.69},
    ]

    # Threshold 0.70: Alice and Bob pass
    result = filter_visible_attendance(rows, threshold=0.70)
    assert len(result) == 2
    assert [r["person_name"] for r in result] == ["Alice", "Bob"]

    # Threshold 0.75: only Alice passes
    result = filter_visible_attendance(rows, threshold=0.75)
    assert len(result) == 1
    assert result[0]["person_name"] == "Alice"


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
