from datetime import date, datetime, time, timedelta, timezone
import pyarrow as pa

from app.ui.renderers import (
    audio_log_frame_to_csv,
    build_audio_log_frame,
    format_audio_log_time,
    initialize_realtime_audio_log_state,
)


def test_track_column_uses_nullable_integer_dtype_for_arrow():
    frame = build_audio_log_frame([
        {"Person": "Alice", "Track": 12},
        {"Person": "Unknown", "Track": None},
    ])

    assert str(frame["Track"].dtype) == "Int64"
    arrow_table = pa.Table.from_pandas(frame, preserve_index=False)
    assert arrow_table.column("Track").to_pylist() == [12, None]


def test_format_audio_log_time_video_and_realtime():
    # Video clip offsets
    assert format_audio_log_time(0.0) == "00:00"
    assert format_audio_log_time(65.4) == "01:05"
    assert format_audio_log_time(3665.0) == "1:01:05"

    # Realtime epoch timestamps (> 1e8)
    dt = datetime(2026, 10, 21, 14, 30, 0, tzinfo=timezone.utc)
    epoch_ts = dt.timestamp()
    formatted = format_audio_log_time(epoch_ts, is_realtime=True)
    expected_local = dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    assert formatted == expected_local


def test_audio_log_frame_to_csv_escapes_formula_cells():
    frame = build_audio_log_frame([
        {"Person": "=Alice", "Track": 1, "Transcript": "+secret"}
    ])
    csv_bytes = audio_log_frame_to_csv(frame)
    csv_text = csv_bytes.decode("utf-8-sig")
    assert "'=Alice" in csv_text
    assert "'+secret" in csv_text


def test_initialize_realtime_audio_log_state():
    state = {}
    now = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
    initialize_realtime_audio_log_state(state, now)

    assert state["audio_logs_start_date"] == (now - timedelta(hours=24)).date()
    assert state["audio_logs_start_time"] == time(12, 0)
    assert state["audio_logs_up_to_now"] is True
    assert state["audio_logs_end_date"] == now.date()
    assert state["audio_logs_end_time"] == time(12, 0)
