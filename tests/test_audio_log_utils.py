import pyarrow as pa

from audio_log_utils import build_audio_log_frame


def test_track_column_uses_nullable_integer_dtype_for_arrow():
    frame = build_audio_log_frame([
        {"Person": "Alice", "Track": 12},
        {"Person": "Unknown", "Track": None},
    ])

    assert str(frame["Track"].dtype) == "Int64"
    arrow_table = pa.Table.from_pandas(frame, preserve_index=False)
    assert arrow_table.column("Track").to_pylist() == [12, None]