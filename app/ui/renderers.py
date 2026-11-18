from datetime import datetime, timedelta, timezone
import pandas as pd


def build_audio_log_frame(rows):
    frame = pd.DataFrame(rows)
    if "Track" in frame.columns:
        frame["Track"] = pd.array(frame["Track"], dtype="Int64")
    return frame


def _escape_csv_formula_cell(value):
    if isinstance(value, str) and value and value[0] in ("=", "+", "-", "@"):
        return "'" + value
    return value


def audio_log_frame_to_csv(frame):
    safe_frame = frame.copy()
    for column in safe_frame.columns:
        safe_frame[column] = safe_frame[column].map(_escape_csv_formula_cell)

    csv_text = safe_frame.to_csv(index=False)
    return csv_text.encode("utf-8-sig")


def format_audio_log_time(seconds: float, is_realtime: bool = False) -> str:
    seconds = float(seconds)
    if is_realtime and seconds > 100_000_000:
        return (
            datetime.fromtimestamp(seconds, timezone.utc)
            .astimezone()
            .strftime("%Y-%m-%d %H:%M:%S")
        )

    total_secs = max(0, int(seconds))
    hours, remainder = divmod(total_secs, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def initialize_realtime_audio_log_state(state, now_local: datetime) -> None:
    default_start = now_local - timedelta(hours=24)
    defaults = {
        "audio_logs_start_date": default_start.date(),
        "audio_logs_start_time": default_start.time().replace(tzinfo=None, microsecond=0),
        "audio_logs_up_to_now": True,
        "audio_logs_end_date": now_local.date(),
        "audio_logs_end_time": now_local.time().replace(tzinfo=None, microsecond=0),
    }
    for key, value in defaults.items():
        if key not in state:
            state[key] = value
