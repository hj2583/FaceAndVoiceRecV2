from pathlib import PurePath

import pandas as pd


ATTENDANCE_COLUMNS = [
    "Person",
    "Source",
    "Observed (UTC)",
    "Video",
    "Source Ref",
    "Clip Time",
    "Confidence",
    "Track ID",
]


def _row_get(row, key, default=None):
    if hasattr(row, "keys") and key in row.keys():
        return row[key]
    if hasattr(row, "get"):
        return row.get(key, default)
    return default


def filter_visible_attendance(rows):
    """Exclude rows that do not represent a valid identified attendance event."""
    visible = []
    for row in rows:
        person_id = _row_get(row, "person_id")
        person_name = str(_row_get(row, "person_name") or "").strip()
        confidence = _row_get(row, "confidence")

        if person_id is None:
            continue
        if not person_name or person_name.lower() == "unknown":
            continue
        if confidence is None:
            continue

        visible.append(row)

    return visible


def format_clip_offset(media_offset_ms):
    if media_offset_ms is None:
        return ""

    total_seconds = max(0, int(float(media_offset_ms) // 1000))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _video_name(source_ref):
    if not source_ref:
        return ""
    normalized = str(source_ref).replace("\\", "/")
    return PurePath(normalized).name


def build_attendance_frame(rows):
    table_rows = []

    for row in rows:
        source = _row_get(row, "source") or ""
        source_ref = _row_get(row, "source_ref") or ""
        observed_utc = ""
        video = ""
        clip_time = ""

        if source == "realtime":
            raw_observed = _row_get(row, "observed_at_utc")
            observed_utc = raw_observed or ""
        elif source == "video":
            media_offset_ms = _row_get(row, "media_offset_ms")
            video = _video_name(source_ref)
            clip_time = format_clip_offset(media_offset_ms)

        person = _row_get(row, "person_name")
        confidence = _row_get(row, "confidence")
        track_id = _row_get(row, "track_id")

        table_rows.append(
            {
                "Person": person or "",
                "Source": source,
                "Observed (UTC)": observed_utc,
                "Video": video,
                "Source Ref": source_ref,
                "Clip Time": clip_time,
                "Confidence": confidence if confidence is not None else "",
                "Track ID": track_id if track_id is not None else "",
            }
        )

    if not table_rows:
        return pd.DataFrame(columns=ATTENDANCE_COLUMNS)

    return pd.DataFrame(table_rows, columns=ATTENDANCE_COLUMNS)


def _escape_csv_formula_cell(value):
    if isinstance(value, str) and value and value[0] in ("=", "+", "-", "@"):
        return "'" + value
    return value


def attendance_frame_to_csv(frame):
    safe_frame = frame.copy()
    for column in safe_frame.columns:
        safe_frame[column] = safe_frame[column].map(_escape_csv_formula_cell)

    csv_text = safe_frame.to_csv(index=False)
    return csv_text.encode("utf-8-sig")
