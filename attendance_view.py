from pathlib import PurePath

import pandas as pd


ATTENDANCE_COLUMNS = [
    "Person",
    "Source",
    "Observed (UTC)",
    "Video",
    "Clip Time",
    "Confidence",
    "Track ID",
]


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
        source = (row["source"] or "") if "source" in row.keys() else (row.get("source") or "")
        observed_utc = ""
        video = ""
        clip_time = ""

        if source == "realtime":
            raw_observed = row["observed_at_utc"] if "observed_at_utc" in row.keys() else row.get("observed_at_utc")
            observed_utc = raw_observed or ""
        elif source == "video":
            source_ref = row["source_ref"] if "source_ref" in row.keys() else row.get("source_ref")
            media_offset_ms = row["media_offset_ms"] if "media_offset_ms" in row.keys() else row.get("media_offset_ms")
            video = _video_name(source_ref)
            clip_time = format_clip_offset(media_offset_ms)

        person = row["person_name"] if "person_name" in row.keys() else row.get("person_name")
        confidence = row["confidence"] if "confidence" in row.keys() else row.get("confidence")
        track_id = row["track_id"] if "track_id" in row.keys() else row.get("track_id")

        table_rows.append(
            {
                "Person": person or "",
                "Source": source,
                "Observed (UTC)": observed_utc,
                "Video": video,
                "Clip Time": clip_time,
                "Confidence": confidence if confidence is not None else "",
                "Track ID": track_id if track_id is not None else "",
            }
        )

    if not table_rows:
        return pd.DataFrame(columns=ATTENDANCE_COLUMNS)

    return pd.DataFrame(table_rows, columns=ATTENDANCE_COLUMNS)


def attendance_frame_to_csv(frame):
    csv_text = frame.to_csv(index=False)
    return csv_text.encode("utf-8-sig")
