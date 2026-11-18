from datetime import datetime
from pathlib import Path

import streamlit as st

from app.ui.attendance_view import build_realtime_attendance_range
from app.ui.renderers import (
    audio_log_frame_to_csv,
    build_audio_log_frame,
    format_audio_log_time,
    initialize_realtime_audio_log_state,
)
from face_voice_rec.storage.database import (
    fetch_audio_logs,
    list_audio_log_video_sources,
    list_persons,
)


def render() -> None:
    st.header("🔊 Audio Logs")

    source_mode = st.selectbox(
        "Audio log source",
        ["Realtime", "Uploaded video"],
    )

    persons = list_persons()
    person_options = {"All persons": None}
    for person_id, person_name, _created_at, _updated_at in persons:
        person_options[f"{person_name} (#{person_id})"] = person_id
    person_options["Unknown / Unresolved"] = "UNKNOWN"

    person_label = st.selectbox("Person", list(person_options.keys()))
    person_id_filter = person_options[person_label]

    rows = []

    if source_mode == "Realtime":
        now_local = datetime.now().astimezone().replace(microsecond=0)
        initialize_realtime_audio_log_state(st.session_state, now_local)
        start_column, end_column = st.columns(2)

        with start_column:
            st.caption("Start (machine-local time)")
            start_date = st.date_input(
                "Start date",
                key="audio_logs_start_date",
            )
            start_time = st.time_input(
                "Start time",
                key="audio_logs_start_time",
            )

        with end_column:
            st.caption("End (machine-local time)")
            up_to_now = st.checkbox(
                "Up to now",
                key="audio_logs_up_to_now",
            )
            end_date = st.date_input(
                "End date",
                key="audio_logs_end_date",
                disabled=up_to_now,
            )
            end_time = st.time_input(
                "End time",
                key="audio_logs_end_time",
                disabled=up_to_now,
            )
            if up_to_now:
                st.caption("End time updates to the current time when the view reruns.")

        try:
            observed_from_utc, observed_to_utc = build_realtime_attendance_range(
                start_date,
                start_time,
                up_to_now=up_to_now,
                end_date=None if up_to_now else end_date,
                end_time=None if up_to_now else end_time,
                now_local=now_local,
            )
        except ValueError:
            st.warning("Start must be earlier than or equal to end.")
            return

        from_ts = datetime.fromisoformat(observed_from_utc).timestamp()
        to_ts = datetime.fromisoformat(observed_to_utc).timestamp()

        rows = fetch_audio_logs(
            source="realtime",
            person_id=person_id_filter,
            from_time=from_ts,
            to_time=to_ts,
        )
    else:
        video_options = {"All videos": None}
        for video_path in list_audio_log_video_sources():
            label = Path(video_path).name or video_path
            if label in video_options:
                label = video_path
            video_options[label] = video_path
        video_label = st.selectbox("Video", list(video_options))
        rows = fetch_audio_logs(
            source="video",
            person_id=person_id_filter,
            source_ref=video_options[video_label],
        )

    if not rows:
        st.info("No audio logs match the current filters.")
        return

    st.caption(
        "Speaker scores are similarity or activity scores, not calibrated "
        "identification probabilities. Video active-speaker values are lip-motion "
        "activity scores; realtime values may be voice cosine similarities."
    )
    is_realtime = source_mode == "Realtime"
    data = []

    for row in rows:
        _id, start, end, name, confidence, source, transcript, track_id = row

        data.append({
            "Start": format_audio_log_time(start, is_realtime=is_realtime),
            "End": format_audio_log_time(end, is_realtime=is_realtime),
            "Duration (s)": round(max(0.0, float(end) - float(start)), 2),
            "Person": name or "Unknown",
            "Track": track_id,
            "Speaker score (not probability)": round(float(confidence), 3) if confidence is not None else 0.0,
            "Source": source,
            "Transcript": transcript or "",
        })

    frame = build_audio_log_frame(data)

    if frame.empty:
        st.info("No audio logs match the current filters.")
        return

    st.dataframe(
        frame,
        width="stretch",
    )
    st.download_button(
        "⬇️ Export CSV",
        data=audio_log_frame_to_csv(frame),
        file_name="audio_logs.csv",
        mime="text/csv",
    )
