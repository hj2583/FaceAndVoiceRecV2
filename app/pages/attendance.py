from datetime import datetime

import streamlit as st

from app.ui.attendance_view import (
    attendance_frame_to_csv,
    build_attendance_frame,
    build_realtime_attendance_range,
    filter_visible_attendance,
    initialize_realtime_attendance_state,
    video_source_options,
)
from face_voice_rec.storage.database import fetch_attendance, list_persons


def render() -> None:
    st.header("✅ Attendance")

    source_mode = st.selectbox(
        "Attendance source",
        ["Realtime", "Uploaded video"],
    )

    persons = list_persons()
    person_options = {"All persons": None}
    for person_id, person_name, _created_at, _updated_at in persons:
        person_options[f"{person_name} (#{person_id})"] = person_id

    person_label = st.selectbox("Person", list(person_options.keys()))
    person_id_filter = person_options[person_label]

    rows = []

    if source_mode == "Realtime":
        now_local = datetime.now().astimezone().replace(microsecond=0)
        initialize_realtime_attendance_state(st.session_state, now_local)
        start_column, end_column = st.columns(2)

        with start_column:
            st.caption("Start (machine-local time)")
            start_date = st.date_input(
                "Start date",
                key="attendance_start_date",
            )
            start_time = st.time_input(
                "Start time",
                key="attendance_start_time",
            )

        with end_column:
            st.caption("End (machine-local time)")
            up_to_now = st.checkbox(
                "Up to now",
                key="attendance_up_to_now",
            )
            end_date = st.date_input(
                "End date",
                key="attendance_end_date",
                disabled=up_to_now,
            )
            end_time = st.time_input(
                "End time",
                key="attendance_end_time",
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

        rows = fetch_attendance(
            source="realtime",
            person_id=person_id_filter,
            observed_from_utc=observed_from_utc,
            observed_to_utc=observed_to_utc,
        )
    else:
        video_rows = fetch_attendance(source="video")
        options, labels = video_source_options([row["source_ref"] for row in video_rows])

        if not options:
            st.info("No uploaded video attendance observations yet.")
            return

        selected_source_ref = st.selectbox(
            "Video source",
            options,
            format_func=lambda value: labels.get(value, value),
        )

        rows = fetch_attendance(
            source="video",
            person_id=person_id_filter,
            source_ref=selected_source_ref,
        )

    visible_rows = filter_visible_attendance(rows)

    frame = build_attendance_frame(visible_rows)

    if frame.empty:
        st.info("No attendance observations match the current filters.")
        return

    st.dataframe(frame, width="stretch")
    st.download_button(
        "⬇️ Export CSV",
        data=attendance_frame_to_csv(frame),
        file_name="attendance.csv",
        mime="text/csv",
    )
