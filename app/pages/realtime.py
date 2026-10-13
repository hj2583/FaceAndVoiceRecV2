from pathlib import Path

import streamlit as st

from app.ui.renderers import build_audio_log_frame
from face_voice_rec.config import LOG_DIR
from face_voice_rec.pipelines.realtime.realtime_launcher import launch_realtime
from face_voice_rec.storage.database import fetch_audio_logs


def _fmt_time(seconds: float) -> str:
    seconds = float(seconds)
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def render() -> None:
    st.header("📹 Realtime Face Recognition")

    st.info(
        "Realtime mode runs the local webcam and microphone in a separate process. "
        "Press Q or ESC in the camera window to stop it."
    )

    camera = st.number_input(
        "Camera index",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
    )

    if st.button("▶ Start Realtime Recognition", type="primary"):
        with st.spinner("Starting realtime recognition..."):
            result = launch_realtime(
                camera=int(camera),
                cwd=Path(__file__).resolve().parents[2],
                log_dir=LOG_DIR,
            )

        if result.started:
            st.success(
                "Realtime recognition started. A camera window should appear. "
                "Press Q/ESC in that window to stop."
            )
        else:
            st.error(f"Realtime recognition failed to start:\n\n{result.error}")

    st.markdown("---")
    st.subheader("Recent Audio Logs")

    rows = fetch_audio_logs("realtime")

    if rows:
        st.caption(
            "Speaker scores are similarity or activity scores, not calibrated "
            "identification probabilities."
        )
        data = []
        for row in rows:
            _id, start, end, name, confidence, source, transcript, track_id = row
            data.append({
                "Start": _fmt_time(start),
                "End": _fmt_time(end),
                "Person": name or "Unknown",
                "Track": track_id,
                "Speaker score (not probability)": round(confidence, 3),
                "Transcript": transcript or "",
            })

        st.dataframe(build_audio_log_frame(data), width="stretch")
    else:
        st.info("No realtime speaking logs yet.")
