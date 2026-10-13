import streamlit as st

from app.pages import (
    attendance,
    audio_logs,
    face_database,
    realtime,
    transcripts,
    video,
    voice_enrollment,
)
from app.ui.status import get_runtime_status
from face_voice_rec.config import initialize_runtime_paths
from face_voice_rec.storage.database import init_db


def main() -> None:
    st.set_page_config(
        page_title="AI Face + Audio Recognition",
        page_icon="🎥",
        layout="wide",
    )

    initialize_runtime_paths()
    init_db()

    st.title("🎥 AI Face + Active Speaker Recognition")

    runtime_status = get_runtime_status()
    st.caption(f"Audio models (PyTorch): {runtime_status['torch']}")
    st.caption(f"Face models (ONNX Runtime): {runtime_status['face']}")

    tabs = st.tabs(
        [
            "📹 Realtime",
            "🎬 Video",
            "👤 Face Database",
            "🔊 Audio Logs",
            "✅ Attendance",
            "📝 Transcripts",
            "🎙️ Voice Enrollment",
        ]
    )

    with tabs[0]:
        realtime.render()

    with tabs[1]:
        video.render()

    with tabs[2]:
        face_database.render()

    with tabs[3]:
        audio_logs.render()

    with tabs[4]:
        attendance.render()

    with tabs[5]:
        transcripts.render()

    with tabs[6]:
        voice_enrollment.render()


if __name__ == "__main__":
    main()
