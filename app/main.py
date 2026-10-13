import numpy as np
import sys


def _install_numpy_compat_aliases() -> None:
    # NumPy 1.26 removes legacy aliases some dependencies still expect.
    if not hasattr(np, "long"):
        np.long = np.int_
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "complex"):
        np.complex = complex


_install_numpy_compat_aliases()

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


def render_application() -> None:
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


def main() -> None:
    st.set_page_config(
        page_title="AI Face + Audio Recognition",
        page_icon="🎥",
        layout="wide",
    )

    initialize_runtime_paths()
    init_db()

    navigation = st.navigation(
        [
            st.Page(
                render_application,
                title="AI Face + Active Speaker Recognition",
                default=True,
            )
        ],
        position="hidden",
    )
    navigation.run()


def _is_running_via_streamlit() -> bool:
    argv = " ".join(sys.argv).lower()
    if "streamlit" in argv and " run " in f" {argv} ":
        return True

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    return get_script_run_ctx() is not None


if __name__ == "__main__" or _is_running_via_streamlit():
    main()
