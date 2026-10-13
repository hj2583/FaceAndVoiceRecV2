import numpy as np

# NumPy 1.26 removes some legacy aliases used by older third-party code.
# Keep a compatibility layer in place before importing pandas/streamlit.
if not hasattr(np, "long"):
    np.long = np.int_
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "complex"):
    np.complex = complex

import streamlit as st

from app.ui.renderers import build_audio_log_frame
from face_voice_rec.storage.database import fetch_audio_logs


def _fmt_time(seconds: float) -> str:
    seconds = float(seconds)
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def render() -> None:
    st.header("🔊 Audio Logs")

    rows = fetch_audio_logs()

    if not rows:
        st.info("No audio logs yet.")
        return

    st.caption(
        "Speaker scores are similarity or activity scores, not calibrated "
        "identification probabilities. Video active-speaker values are lip-motion "
        "activity scores; realtime values may be voice cosine similarities."
    )
    data = []

    for row in rows:
        _id, start, end, name, confidence, source, transcript, track_id = row

        data.append({
            "Start": _fmt_time(start),
            "End": _fmt_time(end),
            "Duration": round(float(end) - float(start), 2),
            "Person": name or "Unknown",
            "Track": track_id,
            "Speaker score (not probability)": round(float(confidence), 3),
            "Source": source,
            "Transcript": transcript or "",
        })

    st.dataframe(
        build_audio_log_frame(data),
        width="stretch",
    )
