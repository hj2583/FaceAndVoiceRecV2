import json
from pathlib import Path, PureWindowsPath

import pandas as pd
import streamlit as st

from face_voice_rec.config import INITIAL_VIDEO_DIR, LOG_DIR, TRACKED_VIDEO_DIR
from face_voice_rec.pipelines.video.video_workflow import (
    TranscriptStageError,
    process_video_and_transcribe,
)
from face_voice_rec.domain.transcription.transcription_core import process_meeting_transcription
from face_voice_rec.storage.database import list_meeting_introductions


def render() -> None:
    st.header("🎬 Upload Video Recognition")

    uploaded = st.file_uploader(
        "Upload a video or MP3 audio file",
        type=["mp4", "avi", "mov", "mkv", "mp3"],
    )

    if uploaded:
        safe_name = PureWindowsPath(uploaded.name.replace("/", "\\")).name
        if not safe_name or Path(safe_name).suffix.lower() not in {
            ".mp4", ".avi", ".mov", ".mkv", ".mp3",
        }:
            st.error("Invalid file name or type.")
            return
        input_path = INITIAL_VIDEO_DIR / safe_name
        with open(input_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.success(f"Saved: {input_path.name}")

    media_files = sorted(
        [
            p.name
            for p in INITIAL_VIDEO_DIR.iterdir()
            if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".mp3"}
        ]
    )

    if not media_files:
        st.info("No videos or MP3 audio files available.")
        return

    selected = st.selectbox("Select media", media_files)

    input_path = INITIAL_VIDEO_DIR / selected
    if input_path.suffix.lower() == ".mp3":
        st.audio(input_path.read_bytes(), format="audio/mp3")
        st.info(
            "MP3 files use voice recognition and transcription only. "
            "Face detection and tracked-video output require a video file."
        )
        if st.button("🎙️ Recognize Speakers & Transcribe Audio", type="primary"):
            with st.spinner("Recognizing speakers and transcribing audio..."):
                try:
                    meeting_id = process_meeting_transcription(input_path)
                    st.success(
                        f"Audio recognition completed. Open meeting {meeting_id} "
                        "in the Transcripts tab to review the results."
                    )
                except Exception as exc:
                    st.error(f"Audio recognition failed: {exc}")
        return

    output_path = TRACKED_VIDEO_DIR / f"tracked_{input_path.stem}.mp4"
    log_path = LOG_DIR / f"{input_path.stem}.json"

    if st.button("🚀 Process Video + Transcript", type="primary"):
        progress = st.progress(0)
        status = st.empty()

        def update_progress(value):
            value = max(0.0, min(1.0, float(value)))
            progress.progress(value)
            status.write(f"Processing: {value * 100:.1f}%")

        def update_stage(message):
            status.write(message)

        try:
            meeting_id = process_video_and_transcribe(
                input_path,
                output_path,
                log_path,
                progress_callback=update_progress,
                stage_callback=update_stage,
            )
            if any(
                introduction[14] is not None
                for introduction in list_meeting_introductions(meeting_id)
            ):
                try:
                    from app.pages.face_database import get_face_index

                    get_face_index().reload()
                except Exception as error:
                    st.warning(f"Face identity index refresh failed: {error}")
            progress.progress(1.0)
            status.success("Processing and transcription completed.")
            st.success(
                f"✅ Meeting {meeting_id} processed successfully.\n"
                f"Tracked video: {output_path}\n"
                "Includes source audio when present and burned-in subtitles."
            )
        except TranscriptStageError as exc:
            progress.progress(1.0)
            if exc.stage == "transcription" and exc.meeting_id is not None:
                stage_note = "Transcript saved, but subtitle preparation failed before final subtitle burn-in."
            elif exc.stage == "transcription":
                stage_note = "Transcription failed before transcript persistence."
            elif exc.stage == "subtitle":
                stage_note = "Subtitle burn-in failed after transcription completed."
            elif exc.stage == "audio_mux" and exc.output_updated:
                stage_note = "Fallback output is a silent annotated video with no subtitles."
            elif exc.stage == "audio_mux":
                stage_note = "Fallback replacement failed; output path may be stale or absent."
            else:
                stage_note = "Finalization failed."

            if exc.output_updated:
                freshness_note = "Output path was updated in this run."
            else:
                freshness_note = "Output path may be stale or absent."

            meeting_note = f" (Meeting ID: {exc.meeting_id})" if exc.meeting_id else ""
            status.warning(
                f"Stage '{exc.stage}' failed{meeting_note}. {stage_note} {freshness_note}"
            )
            st.warning(
                f"**Stage:** {exc.stage}  \n"
                f"**Output path:** {exc.video_output_path}  \n"
                f"**Partial result:** {stage_note}  \n"
                f"**Output freshness:** {freshness_note}  \n"
                f"**Error:** {exc.cause}"
                + (f"  \n**Meeting ID:** {exc.meeting_id}" if exc.meeting_id else "")
            )
        except Exception as exc:
            st.exception(exc)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        with open(input_path, "rb") as f:
            st.video(f.read())

    with col2:
        st.subheader("Tracked")
        if output_path.exists():
            with open(output_path, "rb") as f:
                st.video(f.read())
        else:
            st.info("Run processing first.")

    if log_path.exists():
        st.subheader("Recognition / Speaking Logs")

        with open(log_path, "r", encoding="utf-8") as f:
            logs = json.load(f)

        speech = logs.get("speech", [])

        if speech:
            st.dataframe(
                pd.DataFrame(speech),
                width="stretch",
            )
        else:
            st.info("No active speaker segments detected.")
