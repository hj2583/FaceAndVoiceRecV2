import json
import os
import sqlite3
import tempfile
import uuid
import wave
from pathlib import Path

from PIL import Image

import numpy as np

# NumPy 1.26 removes some legacy aliases used by older third-party code.
# Keep a compatibility layer in place before importing pandas/streamlit.
if not hasattr(np, "long"):
    np.long = np.int_
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "complex"):
    np.complex = complex

import pandas as pd
import streamlit as st

from audio_core import read_wav_pcm
from config import (
    DB_PATH,
    INITIAL_VIDEO_DIR,
    KNOWN_FACES_DIR,
    LOG_DIR,
    TRACKED_VIDEO_DIR,
    UNKNOWN_FACES_DIR,
)
from database import (
    add_voice_embedding,
    create_person,
    delete_unknown_voice,
    fetch_audio_logs,
    init_db,
    list_persons,
    list_unknowns,
    list_unknown_samples,
    list_unknown_voice_samples,
    list_unknown_voices,
    delete_low_quality_unknowns,
    delete_unknown,
    resolve_unknown,
    resolve_unknown_voice,
)
from face_core import FaceIndex
from realtime_launcher import launch_realtime
from video_processor import process_video_pipeline
from transcription_core import extract_audio_from_video, process_meeting_transcription
import voice_core


st.set_page_config(
    page_title="AI Face + Audio Recognition",
    page_icon="🎥",
    layout="wide",
)

init_db()


@st.cache_resource
def get_face_index():
    return FaceIndex()


def fmt_time(seconds):
    seconds = float(seconds)
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def get_runtime_mode():
    try:
        import torch

        if torch.cuda.is_available():
            return "GPU (CUDA)"
    except Exception:
        pass

    try:
        from face_backend import get_cuda_providers

        provider = get_cuda_providers()[0]
        if provider == "CUDAExecutionProvider":
            return "GPU (CUDA)"
        if provider == "CPUExecutionProvider":
            return "CPU"
        return provider
    except Exception:
        return "CPU"


def render_realtime():
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
                script=Path(__file__).parent / "realtime.py",
                camera=int(camera),
                cwd=Path(__file__).parent,
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
        data = []
        for row in rows:
            _id, start, end, name, confidence, source, transcript = row
            data.append({
                "Start": fmt_time(start),
                "End": fmt_time(end),
                "Person": name or "Unknown",
                "Confidence": round(confidence, 3),
                "Transcript": transcript or "",
            })

        st.dataframe(pd.DataFrame(data), use_container_width=True)
    else:
        st.info("No realtime speaking logs yet.")


def render_video():
    st.header("🎬 Upload Video Recognition")

    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded:
        input_path = INITIAL_VIDEO_DIR / uploaded.name
        with open(input_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.success(f"Saved: {input_path.name}")

    videos = sorted(
        [
            p.name
            for p in INITIAL_VIDEO_DIR.iterdir()
            if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        ]
    )

    if not videos:
        st.info("No videos available.")
        return

    selected = st.selectbox("Select video", videos)

    input_path = INITIAL_VIDEO_DIR / selected
    output_path = TRACKED_VIDEO_DIR / f"tracked_{input_path.stem}.mp4"
    log_path = LOG_DIR / f"{input_path.stem}.json"

    if st.button("🚀 Process Video", type="primary"):
        progress = st.progress(0)
        status = st.empty()

        def update_progress(value):
            value = max(0.0, min(1.0, float(value)))
            progress.progress(value)
            status.write(f"Processing: {value * 100:.1f}%")

        try:
            process_video_pipeline(
                input_path,
                output_path,
                log_path,
                progress_callback=update_progress,
            )
            progress.progress(1.0)
            status.success("Processing completed.")
        except Exception as exc:
            st.exception(exc)

    if st.button("📝 Transcribe Selected Video"):
        with st.spinner("Extracting audio and transcribing..."):
            try:
                meeting_id = process_meeting_transcription(input_path)
                st.success(f"Transcription completed for meeting {meeting_id}.")
            except Exception as exc:
                st.error(f"Transcription failed: {exc}")

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
                use_container_width=True,
            )
        else:
            st.info("No active speaker segments detected.")


def render_database():
    st.header("👤 Face Database")

    index = get_face_index()

    persons = list_persons()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Known Persons", len(persons))

    with col2:
        st.metric("Face Embeddings", len(index.ids))

    # ============================================================
    # KNOWN PERSONS
    # ============================================================

    if persons:
        st.subheader("Known Persons")

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Person ID": p[0],
                        "Name": p[1],
                        "Created": p[2],
                    }
                    for p in persons
                ]
            ),
            use_container_width=True,
        )

    # ============================================================
    # ADD PERSON
    # ============================================================

    st.markdown("---")
    st.subheader("➕ Add a Person")

    with st.form("add_person"):
        name = st.text_input("Person name")
        submitted = st.form_submit_button("Create Person")

        if submitted:
            if not name.strip():
                st.error("Name cannot be empty.")
            else:
                create_person(name)
                st.success(f"Created person: {name}")
                st.rerun()

    # ============================================================
    # UNKNOWN FACES
    # ============================================================

    st.markdown("---")
    st.subheader("❓ Pending Unknown People")

    unknowns = list_unknowns()

    if not unknowns:
        st.success("No unresolved unknown faces.")
        return

    st.info(
        "Review the captured samples below. "
        "You can assign an unknown face to an existing person "
        "or create a new person from the captured face."
    )

    if st.button(
        "🗑️ Delete Unknown Faces Below 40% Quality",
        key="delete_low_quality_unknowns",
        help="Permanently removes unresolved unknown faces whose best sample quality is below 0.40.",
    ):
        deleted_ids = delete_low_quality_unknowns(0.40)
        if deleted_ids:
            st.success(f"Deleted {len(deleted_ids)} low-quality unknown face(s).")
            get_face_index().rebuild()
            st.rerun()
        else:
            st.info("No unresolved unknown faces below 40% quality were found.")

    people = list_persons()
    options = {p[1]: p[0] for p in people}

    for (
        unknown_id,
        label,
        image_path,
        embedding_path,
        created,
        _resolved,
    ) in unknowns:

        # --------------------------------------------------------
        # Get samples belonging to this unknown record
        # --------------------------------------------------------

        samples = list_unknown_samples(unknown_id)

        # --------------------------------------------------------
        # Build sample list
        #
        # New records should have unknown_samples.
        #
        # Older records may not, so fall back to the main
        # image/embedding stored in unknown_tracks.
        # --------------------------------------------------------

        sample_records = []

        for sample in samples:
            (
                sample_id,
                sample_image_path,
                sample_embedding_path,
                quality,
                angle_score,
                sample_created,
            ) = sample

            if (
                sample_image_path
                and os.path.exists(sample_image_path)
                and sample_embedding_path
                and os.path.exists(sample_embedding_path)
            ):
                sample_records.append({
                    "sample_id": sample_id,
                    "image_path": sample_image_path,
                    "embedding_path": sample_embedding_path,
                    "quality": float(quality),
                    "angle_score": float(angle_score),
                    "created": sample_created,
                })

        # --------------------------------------------------------
        # Fallback for older records
        # --------------------------------------------------------

        if not sample_records:
            if (
                image_path
                and str(image_path).lower() != "none"
                and os.path.exists(image_path)
                and embedding_path
                and os.path.exists(embedding_path)
            ):
                sample_records.append({
                    "sample_id": None,
                    "image_path": image_path,
                    "embedding_path": embedding_path,
                    "quality": 0.0,
                    "angle_score": 0.0,
                    "created": created,
                })

        # --------------------------------------------------------
        # Sort highest-quality samples first
        # --------------------------------------------------------

        sample_records.sort(
            key=lambda x: x["quality"],
            reverse=True,
        )

        sample_count = len(sample_records)

        best_image = (
            sample_records[0]["image_path"]
            if sample_records
            else None
        )

        best_quality = (
            sample_records[0]["quality"]
            if sample_records
            else 0.0
        )

        # ========================================================
        # UNKNOWN CARD
        # ========================================================

        with st.container(border=True):

            st.markdown(
                f"### ❓ Unknown #{unknown_id}"
            )

            st.caption(
                f"Track: `{label}`"
            )

            st.caption(
                f"Captured: {sample_count} sample"
                f"{'s' if sample_count != 1 else ''}"
                f"  •  Created: {created}"
            )

            if st.button(
                "🗑️ Delete Unknown",
                key=f"delete_unknown_{unknown_id}",
                help="Permanently delete this unresolved unknown face and its saved samples.",
            ):
                if delete_unknown(unknown_id):
                    get_face_index().rebuild()
                    st.rerun()
                else:
                    st.warning("This unknown face is no longer unresolved.")

            # ----------------------------------------------------
            # BEST IMAGE
            # ----------------------------------------------------

            if best_image and os.path.exists(best_image):

                st.image(
                    best_image,
                    caption=f"Best sample • Quality {best_quality:.2f}",
                    width=300,
                )

            else:
                st.warning(
                    "No valid face image is available for this "
                    "unknown record."
                )

            # ----------------------------------------------------
            # ALL SAMPLES
            # ----------------------------------------------------

            if sample_records:

                st.markdown(
                    f"**Face Samples ({sample_count})**"
                )

                # Maximum 6 per row
                columns = st.columns(
                    min(6, max(1, sample_count))
                )

                for i, sample in enumerate(sample_records):

                    with columns[i % len(columns)]:
                        
                        if os.path.exists(sample["image_path"]):
                            try:
                                img = Image.open(sample["image_path"]).convert("RGB")

                                # Keep every thumbnail visually consistent
                                img.thumbnail((180, 180))

                                # Create a fixed-size white canvas
                                canvas = Image.new(
                                    "RGB",
                                    (180, 180),
                                    "white",
                                )

                                x = (180 - img.width) // 2
                                y = (180 - img.height) // 2

                                canvas.paste(img, (x, y))

                                st.image(
                                    canvas,
                                    width=180,
                                )

                                st.caption(
                                    f"#{i + 1} • Quality: "
                                    f"{sample['quality']:.2f}"
                                )

                            except Exception as exc:
                                st.warning(
                                    f"Could not display sample: {exc}"
                                )

            else:

                st.info(
                    "No individual samples were found. "
                    "Using the original unknown embedding."
                )

            st.markdown("---")

            # ====================================================
            # ASSIGN TO EXISTING PERSON
            # ====================================================

            if options:

                st.markdown("#### 👤 Assign to Existing Person")

                selected_name = st.selectbox(
                    "Select person",
                    ["-- Select --"] + list(options.keys()),
                    key=f"unknown_person_{unknown_id}",
                )

                if st.button(
                    "✅ Assign Existing Person",
                    key=f"assign_{unknown_id}",
                    type="primary",
                ):

                    if selected_name == "-- Select --":

                        st.warning(
                            "Please select a person first."
                        )

                    else:

                        selected_person_id = options[selected_name]

                        embeddings_added = 0

                        # ------------------------------------------------
                        # Add every available sample embedding
                        # ------------------------------------------------

                        for sample in sample_records:

                            try:

                                emb = np.load(
                                    sample["embedding_path"],
                                    allow_pickle=True,
                                )
                                if emb.dtype == object:
                                    emb = np.asarray(emb.tolist(), dtype=np.float32)

                                if emb is None:
                                    continue

                                from face_core import (
                                    save_embedding_for_person,
                                )

                                save_embedding_for_person(
                                    selected_name,
                                    emb,
                                    image_path=sample["image_path"],
                                    quality=sample["quality"],
                                )

                                embeddings_added += 1

                            except Exception as exc:

                                st.warning(
                                    f"Could not add sample "
                                    f"{sample['sample_id']}: {exc}"
                                )

                        # ------------------------------------------------
                        # Fallback: use original unknown embedding
                        # ------------------------------------------------

                        if embeddings_added == 0:

                            if (
                                embedding_path
                                and os.path.exists(embedding_path)
                            ):

                                try:

                                    emb = np.load(
                                        embedding_path,
                                        allow_pickle=True,
                                    )
                                    if emb.dtype == object:
                                        emb = np.asarray(emb.tolist(), dtype=np.float32)

                                    from face_core import (
                                        save_embedding_for_person,
                                    )

                                    save_embedding_for_person(
                                        selected_name,
                                        emb,
                                        image_path=best_image,
                                        quality=best_quality,
                                    )

                                    embeddings_added = 1

                                except Exception as exc:

                                    st.error(
                                        "Could not add the unknown "
                                        f"embedding: {exc}"
                                    )

                        if embeddings_added > 0:

                            resolve_unknown(
                                unknown_id,
                                selected_person_id,
                            )

                            # Rebuild NumPy face index so the new
                            # embeddings become available immediately.
                            get_face_index().rebuild()

                            st.success(
                                f"{label} assigned to "
                                f"{selected_name}. "
                                f"{embeddings_added} face sample"
                                f"{'s' if embeddings_added != 1 else ''} "
                                f"added to the face database."
                            )

                            st.rerun()

                        else:

                            st.error(
                                "No valid face embedding could be "
                                "added."
                            )

            else:

                st.info(
                    "No existing persons available. "
                    "Create a person below."
                )

            # ====================================================
            # CREATE NEW PERSON
            # ====================================================

            st.markdown("#### 🆕 Create New Person")

            new_name = st.text_input(
                "New person's name",
                key=f"new_person_name_{unknown_id}",
            )

            if st.button(
                "➕ Create New Person",
                key=f"create_person_{unknown_id}",
            ):

                if not new_name.strip():

                    st.warning(
                        "Please enter a name."
                    )

                else:

                    new_name = new_name.strip()

                    try:

                        # --------------------------------------------
                        # Create person first
                        # --------------------------------------------

                        new_person_id = create_person(
                            new_name
                        )

                        embeddings_added = 0

                        # --------------------------------------------
                        # Add all available unknown samples
                        # --------------------------------------------

                        from face_core import (
                            load_or_extract_embedding,
                            save_embedding_for_person,
                        )

                        for sample in sample_records:

                            try:

                                emb = load_or_extract_embedding(
                                    sample["embedding_path"],
                                    sample["image_path"],
                                )
                                if emb is None:
                                    raise ValueError("Could not extract a valid face embedding from sample image")

                                save_embedding_for_person(
                                    new_name,
                                    emb,
                                    image_path=sample["image_path"],
                                    quality=sample["quality"],
                                )

                                embeddings_added += 1

                            except Exception as exc:

                                st.warning(
                                    f"Could not add sample "
                                    f"{sample['sample_id']}: {exc}"
                                )

                        # --------------------------------------------
                        # Fallback to original embedding
                        # --------------------------------------------

                        if embeddings_added == 0:

                            if (
                                embedding_path
                                and os.path.exists(embedding_path)
                            ):

                                emb = load_or_extract_embedding(
                                    embedding_path,
                                    best_image,
                                )
                                if emb is None:
                                    raise ValueError("Could not extract a valid face embedding from sample image")

                                save_embedding_for_person(
                                    new_name,
                                    emb,
                                    image_path=best_image,
                                    quality=best_quality,
                                )

                                embeddings_added = 1

                        if embeddings_added > 0:

                            resolve_unknown(
                                unknown_id,
                                new_person_id,
                            )

                            get_face_index().rebuild()

                            st.success(
                                f"Created {new_name} and added "
                                f"{embeddings_added} face sample"
                                f"{'s' if embeddings_added != 1 else ''}."
                            )

                            st.rerun()

                        else:

                            st.error(
                                "Person was created, but no valid "
                                "face embedding was available."
                            )

                    except Exception as exc:

                        st.exception(exc)


def render_audio_logs():
    st.header("🔊 Audio Logs")

    rows = fetch_audio_logs()

    if not rows:
        st.info("No audio logs yet.")
        return

    data = []

    for row in rows:
        _id, start, end, name, confidence, source, transcript = row

        data.append({
            "Start": fmt_time(start),
            "End": fmt_time(end),
            "Duration": round(float(end) - float(start), 2),
            "Person": name or "Unknown",
            "Confidence": round(float(confidence), 3),
            "Source": source,
            "Transcript": transcript or "",
        })

    st.dataframe(
        pd.DataFrame(data),
        use_container_width=True,
    )


def render_transcripts():
    st.header("📝 Meeting Transcripts")
    with sqlite3.connect(DB_PATH) as connection:
        meetings = connection.execute(
            """
            SELECT meeting_id, video_path, transcription_status, error_log
            FROM meetings ORDER BY created_at DESC
            """
        ).fetchall()

    if not meetings:
        st.info("No meetings have been transcribed yet.")
        return

    options = {
        meeting_id: f"{Path(video_path).name} ({status})"
        for meeting_id, video_path, status, _error in meetings
    }
    selected_id = st.selectbox(
        "Meeting",
        list(options),
        format_func=lambda meeting_id: options[meeting_id],
    )

    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT speaker_label, start_ms, end_ms, text, confidence
            FROM transcription_segments
            WHERE meeting_id=? ORDER BY start_ms
            """,
            (selected_id,),
        ).fetchall()

    if not rows:
        st.info("This meeting has no transcript segments.")
        return

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Speaker": speaker,
                    "Start": fmt_time(start_ms / 1000),
                    "End": fmt_time(end_ms / 1000),
                    "Text": text,
                    "Confidence": confidence,
                }
                for speaker, start_ms, end_ms, text, confidence in rows
            ]
        ),
        use_container_width=True,
    )


def render_voice_enrollment():
    st.header("🎙️ Voice Enrollment")

    persons = {name: person_id for person_id, name, _created, _updated in list_persons()}

    st.subheader("Enroll a clean voice sample")
    selected_name = st.selectbox(
        "Person",
        ["-- Select --"] + list(persons.keys()),
        key="voice_enroll_person",
    )
    uploaded = st.file_uploader(
        "Upload a short voice sample (WAV or MP3)",
        type=["wav", "mp3"],
    )
    if st.button("➕ Add Voice Sample", key="add_voice_sample"):
        if selected_name == "-- Select --":
            st.warning("Please select a person first.")
        elif uploaded is None:
            st.warning("Please upload a voice sample first.")
        else:
            try:
                if uploaded.name.lower().endswith(".mp3"):
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        mp3_path = Path(tmp_dir) / "upload.mp3"
                        mp3_path.write_bytes(uploaded.getvalue())
                        wav_path = extract_audio_from_video(mp3_path, Path(tmp_dir) / "upload.wav")
                        pcm = read_wav_pcm(wav_path)
                else:
                    pcm = read_wav_pcm(uploaded)
            except (ValueError, wave.Error, EOFError) as error:
                st.error(f"Invalid audio file (expected 16kHz mono 16-bit PCM): {error}")
            except RuntimeError as error:
                st.error(f"Could not convert the uploaded MP3: {error}")
            else:
                embedding = voice_core.extract_voice_embedding(pcm)
                if embedding is None:
                    st.error("Could not extract a voice embedding from this sample.")
                else:
                    person_id = persons[selected_name]
                    embedding_dir = KNOWN_FACES_DIR.parent / "known_voices" / str(person_id)
                    embedding_dir.mkdir(parents=True, exist_ok=True)
                    embedding_path = embedding_dir / f"{person_id}_{uuid.uuid4().hex}.npy"
                    np.save(embedding_path, embedding)
                    add_voice_embedding(person_id, embedding_path, quality=1.0)
                    st.success(f"Voice sample added for {selected_name}.")
                    st.rerun()

    st.subheader("Unresolved unknown voices")
    unknown_voices = list_unknown_voices()
    if not unknown_voices:
        st.info("No unresolved unknown voices.")
        return

    for unknown_voice_id, label, embedding_path, created_at, _resolved in unknown_voices:
        with st.container(border=True):
            st.markdown(f"### ❓ {label} (#{unknown_voice_id})")
            sample_count = len(list_unknown_voice_samples(unknown_voice_id))
            st.caption(
                f"Samples: {sample_count}  •  Created: {created_at}"
            )

            if st.button("🗑️ Delete", key=f"delete_unknown_voice_{unknown_voice_id}"):
                if delete_unknown_voice(unknown_voice_id):
                    st.rerun()
                else:
                    st.warning("This unknown voice is no longer unresolved.")

            assign_name = st.selectbox(
                "Assign to person",
                ["-- Select --"] + list(persons.keys()),
                key=f"assign_voice_{unknown_voice_id}",
            )
            if st.button("✅ Assign", key=f"assign_voice_btn_{unknown_voice_id}"):
                if assign_name == "-- Select --":
                    st.warning("Please select a person first.")
                elif not os.path.exists(embedding_path):
                    st.error("The stored embedding for this unknown voice is missing.")
                else:
                    person_id = persons[assign_name]
                    add_voice_embedding(person_id, embedding_path, quality=0.7)
                    resolve_unknown_voice(unknown_voice_id, person_id)
                    st.success(f"{label} assigned to {assign_name}.")
                    st.rerun()


def main():
    st.title("🎥 AI Face + Active Speaker Recognition")

    runtime_mode = get_runtime_mode()
    if runtime_mode == "GPU (CUDA)":
        st.success(f"Runtime mode: {runtime_mode}")
    elif runtime_mode == "CPU":
        st.warning(f"Runtime mode: {runtime_mode}")
    else:
        st.info(f"Runtime mode: {runtime_mode}")

    # st.caption(
    #     "Shared ArcFace embeddings + SQLite database + NumPy similarity index. "
    #     "No FAISS."
    # )

    tabs = st.tabs(
        [
            "📹 Realtime",
            "🎬 Video",
            "👤 Face Database",
            "🔊 Audio Logs",
            "📝 Transcripts",
            "🎙️ Voice Enrollment",
        ]
    )

    with tabs[0]:
        render_realtime()

    with tabs[1]:
        render_video()

    with tabs[2]:
        render_database()

    with tabs[3]:
        render_audio_logs()

    with tabs[4]:
        render_transcripts()

    with tabs[5]:
        render_voice_enrollment()


if __name__ == "__main__":
    main()
