import io
import sqlite3
import tempfile
import uuid
import wave
from pathlib import Path

import numpy as np

import streamlit as st

from face_voice_rec.config import KNOWN_FACES_DIR, UNKNOWN_VOICES_DIR
from face_voice_rec.domain.transcription.transcription_core import extract_audio_from_video
from face_voice_rec.domain.voice import voice_core
from face_voice_rec.domain.voice.audio_core import read_wav_pcm
from face_voice_rec.storage.database import (
    add_voice_embedding,
    create_person,
    delete_unknown_voice,
    list_persons,
    list_unknown_voice_samples,
    list_unknown_voices,
    list_voice_enrollments,
    reassign_voice_embedding,
    resolve_unknown_voice,
)


def _fmt_time(seconds: float) -> str:
    seconds = float(seconds)
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def render() -> None:
    st.header("🎙️ Voice Enrollment")

    enrollment_generation = st.session_state.setdefault(
        "voice_enrollment_generation",
        0,
    )
    person_rows = list_persons()
    persons = {name: person_id for person_id, name, _created, _updated in person_rows}
    known_voice_root = (KNOWN_FACES_DIR.parent / "known_voices").resolve()
    unknown_voice_root = Path(UNKNOWN_VOICES_DIR).resolve()

    def load_audio(path):
        audio_path = Path(path).resolve()
        if not any(
            root == audio_path or root in audio_path.parents
            for root in (known_voice_root, unknown_voice_root)
        ):
            raise ValueError("Audio path is outside the application's voice storage.")
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio sample is missing: {audio_path.name}")
        audio_bytes = audio_path.read_bytes()
        with wave.open(io.BytesIO(audio_bytes), "rb") as audio_file:
            duration = audio_file.getnframes() / audio_file.getframerate()
        return audio_bytes, duration

    def select_audio(label, audio_bytes, mime_type="audio/wav", duration=None):
        st.session_state["voice_enrollment_playback"] = {
            "label": label,
            "audio": audio_bytes,
            "mime_type": mime_type,
            "duration": duration,
        }

    current_playback = st.session_state.get("voice_enrollment_playback")
    if current_playback:
        st.caption(f"Selected sample: {current_playback['label']}")
        try:
            st.audio(
                current_playback["audio"],
                format=current_playback["mime_type"],
                autoplay=True,
            )
        except Exception as error:
            st.error(f"Could not play this voice sample: {error}")
        st.caption(
            "Use the player controls to pause, replay, or seek. Selecting another "
            "sample replaces this player so voice samples do not play simultaneously."
        )

    st.subheader("Enroll a clean voice sample")
    selected_name = st.selectbox(
        "Person",
        ["-- Select --", "➕ Create a new person"] + list(persons.keys()),
        key="voice_enroll_person",
    )
    new_person_name = None
    if selected_name == "➕ Create a new person":
        new_person_name = st.text_input(
            "New person's name",
            key="voice_enroll_new_person",
        )
    uploaded = st.file_uploader(
        "Upload a short voice sample (WAV or MP3)",
        type=["wav", "mp3"],
        key=f"voice_enrollment_upload_{enrollment_generation}",
    )
    recorded = st.audio_input(
        "Or record a voice sample",
        key=f"voice_enrollment_recording_{enrollment_generation}",
    )
    sample_payload = recorded.getvalue() if recorded is not None else (
        uploaded.getvalue() if uploaded is not None else None
    )
    sample_mime = "audio/wav"
    if uploaded is not None and uploaded.name.lower().endswith(".mp3") and recorded is None:
        sample_mime = "audio/mpeg"
    if sample_payload:
        if st.button(
            "▶ Play uploaded/recorded sample",
            key=f"preview_voice_enrollment_{enrollment_generation}",
        ):
            select_audio(
                "New voice sample",
                sample_payload,
                sample_mime,
            )
            st.rerun()

    confirm_enrollment = st.checkbox(
        "I have listened to this sample and confirm it belongs to the selected person.",
        key=f"confirm_voice_enrollment_{enrollment_generation}",
    )
    if st.button(
        "➕ Add Voice Sample",
        key=f"add_voice_sample_{enrollment_generation}",
    ):
        if selected_name == "-- Select --":
            st.warning("Please select a person first.")
        elif not confirm_enrollment:
            st.warning("Listen to the sample and confirm its speaker before saving.")
        elif uploaded is None and recorded is None:
            st.warning("Please upload or record a voice sample first.")
        elif (
            selected_name == "➕ Create a new person"
            and not (new_person_name or "").strip()
        ):
            st.warning("Enter a name for the new person.")
        else:
            try:
                if recorded is not None:
                    pcm = read_wav_pcm(io.BytesIO(recorded.getvalue()))
                elif uploaded.name.lower().endswith(".mp3"):
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        mp3_path = Path(tmp_dir) / "upload.mp3"
                        mp3_path.write_bytes(uploaded.getvalue())
                        wav_path = extract_audio_from_video(mp3_path, Path(tmp_dir) / "upload.wav")
                        pcm = read_wav_pcm(wav_path)
                else:
                    pcm = read_wav_pcm(io.BytesIO(uploaded.getvalue()))
            except (ValueError, wave.Error, EOFError) as error:
                st.error(f"Invalid audio file (expected 16kHz mono 16-bit PCM): {error}")
            except (OSError, RuntimeError) as error:
                st.error(f"Could not convert the uploaded MP3: {error}")
            else:
                embedding = voice_core.extract_voice_embedding(pcm)
                if embedding is None:
                    st.error("Could not extract a voice embedding from this sample.")
                else:
                    embedding_path = None
                    audio_path = None
                    try:
                        person_id = (
                            persons[selected_name]
                            if selected_name != "➕ Create a new person"
                            else create_person(new_person_name.strip())
                        )
                        embedding_dir = known_voice_root / str(person_id)
                        embedding_dir.mkdir(parents=True, exist_ok=True)
                        sample_id = uuid.uuid4().hex
                        embedding_path = embedding_dir / f"{person_id}_{sample_id}.npy"
                        audio_path = embedding_dir / f"{person_id}_{sample_id}.wav"
                        np.save(embedding_path, embedding)
                        with wave.open(str(audio_path), "wb") as audio_file:
                            audio_file.setnchannels(1)
                            audio_file.setsampwidth(2)
                            audio_file.setframerate(16000)
                            audio_file.writeframes(pcm)
                        add_voice_embedding(
                            person_id,
                            embedding_path,
                            quality=1.0,
                            audio_path=audio_path,
                        )
                    except (OSError, ValueError, sqlite3.Error, wave.Error) as error:
                        for artifact_path in (embedding_path, audio_path):
                            if artifact_path is not None:
                                try:
                                    artifact_path.unlink(missing_ok=True)
                                except OSError as cleanup_error:
                                    st.error(
                                        f"Could not clean up failed voice-sample file "
                                        f"{artifact_path.name}: {cleanup_error}"
                                    )
                        st.error(f"Could not save this voice sample: {error}")
                    else:
                        saved_name = (
                            new_person_name.strip()
                            if selected_name == "➕ Create a new person"
                            else selected_name
                        )
                        st.success(f"Voice sample added for {saved_name}.")
                        st.session_state["voice_enrollment_generation"] = (
                            enrollment_generation + 1
                        )
                        st.session_state.pop("voice_enrollment_playback", None)
                        st.rerun()

    voice_enrollments = list_voice_enrollments()
    with st.expander(
        f"Saved enrolled voice samples ({len(voice_enrollments)})",
        expanded=False,
    ):
        if not voice_enrollments:
            st.info("No voice samples have been enrolled yet.")
        else:
            for embedding_id, person_id, person_name, _embedding_path, audio_path, quality, created_at in voice_enrollments:
                with st.container(border=True):
                    st.markdown(f"**Sample #{embedding_id} · {person_name}**")
                    st.caption(
                        f"Person ID: {person_id} · Status: Enrolled · "
                        f"Quality: {float(quality or 0.0):.2f} · Added: {created_at}"
                    )
                    if audio_path:
                        try:
                            audio_bytes, duration = load_audio(audio_path)
                            st.caption(f"Duration: {_fmt_time(duration)}")
                            if st.button(
                                f"▶ Play sample #{embedding_id}",
                                key=f"play_enrolled_voice_{embedding_id}",
                            ):
                                select_audio(
                                    f"Enrolled sample #{embedding_id} · {person_name}",
                                    audio_bytes,
                                    duration=duration,
                                )
                                st.rerun()
                        except (OSError, ValueError, wave.Error, EOFError) as error:
                            st.error(f"Sample #{embedding_id} audio is unavailable: {error}")
                    else:
                        st.warning(
                            "No original audio was saved for this older enrollment; "
                            "only its voice embedding is available."
                        )
                    edit_key = f"edit_enrolled_voice_{embedding_id}"
                    if st.button(
                        "✏️ Edit person",
                        key=f"edit_enrolled_voice_button_{embedding_id}",
                    ):
                        st.session_state["editing_voice_enrollment_id"] = embedding_id
                    if st.session_state.get("editing_voice_enrollment_id") == embedding_id:
                        person_names = list(persons)
                        selected_index = next(
                            (
                                index
                                for index, name in enumerate(person_names)
                                if persons[name] == person_id
                            ),
                            None,
                        )
                        if selected_index is None:
                            st.error(
                                "The currently assigned person is unavailable. "
                                "Refresh the person list before editing this sample."
                            )
                        else:
                            with st.form(edit_key):
                                replacement_name = st.selectbox(
                                    "Assign this saved sample to",
                                    person_names,
                                    index=selected_index,
                                    key=f"{edit_key}_person",
                                )
                                save_assignment = st.form_submit_button(
                                    "Save person assignment"
                                )
                                cancel_assignment = st.form_submit_button("Cancel")
                            if cancel_assignment:
                                st.session_state.pop(
                                    "editing_voice_enrollment_id",
                                    None,
                                )
                                st.rerun()
                            if save_assignment:
                                try:
                                    if reassign_voice_embedding(
                                        embedding_id,
                                        persons[replacement_name],
                                    ):
                                        voice_core.index_voice_enrollments_for_person(
                                            persons[replacement_name]
                                        )
                                        st.session_state.pop(
                                            "editing_voice_enrollment_id",
                                            None,
                                        )
                                        st.success(
                                            f"Sample #{embedding_id} is now assigned "
                                            f"to {replacement_name}."
                                        )
                                        st.rerun()
                                    else:
                                        st.error(
                                            f"Saved voice sample #{embedding_id} "
                                            "could not be found."
                                        )
                                except (TypeError, ValueError, sqlite3.Error) as error:
                                    st.error(
                                        f"Could not change the person for sample "
                                        f"#{embedding_id}: {error}"
                                    )

    st.subheader("Unresolved unknown voices")
    unknown_voices = list_unknown_voices()
    if not unknown_voices:
        st.info("No unresolved unknown voices.")
    else:
        for unknown_voice_id, label, embedding_path, created_at, resolved_person_id in unknown_voices:
            with st.container(border=True):
                st.markdown(f"### ❓ {label} (Unknown voice #{unknown_voice_id})")
                sample_rows = list_unknown_voice_samples(unknown_voice_id)
                st.caption(
                    f"Samples: {len(sample_rows)} · Created: {created_at} · "
                    f"Status: {'Resolved' if resolved_person_id else 'Unresolved'}"
                )
                if not sample_rows and embedding_path:
                    st.warning(
                        "This legacy voice record has an embedding but no playable "
                        "audio sample was preserved."
                    )
                for sample_id, sample_embedding, quality, sample_created, sample_audio, source_ref, start_ms, end_ms in sample_rows:
                    st.markdown(f"**Sample #{sample_id}**")
                    duration_text = "Duration unavailable"
                    if start_ms is not None and end_ms is not None:
                        duration_text = f"Duration: {_fmt_time((end_ms - start_ms) / 1000)}"
                    st.caption(
                        f"{duration_text} · Quality: {float(quality or 0.0):.2f} · "
                        f"Recorded: {sample_created}"
                    )
                    if sample_audio:
                        try:
                            audio_bytes, duration = load_audio(sample_audio)
                            st.caption(f"Audio duration: {_fmt_time(duration)}")
                            if st.button(
                                f"▶ Play sample #{sample_id}",
                                key=f"play_unknown_voice_{sample_id}",
                            ):
                                select_audio(
                                    f"{label} · sample #{sample_id}",
                                    audio_bytes,
                                    duration=duration,
                                )
                                st.rerun()
                        except (OSError, ValueError, wave.Error, EOFError) as error:
                            st.error(f"Sample #{sample_id} audio is unavailable: {error}")
                    else:
                        st.warning(
                            "No original audio was saved for this sample, so it cannot "
                            "be played. Reprocess its source video after updating the app."
                        )

                if st.button(
                    "🗑️ Delete unresolved voice",
                    key=f"delete_unknown_voice_{unknown_voice_id}",
                ):
                    if delete_unknown_voice(unknown_voice_id):
                        st.rerun()
                    st.warning("This unknown voice is no longer unresolved.")

                with st.form(f"resolve_unknown_voice_{unknown_voice_id}"):
                    assignment_kind = st.radio(
                        "Assign this voice to",
                        ["Existing person", "Create new person"],
                        key=f"assignment_kind_{unknown_voice_id}",
                        horizontal=True,
                    )
                    person_id = None
                    new_name = None
                    if assignment_kind == "Existing person":
                        if persons:
                            selected_person = st.selectbox(
                                "Person",
                                list(persons.keys()),
                                key=f"assign_voice_{unknown_voice_id}",
                            )
                            person_id = persons[selected_person]
                        else:
                            st.info("No people exist yet; choose Create new person.")
                    else:
                        new_name = st.text_input(
                            "New person's name",
                            key=f"new_voice_person_{unknown_voice_id}",
                        )
                    confirmed = st.checkbox(
                        "I listened to this sample and confirm the speaker's identity.",
                        key=f"confirm_unknown_voice_{unknown_voice_id}",
                    )
                    submitted = st.form_submit_button("✅ Confirm and save voice profile")
                if submitted:
                    if not confirmed:
                        st.warning("Confirm the speaker identity before saving.")
                    elif assignment_kind == "Existing person" and person_id is None:
                        st.warning("Select an existing person or choose Create new person.")
                    elif assignment_kind == "Create new person" and not (new_name or "").strip():
                        st.warning("Enter a name for the new person.")
                    else:
                        try:
                            resolved_id = resolve_unknown_voice(
                                unknown_voice_id,
                                person_id=person_id,
                                new_person_name=new_name if assignment_kind == "Create new person" else None,
                            )
                            voice_core.index_voice_enrollments_for_person(resolved_id)
                            resolved_name = next(
                                name
                                for person_id, name, _created_at, _updated_at in list_persons()
                                if person_id == resolved_id
                            )
                            st.success(f"{label} assigned to {resolved_name}.")
                            st.rerun()
                        except (OSError, ValueError) as error:
                            st.error(f"Could not save the voice assignment: {error}")
