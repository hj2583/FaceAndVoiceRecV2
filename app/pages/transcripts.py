import html
import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from face_voice_rec import config
from face_voice_rec.config import DB_PATH, TRANSCRIPT_REFINEMENT_ENABLED
from face_voice_rec.domain.transcription.transcription_core import (
    SENTENCE_TYPES,
    TranscriptionSegment,
    classify_sentence,
    group_readable_paragraphs,
    refresh_readable_transcript,
)
from face_voice_rec.storage.database import (
    confirm_meeting_introduction,
    create_person,
    list_meeting_introductions,
    list_persons,
    update_transcription_segments,
)

LOW_CONFIDENCE_WARN = 0.7
LOW_CONFIDENCE_BAD = 0.5


def _fmt_time(seconds: float) -> str:
    seconds = float(seconds)
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def _parse_refinement_method(method: str) -> tuple[str, str]:
    value = str(method or "").strip()
    if not value:
        return "unknown", ""
    if value == "deterministic":
        return "deterministic", ""
    provider, _, model = value.partition(":")
    return provider.strip().lower() or "unknown", model.strip()


def _provider_caption(method: str) -> str:
    provider, model = _parse_refinement_method(method)
    if provider == "foundry":
        return f"Cloud Foundry: {model}" if model else "Cloud Foundry"
    if provider == "ollama":
        return f"Local Ollama: {model}" if model else "Local Ollama"
    if provider == "deterministic":
        return "Deterministic"
    return f"{provider}: {model}" if model else (method or "Unknown method")


def _refinement_status_caption(status: str, method: str) -> str:
    provider_label = _provider_caption(method)
    if status == "complete_local_fallback":
        return f"{provider_label} (cloud fallback)"
    if status == "disabled":
        return "Deterministic (LLM disabled)"
    if status == "fallback":
        return "Deterministic (refinement failed)"
    if status == "complete":
        return provider_label
    return f"{provider_label} ({status})"


def render() -> None:
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
            SELECT speaker_label, start_ms, end_ms, text, original_text,
                   confidence, sentence_type, segment_id
            FROM transcription_segments
            WHERE meeting_id=? ORDER BY start_ms
            """,
            (selected_id,),
        ).fetchall()
        readable_rows = connection.execute(
            """
            SELECT speaker_label, start_ms, end_ms, text, source_segment_ids,
                   sentence_type, refinement_method, refined_segment_id
            FROM refined_transcription_segments
            WHERE meeting_id=? ORDER BY start_ms, refined_segment_id
            """,
            (selected_id,),
        ).fetchall()
        refinement_run = connection.execute(
            """
            SELECT status, method, speaker_change_recommendations, error
            FROM transcript_refinement_runs WHERE meeting_id=?
            """,
            (selected_id,),
        ).fetchone()

    if not rows:
        st.info("This meeting has no transcript segments.")
        return

    introductions = list_meeting_introductions(selected_id)
    st.subheader("Participant self-introductions")
    if not introductions:
        st.info("No explicit self-introductions have been detected for this meeting.")
    for (
        introduction_id,
        name,
        job_title,
        start_ms,
        end_ms,
        speaker_label,
        _source_segment_ids,
        _voice_unknown_id,
        face_track_id,
        person_id,
        parser_confidence,
        match_confidence,
        status,
        reason,
        face_unknown_id,
    ) in introductions:
        title_text = f" — {job_title}" if job_title else ""
        track_text = (
            f" · face track {face_track_id}"
            if face_track_id is not None
            else ""
        )
        if face_unknown_id is not None:
            track_text += f" · saved face record {face_unknown_id}"
        person_text = f" · person ID {person_id}" if person_id is not None else ""
        match_text = (
            f" · face activity score {float(match_confidence):.2f} (heuristic)"
            if match_confidence is not None
            else ""
        )
        st.caption(
            f"{_fmt_time(start_ms / 1000)}–{_fmt_time(end_ms / 1000)} · "
            f"{name}{title_text} · voice label {speaker_label} · "
            f"{status}{person_text}{track_text}{match_text} · "
            f"introduction parser score {float(parser_confidence):.2f} (heuristic)"
        )
        if reason:
            st.info(reason)
        if status == "pending_multiple_introductions":
            st.warning(
                "This segment contains more than one introduction, so speaker "
                "boundaries must be separated before confirming either identity."
            )
        elif (
            str(status).startswith("pending")
            or str(status).endswith("_face_pending")
        ):
            persons = list_persons()
            person_options = {
                f"{person_name} (ID {existing_person_id})": existing_person_id
                for existing_person_id, person_name, _created, _updated in persons
            }
            create_option = f"Create new person: {name}"
            options = [*person_options, create_option]
            with st.form(key=f"confirm_introduction_{introduction_id}"):
                chosen = st.selectbox(
                    "Confirm this introduction as",
                    options,
                    key=f"introduction_person_{introduction_id}",
                )
                submitted = st.form_submit_button("Confirm identity")
            if submitted:
                try:
                    person_id_to_assign = (
                        create_person(name)
                        if chosen == create_option
                        else person_options[chosen]
                    )
                    confirm_meeting_introduction(
                        introduction_id,
                        person_id_to_assign,
                    )
                    from app.pages.face_database import get_face_index

                    get_face_index().rebuild()
                except (ValueError, OSError, sqlite3.Error) as error:
                    st.error(f"Could not confirm introduction: {error}")
                else:
                    st.success(f"Confirmed {name} as {person_id_to_assign}.")
                    st.rerun()

    if not readable_rows and st.button(
        "Build readable transcript",
        key=f"build_readable_{selected_id}",
    ):
        with st.spinner("Building a readable view from the detailed transcript..."):
            refresh_readable_transcript(selected_id, use_llm=False)
        st.rerun()

    if TRANSCRIPT_REFINEMENT_ENABLED:
        if config.TRANSCRIPT_REFINEMENT_USE_CLOUD:
            action_label = (
                "Refine readable transcript with cloud Foundry "
                "(fallback to local Ollama)"
            )
            spinner_label = (
                "Asking the configured cloud Foundry model to format the transcript "
                "(local fallback on failure)..."
            )
        else:
            action_label = "Refine readable transcript with local Ollama"
            spinner_label = (
                "Asking the configured local Ollama model to format the transcript..."
            )
        if st.button(
            action_label,
            key=f"refine_readable_{selected_id}",
        ):
            with st.spinner(spinner_label):
                refresh_readable_transcript(selected_id, use_llm=True)
            st.rerun()
    else:
        action_hint = (
            "cloud Foundry (with local fallback)"
            if config.TRANSCRIPT_REFINEMENT_USE_CLOUD
            else "local Ollama"
        )
        st.caption(
            "Transcript refinement is disabled. Set TRANSCRIPT_REFINEMENT_ENABLED=1 "
            f"before starting the app to enable the {action_hint} action."
        )

    if refinement_run:
        status, method, recommendations_json, refinement_error = refinement_run
        st.caption(f"Readable version: {_refinement_status_caption(status, method)}")
        if status == "fallback" and refinement_error:
            st.warning(
                "Refinement failed; the deterministic version is shown. "
                f"Reason: {refinement_error}"
            )
        elif status == "complete_local_fallback" and refinement_error:
            st.info(
                f"Cloud fallback reason: {refinement_error}"
            )
        try:
            recommendations = json.loads(recommendations_json or "[]")
        except json.JSONDecodeError:
            recommendations = []
        for recommendation in recommendations:
            st.info(
                f"Review speaker attribution after detailed segment "
                f"#{recommendation['after_segment_id']} against the audio: "
                f"{recommendation['reason']}"
            )

    view_kind = st.radio(
        "Transcript view",
        ["Readable", "Detailed", "Whisper source"],
        horizontal=True,
        key=f"transcript_view_{selected_id}",
    )

    is_readable_view = view_kind == "Readable" and bool(readable_rows)
    is_whisper_source_view = view_kind == "Whisper source"
    readable_segments = []
    if is_readable_view:
        for (
            speaker, start_ms, end_ms, text, source_ids,
            _sentence_type, _method, _refined_id,
        ) in readable_rows:
            try:
                parsed_source_ids = json.loads(source_ids)
                if not isinstance(parsed_source_ids, list) or any(
                    isinstance(segment_id, bool) or not isinstance(segment_id, int)
                    for segment_id in parsed_source_ids
                ):
                    raise ValueError("Source segment references must be integer IDs.")
            except (TypeError, json.JSONDecodeError, ValueError) as error:
                st.warning(f"Cannot group a readable transcript row with invalid source IDs: {error}")
                parsed_source_ids = []
            readable_segments.append(TranscriptionSegment(
                speaker,
                int(start_ms),
                int(end_ms),
                text,
                source_segment_ids=tuple(parsed_source_ids),
            ))
    readable_paragraphs = (
        group_readable_paragraphs(readable_segments)
        if is_readable_view
        else []
    )
    original_data = pd.DataFrame(
        [
            {
                "Speaker": speaker,
                "Start": _fmt_time(start_ms / 1000),
                "End": _fmt_time(end_ms / 1000),
                "Type": sentence_type or "",
                "Text": text,
                "Whisper score (heuristic)": confidence,
                "source_segment_ids": json.dumps([segment_id]),
                "segment_id": segment_id,
            }
            for (
                speaker, start_ms, end_ms, text, _whisper_text, confidence,
                sentence_type, segment_id,
            ) in rows
        ]
    )
    whisper_source_data = pd.DataFrame(
        [
            {
                "Speaker": speaker,
                "Start": _fmt_time(start_ms / 1000),
                "End": _fmt_time(end_ms / 1000),
                "Type": sentence_type or "",
                "Text": whisper_text if whisper_text is not None else text,
                "Whisper score (heuristic)": confidence,
                "source_segment_ids": json.dumps([segment_id]),
                "segment_id": segment_id,
            }
            for (
                speaker, start_ms, end_ms, text, whisper_text, confidence,
                sentence_type, segment_id,
            ) in rows
        ]
    )

    if is_readable_view:
        display_data = pd.DataFrame(
            [
                {
                    "Speaker": (
                        paragraph.speaker_labels[0]
                        if len(paragraph.speaker_labels) == 1
                        else "Mixed/uncertain attribution"
                    ),
                    "Speaker labels": " · ".join(paragraph.speaker_labels),
                    "Start": _fmt_time(paragraph.start_ms / 1000),
                    "End": _fmt_time(paragraph.end_ms / 1000),
                    "Type": classify_sentence(paragraph.text),
                    "Text": paragraph.text,
                    "Whisper score (heuristic)": None,
                    "source_segment_ids": json.dumps(paragraph.source_segment_ids),
                    "segment_id": (
                        paragraph.source_segment_ids[0]
                        if paragraph.source_segment_ids
                        else None
                    ),
                }
                for paragraph in readable_paragraphs
            ]
        )
    else:
        display_data = whisper_source_data if is_whisper_source_view else original_data

    with st.container():
        present_types = [t for t in SENTENCE_TYPES if t in set(display_data["Type"])]
        selected_types = st.multiselect(
            "Filter by type",
            present_types,
            key=f"type_filter_{selected_id}",
        )
        visible_data = (
            display_data[display_data["Type"].isin(selected_types)]
            if selected_types
            else display_data
        )

        st.caption(
            "Whisper score (heuristic) is derived from avg_logprob; it is not a "
            "calibrated probability and is not a speaker-match confidence."
            if not is_readable_view
            else "The readable view is linked to its detailed source segment IDs."
        )
        for record in visible_data.to_dict("records"):
            score = record["Whisper score (heuristic)"]
            if score is None or pd.isna(score) or score < LOW_CONFIDENCE_BAD:
                background = "#ffb3b3"
            elif score < LOW_CONFIDENCE_WARN:
                background = "#fff3b0"
            else:
                background = "transparent"
            text_color = "#111" if background != "transparent" else "inherit"
            score_label = (
                f"Whisper score (heuristic) {float(score):.2f}"
                if score is not None and not pd.isna(score)
                else "Whisper score unavailable"
            )
            source_ids = record["source_segment_ids"]
            if is_readable_view:
                try:
                    source_ids = ", ".join(
                        f"#{item}" for item in json.loads(source_ids)
                    )
                except (TypeError, json.JSONDecodeError):
                    source_ids = "unavailable"
                source_label = f" · Source detailed segments {source_ids}"
                speaker_labels = record["Speaker labels"]
                speaker_label_text = (
                    f"{record['Speaker']} · Source speaker labels: {speaker_labels}"
                    if record["Speaker"] == "Mixed/uncertain attribution"
                    else record["Speaker"]
                )
            else:
                source_label = f" · Detailed segment #{record['segment_id']}"
                speaker_label_text = record["Speaker"]
            st.caption(
                f"{speaker_label_text} · {record['Start']}–{record['End']} · "
                f"{record['Type'] or 'Unclassified'} · {score_label}{source_label}"
            )
            st.markdown(
                f'<div style="background-color:{background};color:{text_color};padding:0.2rem 0.35rem;'
                f'white-space:pre-wrap;overflow-wrap:anywhere;margin-bottom:0.8rem">'
                f'{html.escape(str(record["Text"]))}</div>',
                unsafe_allow_html=True,
            )

        low_confidence = visible_data[
            visible_data["Whisper score (heuristic)"].isna()
            | (visible_data["Whisper score (heuristic)"] < LOW_CONFIDENCE_WARN)
        ] if not is_readable_view and not is_whisper_source_view else pd.DataFrame()
        if not low_confidence.empty:
            st.subheader(f"Review low Whisper-score segments ({len(low_confidence)})")
            edited = st.data_editor(
                low_confidence,
                key=f"low_conf_editor_{selected_id}",
                width="stretch",
                hide_index=True,
                disabled=["Speaker", "Start", "End", "Whisper score (heuristic)"],
                column_config={
                    "source_segment_ids": None,
                    "segment_id": None,
                    "Type": st.column_config.SelectboxColumn(
                        "Type", options=list(SENTENCE_TYPES), required=True
                    ),
                    "Text": st.column_config.TextColumn("Text", width="large", required=True),
                    "Whisper score (heuristic)": st.column_config.NumberColumn(
                        "Whisper score (heuristic)", format="%.2f"
                    ),
                },
            )
            if st.button("Save corrections", key=f"save_corrections_{selected_id}"):
                original_by_id = low_confidence.set_index("segment_id")
                updates = []
                for record in edited.to_dict("records"):
                    before = original_by_id.loc[record["segment_id"]]
                    text = str(record["Text"]).strip()
                    if not text:
                        continue
                    if text != before["Text"] or record["Type"] != before["Type"]:
                        updates.append({
                            "segment_id": int(record["segment_id"]),
                            "text": text,
                            "sentence_type": record["Type"] or None,
                            "confidence": before["Whisper score (heuristic)"],
                        })
                if updates:
                    update_transcription_segments(selected_id, updates)
                    st.success(f"Saved {len(updates)} correction(s).")
                    st.rerun()
                else:
                    st.info("No changes to save.")

        original_lines = []
        for (
            speaker, start_ms, _end_ms, _text, whisper_text, _confidence,
            sentence_type, _segment_id,
        ) in rows:
            type_tag = f" ({sentence_type})" if sentence_type else ""
            original_lines.append(
                f"[{_fmt_time(start_ms / 1000)}] {speaker}{type_tag}: "
                f"{whisper_text if whisper_text is not None else _text}"
            )
        st.download_button(
            "Download original transcript",
            "\n".join(original_lines),
            file_name=f"meeting_{selected_id}_original.txt",
            mime="text/plain",
            key=f"download_original_{selected_id}",
        )
        if readable_rows:
            readable_lines = [
                f"[{_fmt_time(start_ms / 1000)}–{_fmt_time(end_ms / 1000)}] "
                f"{speaker}: {text}"
                for (
                    speaker, start_ms, end_ms, text, _source_ids,
                    _sentence_type, _method, _refined_id,
                ) in readable_rows
            ]
            st.download_button(
                "Download readable transcript",
                "\n".join(readable_lines),
                file_name=f"meeting_{selected_id}_readable.txt",
                mime="text/plain",
                key=f"download_readable_{selected_id}",
            )
