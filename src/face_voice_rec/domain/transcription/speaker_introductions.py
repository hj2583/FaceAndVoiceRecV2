"""Extract and safely associate participant self-introductions."""

from dataclasses import dataclass
import logging
import re

from face_voice_rec import config
from face_voice_rec.storage import database


def _index_voice_profile(person_id: int) -> None:
    from face_voice_rec.domain.voice.voice_core import index_voice_enrollments_for_person

    index_voice_enrollments_for_person(person_id)


logger = logging.getLogger(__name__)

_WORD = r"[^\W\d_][\w’'-]*"
_NAME = rf"(?P<name>{_WORD}(?:\s+{_WORD}){{0,2}}?)"
_NAME_END = r"(?=\s*(?:,|[.!?;:]|and\b|dan\b|y\b|et\b|i['’]m\b|i am\b|saya\b|soy\b|je suis\b|$))"
_INTRODUCTIONS = (
    re.compile(
        rf"\b(?:hi|hello|hey|good morning|good afternoon)\b[\s,!-]*"
        rf"(?:my name is|i am|i['’]m|you can call me)\s+{_NAME}{_NAME_END}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:my name is|i am|i['’]m|you can call me)\s+{_NAME}{_NAME_END}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:nama saya|saya bernama|panggil saya)\s+{_NAME}{_NAME_END}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:me llamo|mi nombre es|puedes llamarme)\s+{_NAME}{_NAME_END}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:je m['’]appelle|mon nom est|appelez-moi)\s+{_NAME}{_NAME_END}",
        re.IGNORECASE,
    ),
)
_INVALID_NAMES = {
    "a",
    "an",
    "the",
    "head",
    "chief",
    "director",
    "manager",
    "officer",
    "engineer",
    "administration",
    "finance",
    "research",
    "information",
    "saya",
    "soy",
    "je",
}
_TITLE_PATTERNS = (
    re.compile(
        r"(?:,\s*|(?:and|dan|y|et)\s+)"
        r"(?:(?:i['’]m|i am|i work as|i work in|my role is|"
        r"saya bekerja sebagai|soy|je suis)\s+)?"
        r"(?P<title>[^.!?;]+)",
        re.IGNORECASE,
    ),
)
_GENERIC_PERSON_NAME = re.compile(
    r"^(?:person|the centre|the center)\s+\d+$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SelfIntroduction:
    name: str
    job_title: str | None
    start: int
    end: int
    confidence: float


def _clean_name(value: str) -> str:
    words = re.findall(_WORD, value.strip())
    if not words or len(words) > 3:
        return ""
    if words[0].casefold() in _INVALID_NAMES:
        return ""
    return " ".join(word[0].upper() + word[1:] for word in words)


def _extract_job_title(text: str, match_end: int) -> str | None:
    remainder = text[match_end:].strip()
    if not remainder:
        return None
    for pattern in _TITLE_PATTERNS:
        match = pattern.search(remainder)
        if match is None:
            continue
        title = match.group("title").strip(" ,.-")
        title = re.split(
            r"\s+(?:and i['’]d|and i would|dan saya|y yo|et je)\b",
            title,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" ,.-")
        return title or None
    return None


def extract_self_introductions(text: str) -> list[SelfIntroduction]:
    """Find explicit self-introduction phrases without inferring from context."""
    matches = []
    for pattern in _INTRODUCTIONS:
        for match in pattern.finditer(text):
            name = _clean_name(match.group("name"))
            if not name:
                continue
            matches.append(
                SelfIntroduction(
                    name=name,
                    job_title=_extract_job_title(text, match.end()),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85,
                )
            )
    unique = {}
    for introduction in sorted(matches, key=lambda item: (item.start, item.end)):
        duplicate = any(
            existing.name.casefold() == introduction.name.casefold()
            and min(existing.end, introduction.end)
            >= max(existing.start, introduction.start)
            for existing in unique.values()
        )
        if not duplicate:
            unique[(introduction.start, introduction.name.casefold())] = introduction
    return list(unique.values())


def _collect_utterances(rows: list[tuple]) -> list[dict]:
    utterances = []
    current = None
    for row in rows:
        segment_id, speaker, start_ms, end_ms, text = row
        text = str(text or "").strip()
        if not text or int(end_ms) <= int(start_ms):
            continue
        if (
            current is not None
            and current["speaker_label"] == speaker
            and 0 <= int(start_ms) - current["end_ms"] <= 900
            and current["end_ms"] - current["start_ms"] < 6000
            and not current["text"].rstrip().endswith((".", "?", "!", "。", "！", "？"))
        ):
            current["text"] = f"{current['text']} {text}"
            current["end_ms"] = int(end_ms)
            current["source_segment_ids"].append(int(segment_id))
        else:
            current = {
                "speaker_label": str(speaker),
                "start_ms": int(start_ms),
                "end_ms": int(end_ms),
                "text": text,
                "source_segment_ids": [int(segment_id)],
            }
            utterances.append(current)
    return utterances


def _best_face_event(speech_events: list[dict], start_ms: int, end_ms: int):
    duration = end_ms - start_ms
    if duration <= 0:
        return None, "invalid introduction interval"
    candidates = []
    for event in speech_events:
        try:
            event_start = int(round(float(event["start_time"]) * 1000))
            event_end = int(round(float(event["end_time"]) * 1000))
            activity = float(event.get("activity_score", 0.0))
            track_id = event.get("track_id")
        except (KeyError, TypeError, ValueError):
            continue
        if track_id is None or event_end <= event_start:
            continue
        overlap = max(0, min(end_ms, event_end) - max(start_ms, event_start))
        ratio = overlap / duration
        if ratio >= 0.5 and activity >= config.SPEAKER_MIN_ACTIVITY_SCORE:
            candidates.append((ratio * activity, ratio, event))
    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        return None, "No single active face track covered enough of the introduction"
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.1:
        return None, "Multiple face tracks have similarly strong activity"
    return candidates[0][2], None


def _find_person(name: str):
    folded = name.casefold()
    for person_id, person_name, _created_at, _updated_at in database.list_persons():
        if person_name.casefold() == folded:
            return int(person_id), person_name
    return None


def _person_for_speaker_label(label: str):
    for person_id, person_name, _created_at, _updated_at in database.list_persons():
        if person_name.casefold() == label.casefold():
            return int(person_id), person_name
    return None


def _persist_candidate(
    meeting_id,
    utterance,
    introduction,
    status,
    reason,
    voice_unknown_id=None,
    face_track_id=None,
    face_unknown_id=None,
    person_id=None,
    match_confidence=None,
):
    return database.save_meeting_introduction(
        meeting_id=meeting_id,
        name=introduction.name,
        job_title=introduction.job_title,
        start_ms=utterance["start_ms"],
        end_ms=utterance["end_ms"],
        speaker_label=utterance["speaker_label"],
        source_segment_ids=utterance["source_segment_ids"],
        parser_confidence=introduction.confidence,
        status=status,
        voice_unknown_id=voice_unknown_id,
        face_track_id=face_track_id,
        face_unknown_id=face_unknown_id,
        person_id=person_id,
        match_confidence=match_confidence,
        reason=reason,
    )


def process_meeting_introductions(
    meeting_id: int,
    speech_events: list[dict] | None = None,
) -> list[dict]:
    """Detect introductions, link reliable voice identity, and keep uncertainty pending."""
    with database.get_conn() as connection:
        rows = connection.execute(
            """
            SELECT segment_id, speaker_label, start_ms, end_ms, text
            FROM transcription_segments
            WHERE meeting_id=?
            ORDER BY start_ms, segment_id
            """,
            (int(meeting_id),),
        ).fetchall()

    identity_updates = []
    changed_transcript = False
    speech_events = speech_events or []
    for utterance in _collect_utterances(rows):
        introductions = extract_self_introductions(utterance["text"])
        if not introductions:
            continue
        if len(introductions) > 1:
            for introduction in introductions:
                _persist_candidate(
                    meeting_id,
                    utterance,
                    introduction,
                    "pending_multiple_introductions",
                    "More than one name was spoken in this utterance; speaker boundaries are uncertain.",
                )
            logger.info(
                "Stored multiple self-introductions as pending for meeting %s segments %s",
                meeting_id,
                utterance["source_segment_ids"],
            )
            continue

        introduction = introductions[0]
        face_event, face_reason = _best_face_event(
            speech_events,
            utterance["start_ms"],
            utterance["end_ms"],
        )
        face_track_id = int(face_event["track_id"]) if face_event else None
        face_person_id = face_event.get("person_id") if face_event else None
        face_unknown = None
        if face_event is not None and face_person_id is None:
            face_unknown_label = face_event.get("face_unknown_label")
            if face_unknown_label:
                face_unknown = database.get_unknown_face_track_by_label(
                    face_unknown_label
                )
                if (
                    face_unknown is not None
                    and face_unknown["resolved_person_id"] is not None
                ):
                    face_person_id = face_unknown["resolved_person_id"]
        face_unknown_id = (
            face_unknown["unknown_id"]
            if face_unknown is not None
            else None
        )
        match_score = (
            float(face_event.get("activity_score", 0.0))
            if face_event
            else None
        )
        speaker_label = utterance["speaker_label"]
        existing_name = _find_person(introduction.name)
        speaker_person = _person_for_speaker_label(speaker_label)
        unknown = database.get_unknown_voice_by_label(speaker_label)
        voice_unknown_id = int(unknown[0]) if unknown else None
        voice_person_id = int(unknown[1]) if unknown and unknown[1] is not None else None
        reason = None
        person_id = None
        face_enrolled = False
        status = "pending_speaker_attribution"

        if (
            face_person_id is not None
            and (
                (speaker_person is not None and speaker_person[0] != face_person_id)
                or (
                    voice_person_id is not None
                    and voice_person_id != face_person_id
                )
            )
        ):
            reason = (
                "Voice diarization and the active recognized face point to different "
                "people; neither identity was changed."
            )
            status = "pending_audio_face_conflict"
        elif face_person_id is not None:
            face_person_id = int(face_person_id)
            with database.get_conn() as connection:
                face_person = connection.execute(
                    "SELECT name FROM persons WHERE person_id=?",
                    (face_person_id,),
                ).fetchone()
            if face_person is None:
                reason = "The active face event refers to a person record that no longer exists."
                status = "pending_invalid_face_person"
            elif face_person[0].casefold() == introduction.name.casefold():
                person_id = face_person_id
                if voice_person_id is not None and voice_person_id != face_person_id:
                    person_id = None
                    reason = (
                        "Voice enrollment and active recognized face point to "
                        "different people; neither identity was changed."
                    )
                    status = "pending_audio_face_conflict"
                elif unknown is not None and voice_person_id is None:
                    try:
                        person_id = database.resolve_unknown_voice(
                            voice_unknown_id,
                            person_id=face_person_id,
                        )
                        _index_voice_profile(person_id)
                        status = "voice_enrolled_face_matched"
                        changed_transcript = True
                    except (FileNotFoundError, ValueError, OSError, *database.DATABASE_ERRORS) as error:
                        person_id = None
                        reason = f"Could not enroll the matched voice: {error}"
                        status = "pending_voice_enrollment"
                        logger.warning(
                            "Could not enroll face-matched introduction for meeting %s: %s",
                            meeting_id,
                            error,
                        )
                else:
                    status = "matched_face_and_voice"
            elif _GENERIC_PERSON_NAME.fullmatch(face_person[0]) and speaker_person and speaker_person[0] == face_person_id:
                name_collision = _find_person(introduction.name)
                if name_collision and name_collision[0] != face_person_id:
                    reason = "The introduced name already belongs to a different person record."
                    status = "pending_name_conflict"
                else:
                    database.rename_person(face_person_id, introduction.name)
                    person_id = face_person_id
                    status = "matched_voice_face_named"
                    changed_transcript = True
            else:
                reason = (
                    f"Active face track {face_track_id} is already recognized as "
                    f"{face_person[0]}, not {introduction.name}; no identity was overwritten."
                )
                status = "pending_identity_conflict"
        elif speaker_person is not None:
            person_id, speaker_name = speaker_person
            if speaker_name.casefold() == introduction.name.casefold():
                status = "matched_voice"
            elif _GENERIC_PERSON_NAME.fullmatch(speaker_name):
                name_collision = _find_person(introduction.name)
                if name_collision and name_collision[0] != person_id:
                    reason = "The introduced name already belongs to another person record."
                    status = "pending_name_conflict"
                    person_id = None
                else:
                    database.rename_person(person_id, introduction.name)
                    status = "matched_voice_named"
                    changed_transcript = True
            else:
                reason = (
                    f"Speaker voice is already enrolled as {speaker_name}; "
                    f"the spoken name was {introduction.name}."
                )
                status = "pending_identity_conflict"
                person_id = None
        elif voice_person_id is not None:
            with database.get_conn() as connection:
                voice_person = connection.execute(
                    "SELECT name FROM persons WHERE person_id=?",
                    (voice_person_id,),
                ).fetchone()
            if voice_person and voice_person[0].casefold() == introduction.name.casefold():
                person_id = voice_person_id
                status = "matched_voice"
            elif voice_person and _GENERIC_PERSON_NAME.fullmatch(voice_person[0]):
                collision = _find_person(introduction.name)
                if collision and collision[0] != voice_person_id:
                    reason = "The introduced name already belongs to another person record."
                    status = "pending_name_conflict"
                else:
                    database.rename_person(voice_person_id, introduction.name)
                    person_id = voice_person_id
                    status = "matched_voice_named"
                    changed_transcript = True
            else:
                reason = (
                    f"Voice cluster is already resolved to {voice_person[0] if voice_person else 'a missing person'}."
                )
                status = "pending_identity_conflict"
        elif unknown is not None and existing_name is not None:
            try:
                person_id = database.resolve_unknown_voice(
                    voice_unknown_id,
                    person_id=existing_name[0],
                )
                _index_voice_profile(person_id)
                status = "voice_enrolled_face_pending"
                changed_transcript = True
            except (FileNotFoundError, ValueError, OSError, *database.DATABASE_ERRORS) as error:
                reason = f"Could not enroll the introduced voice: {error}"
                status = "pending_voice_enrollment"
                logger.warning(
                    "Could not enroll introduced voice for meeting %s speaker %s: %s",
                    meeting_id,
                    speaker_label,
                    error,
                )
        elif unknown is not None:
            try:
                person_id = database.resolve_unknown_voice(
                    voice_unknown_id,
                    new_person_name=introduction.name,
                )
                _index_voice_profile(person_id)
                status = "voice_enrolled_face_pending"
                changed_transcript = True
            except (FileNotFoundError, ValueError, OSError, *database.DATABASE_ERRORS) as error:
                reason = f"Could not enroll the introduced voice: {error}"
                status = "pending_voice_enrollment"
                logger.warning(
                    "Could not enroll introduced voice for meeting %s speaker %s: %s",
                    meeting_id,
                    speaker_label,
                    error,
                )
        elif face_event is not None and face_unknown_id is not None:
            person_id = (
                existing_name[0]
                if existing_name is not None
                else database.create_person(introduction.name)
            )
            status = "matched_face_voice_pending"
            reason = (
                "The active face track is uniquely aligned with the introduction, "
                "but no stable voice identity is available."
            )
            changed_transcript = True
        elif existing_name is not None and face_event is not None and face_person_id is None:
            status = "pending_face_identity"
            reason = "A unique active face track was found, but no stable voice identity was available."
        else:
            reason = face_reason or "No stable diarized voice identity was available."

        if person_id is not None and face_unknown_id is not None:
            try:
                from face_voice_rec.domain.face.face_core import enroll_unknown_face_track

                face_enrolled = (
                    enroll_unknown_face_track(face_unknown_id, person_id) > 0
                )
                if status != "matched_face_voice_pending":
                    status = "matched_face_and_voice"
                reason = None
                changed_transcript = True
            except (FileNotFoundError, ValueError, OSError, *database.DATABASE_ERRORS) as error:
                logger.warning(
                    "Could not enroll active face track for meeting %s: %s",
                    meeting_id,
                    error,
                )
                if status.startswith("voice_enrolled_"):
                    status = "voice_enrolled_face_pending"
                elif status.startswith("matched_voice"):
                    status = "matched_voice_face_pending"
                else:
                    person_id = None
                    status = "pending_face_enrollment"
                reason = f"Could not enroll the active face samples: {error}"

        _persist_candidate(
            meeting_id,
            utterance,
            introduction,
            status,
            reason,
            voice_unknown_id=voice_unknown_id,
            face_track_id=face_track_id,
            face_unknown_id=face_unknown_id,
            person_id=person_id,
            match_confidence=match_score,
        )
        if person_id is not None and status.startswith(("matched_", "voice_enrolled_")):
            identity_update = {
                "speaker_label": speaker_label,
                "person_name": introduction.name,
                "start_ms": utterance["start_ms"],
                "end_ms": utterance["end_ms"],
                "global": status.startswith(
                    ("matched_voice_named", "voice_enrolled_")
                ),
            }
            if face_enrolled:
                identity_update["face_enrolled"] = True
            identity_updates.append(identity_update)
        if person_id is not None and status.startswith(("matched_", "voice_enrolled_")):
            changed_transcript = True
        logger.info(
            "Self-introduction meeting=%s name=%s speaker=%s status=%s face_track=%s",
            meeting_id,
            introduction.name,
            speaker_label,
            status,
            face_track_id,
        )

    if changed_transcript:
        from face_voice_rec.domain.transcription import transcription_core

        transcription_core.refresh_readable_transcript(meeting_id, use_llm=False)
    return identity_updates
