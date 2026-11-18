"""Portable relational schema for the PostgreSQL migration path."""

from sqlalchemy import (
    CheckConstraint,
    Column,
    Float as FloatType,
    ForeignKey,
    Index,
    Integer as IntegerType,
    MetaData,
    String as StringType,
    Table,
    Text as TextType,
    UniqueConstraint,
    and_,
    text,
)


def Integer(name, *args, **kwargs):
    return Column(name, IntegerType(), *args, **kwargs)


def Float(name, *args, **kwargs):
    return Column(name, FloatType(), *args, **kwargs)


def String(name, *args, **kwargs):
    return Column(name, StringType(), *args, **kwargs)


def Text(name, *args, **kwargs):
    return Column(name, TextType(), *args, **kwargs)


metadata = MetaData(
    naming_convention={
        "ix": "ix_%(table_name)s_%(column_0_name)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)

persons = Table(
    "persons",
    metadata,
    Integer("person_id", primary_key=True),
    Text("name", nullable=False, unique=True),
    Text("created_at", nullable=False),
    Text("updated_at", nullable=False),
)

face_embeddings = Table(
    "face_embeddings",
    metadata,
    Integer("embedding_id", primary_key=True),
    Integer("person_id", ForeignKey("persons.person_id")),
    Text("embedding_path", nullable=False),
    Text("image_path"),
    Float("quality", server_default=text("0")),
    Text("created_at", nullable=False),
)

unknown_tracks = Table(
    "unknown_tracks",
    metadata,
    Integer("unknown_id", primary_key=True),
    Text("label", nullable=False),
    Text("image_path"),
    Text("embedding_path"),
    Text("created_at", nullable=False),
    Integer("resolved_person_id", ForeignKey("persons.person_id")),
)

unknown_samples = Table(
    "unknown_samples",
    metadata,
    Integer("sample_id", primary_key=True),
    Integer(
        "unknown_id",
        ForeignKey("unknown_tracks.unknown_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Text("image_path", nullable=False),
    Text("embedding_path", nullable=False),
    Float("quality", server_default=text("0")),
    Float("angle_score", server_default=text("0")),
    Text("created_at", nullable=False),
)

unknown_voices = Table(
    "unknown_voices",
    metadata,
    Integer("unknown_voice_id", primary_key=True),
    Text("label", nullable=False),
    Text("embedding_path"),
    Text("created_at", nullable=False),
    Integer("resolved_person_id", ForeignKey("persons.person_id")),
)

unknown_voice_samples = Table(
    "unknown_voice_samples",
    metadata,
    Integer("sample_id", primary_key=True),
    Integer(
        "unknown_voice_id",
        ForeignKey("unknown_voices.unknown_voice_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Text("embedding_path", nullable=False),
    Text("audio_path"),
    Text("source_ref"),
    Integer("start_ms"),
    Integer("end_ms"),
    Float("quality", server_default=text("0")),
    Text("created_at", nullable=False),
)

Index(
    "idx_unknown_voice_sample_source_window",
    unknown_voice_samples.c.unknown_voice_id,
    unknown_voice_samples.c.source_ref,
    unknown_voice_samples.c.start_ms,
    unknown_voice_samples.c.end_ms,
    unique=True,
    sqlite_where=and_(
        unknown_voice_samples.c.source_ref.is_not(None),
        unknown_voice_samples.c.start_ms.is_not(None),
        unknown_voice_samples.c.end_ms.is_not(None),
    ),
    postgresql_where=and_(
        unknown_voice_samples.c.source_ref.is_not(None),
        unknown_voice_samples.c.start_ms.is_not(None),
        unknown_voice_samples.c.end_ms.is_not(None),
    ),
)

recognition_logs = Table(
    "recognition_logs",
    metadata,
    Integer("log_id", primary_key=True),
    Float("timestamp_sec", nullable=False),
    Integer("person_id", ForeignKey("persons.person_id")),
    Text("person_name"),
    Float("confidence"),
    Text("source", nullable=False),
)

audio_logs = Table(
    "audio_logs",
    metadata,
    Integer("log_id", primary_key=True),
    Float("start_time", nullable=False),
    Float("end_time", nullable=False),
    Integer("person_id", ForeignKey("persons.person_id")),
    Text("person_name"),
    Float("confidence"),
    Text("source", nullable=False),
    Text("transcript"),
    Integer("track_id"),
    Text("source_ref"),
)

attendance_events = Table(
    "attendance_events",
    metadata,
    Integer("event_id", primary_key=True),
    Integer("person_id", ForeignKey("persons.person_id"), nullable=False),
    Text("person_name", nullable=False),
    String("source", nullable=False),
    Text("source_ref", nullable=False),
    Integer("track_id", nullable=False),
    Float("confidence", nullable=False),
    Text("observed_at_utc"),
    Integer("media_offset_ms"),
    Text("created_at", nullable=False),
    UniqueConstraint("source", "source_ref", "track_id", "person_id"),
    CheckConstraint(
        "source IN ('realtime', 'video')",
        name="attendance_source",
    ),
    CheckConstraint(
        "(source = 'realtime' AND observed_at_utc IS NOT NULL AND media_offset_ms IS NULL) "
        "OR (source = 'video' AND observed_at_utc IS NULL AND media_offset_ms IS NOT NULL)",
        name="attendance_source_fields",
    ),
)

meetings = Table(
    "meetings",
    metadata,
    Integer("meeting_id", primary_key=True),
    Text("video_path", nullable=False, unique=True),
    Text("created_at", nullable=False),
    Text("processed_at"),
    Text("transcription_status", nullable=False, server_default=text("'pending'")),
    Text("error_log"),
)

meeting_introductions = Table(
    "meeting_introductions",
    metadata,
    Integer("introduction_id", primary_key=True),
    Integer("meeting_id", ForeignKey("meetings.meeting_id", ondelete="CASCADE"), nullable=False),
    Text("name", nullable=False),
    Text("job_title"),
    Integer("start_ms", nullable=False),
    Integer("end_ms", nullable=False),
    Text("speaker_label", nullable=False),
    Text("source_segment_ids", nullable=False),
    Integer(
        "voice_unknown_id",
        ForeignKey("unknown_voices.unknown_voice_id", ondelete="SET NULL"),
    ),
    Integer("face_track_id"),
    Integer("person_id", ForeignKey("persons.person_id")),
    Float("parser_confidence", nullable=False),
    Float("match_confidence"),
    Text("status", nullable=False),
    Text("reason"),
    Integer(
        "face_unknown_id",
        ForeignKey("unknown_tracks.unknown_id", ondelete="SET NULL"),
    ),
    Text("created_at", nullable=False),
    Text("updated_at", nullable=False),
    UniqueConstraint("meeting_id", "name", "start_ms", "end_ms"),
)

transcription_segments = Table(
    "transcription_segments",
    metadata,
    Integer("segment_id", primary_key=True),
    Integer("meeting_id", ForeignKey("meetings.meeting_id"), nullable=False),
    Text("speaker_label", nullable=False),
    Integer("person_id", ForeignKey("persons.person_id")),
    Integer("start_ms", nullable=False),
    Integer("end_ms", nullable=False),
    Text("text", nullable=False),
    Text("original_text"),
    Float("confidence"),
    Text("sentence_type"),
    Text("created_at", nullable=False),
    Text("word_timestamps"),
)
Index("idx_transcription_meeting", transcription_segments.c.meeting_id)

refined_transcription_segments = Table(
    "refined_transcription_segments",
    metadata,
    Integer("refined_segment_id", primary_key=True),
    Integer(
        "meeting_id",
        ForeignKey("meetings.meeting_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Text("speaker_label", nullable=False),
    Integer("person_id", ForeignKey("persons.person_id")),
    Integer("start_ms", nullable=False),
    Integer("end_ms", nullable=False),
    Text("text", nullable=False),
    Text("source_segment_ids", nullable=False),
    Text("sentence_type"),
    Text("refinement_method", nullable=False),
    Text("created_at", nullable=False),
)
Index(
    "idx_refined_transcription_meeting",
    refined_transcription_segments.c.meeting_id,
)

transcript_refinement_runs = Table(
    "transcript_refinement_runs",
    metadata,
    Integer("meeting_id", ForeignKey("meetings.meeting_id", ondelete="CASCADE"), primary_key=True),
    Text("status", nullable=False),
    Text("method", nullable=False),
    Text("speaker_change_recommendations", nullable=False, server_default=text("'[]'")),
    Text("error"),
    Text("updated_at", nullable=False),
)

voice_embeddings = Table(
    "voice_embeddings",
    metadata,
    Integer("embedding_id", primary_key=True),
    Integer("person_id", ForeignKey("persons.person_id"), nullable=False),
    Text("embedding_path", nullable=False),
    Text("audio_path"),
    Float("quality", server_default=text("0")),
    Text("created_at", nullable=False),
)


__all__ = ["metadata"]