"""Create the initial relational schema.

Revision ID: 0001_initial
Revises:
"""

from alembic import op
import sqlalchemy as sa


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "meetings",
        sa.Column("meeting_id", sa.Integer(), primary_key=True),
        sa.Column("video_path", sa.Text(), nullable=False, unique=True),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("processed_at", sa.Text()),
        sa.Column("transcription_status", sa.Text(), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("error_log", sa.Text()),
    )
    op.create_table(
        "persons",
        sa.Column("person_id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False, unique=True),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
    )
    op.create_table(
        "unknown_tracks",
        sa.Column("unknown_id", sa.Integer(), primary_key=True),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("image_path", sa.Text()),
        sa.Column("embedding_path", sa.Text()),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("resolved_person_id", sa.Integer(), sa.ForeignKey("persons.person_id")),
    )
    op.create_table(
        "unknown_voices",
        sa.Column("unknown_voice_id", sa.Integer(), primary_key=True),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("embedding_path", sa.Text()),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("resolved_person_id", sa.Integer(), sa.ForeignKey("persons.person_id")),
    )
    op.create_table(
        "attendance_events",
        sa.Column("event_id", sa.Integer(), primary_key=True),
        sa.Column("person_id", sa.Integer(), sa.ForeignKey("persons.person_id"), nullable=False),
        sa.Column("person_name", sa.Text(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("source_ref", sa.Text(), nullable=False),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("observed_at_utc", sa.Text()),
        sa.Column("media_offset_ms", sa.Integer()),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.CheckConstraint("source IN ('realtime', 'video')", name="attendance_source"),
        sa.CheckConstraint(
            "(source = 'realtime' AND observed_at_utc IS NOT NULL AND media_offset_ms IS NULL) "
            "OR (source = 'video' AND observed_at_utc IS NULL AND media_offset_ms IS NOT NULL)",
            name="attendance_source_fields",
        ),
        sa.UniqueConstraint("source", "source_ref", "track_id", "person_id"),
    )
    op.create_table(
        "audio_logs",
        sa.Column("log_id", sa.Integer(), primary_key=True),
        sa.Column("start_time", sa.Float(), nullable=False),
        sa.Column("end_time", sa.Float(), nullable=False),
        sa.Column("person_id", sa.Integer(), sa.ForeignKey("persons.person_id")),
        sa.Column("person_name", sa.Text()),
        sa.Column("confidence", sa.Float()),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("transcript", sa.Text()),
        sa.Column("track_id", sa.Integer()),
    )
    op.create_table(
        "face_embeddings",
        sa.Column("embedding_id", sa.Integer(), primary_key=True),
        sa.Column("person_id", sa.Integer(), sa.ForeignKey("persons.person_id")),
        sa.Column("embedding_path", sa.Text(), nullable=False),
        sa.Column("image_path", sa.Text()),
        sa.Column("quality", sa.Float(), server_default=sa.text("0")),
        sa.Column("created_at", sa.Text(), nullable=False),
    )
    op.create_table(
        "recognition_logs",
        sa.Column("log_id", sa.Integer(), primary_key=True),
        sa.Column("timestamp_sec", sa.Float(), nullable=False),
        sa.Column("person_id", sa.Integer(), sa.ForeignKey("persons.person_id")),
        sa.Column("person_name", sa.Text()),
        sa.Column("confidence", sa.Float()),
        sa.Column("source", sa.Text(), nullable=False),
    )
    op.create_table(
        "voice_embeddings",
        sa.Column("embedding_id", sa.Integer(), primary_key=True),
        sa.Column("person_id", sa.Integer(), sa.ForeignKey("persons.person_id"), nullable=False),
        sa.Column("embedding_path", sa.Text(), nullable=False),
        sa.Column("audio_path", sa.Text()),
        sa.Column("quality", sa.Float(), server_default=sa.text("0")),
        sa.Column("created_at", sa.Text(), nullable=False),
    )
    op.create_table(
        "unknown_samples",
        sa.Column("sample_id", sa.Integer(), primary_key=True),
        sa.Column("unknown_id", sa.Integer(), sa.ForeignKey("unknown_tracks.unknown_id", ondelete="CASCADE"), nullable=False),
        sa.Column("image_path", sa.Text(), nullable=False),
        sa.Column("embedding_path", sa.Text(), nullable=False),
        sa.Column("quality", sa.Float(), server_default=sa.text("0")),
        sa.Column("angle_score", sa.Float(), server_default=sa.text("0")),
        sa.Column("created_at", sa.Text(), nullable=False),
    )
    op.create_table(
        "unknown_voice_samples",
        sa.Column("sample_id", sa.Integer(), primary_key=True),
        sa.Column("unknown_voice_id", sa.Integer(), sa.ForeignKey("unknown_voices.unknown_voice_id", ondelete="CASCADE"), nullable=False),
        sa.Column("embedding_path", sa.Text(), nullable=False),
        sa.Column("audio_path", sa.Text()),
        sa.Column("source_ref", sa.Text()),
        sa.Column("start_ms", sa.Integer()),
        sa.Column("end_ms", sa.Integer()),
        sa.Column("quality", sa.Float(), server_default=sa.text("0")),
        sa.Column("created_at", sa.Text(), nullable=False),
    )
    op.create_index(
        "idx_unknown_voice_sample_source_window",
        "unknown_voice_samples",
        ["unknown_voice_id", "source_ref", "start_ms", "end_ms"],
        unique=True,
        sqlite_where=sa.text("source_ref IS NOT NULL AND start_ms IS NOT NULL AND end_ms IS NOT NULL"),
        postgresql_where=sa.text("source_ref IS NOT NULL AND start_ms IS NOT NULL AND end_ms IS NOT NULL"),
    )
    op.create_table(
        "meeting_introductions",
        sa.Column("introduction_id", sa.Integer(), primary_key=True),
        sa.Column("meeting_id", sa.Integer(), sa.ForeignKey("meetings.meeting_id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("job_title", sa.Text()),
        sa.Column("start_ms", sa.Integer(), nullable=False),
        sa.Column("end_ms", sa.Integer(), nullable=False),
        sa.Column("speaker_label", sa.Text(), nullable=False),
        sa.Column("source_segment_ids", sa.Text(), nullable=False),
        sa.Column("voice_unknown_id", sa.Integer(), sa.ForeignKey("unknown_voices.unknown_voice_id", ondelete="SET NULL")),
        sa.Column("face_track_id", sa.Integer()),
        sa.Column("person_id", sa.Integer(), sa.ForeignKey("persons.person_id")),
        sa.Column("parser_confidence", sa.Float(), nullable=False),
        sa.Column("match_confidence", sa.Float()),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text()),
        sa.Column("face_unknown_id", sa.Integer(), sa.ForeignKey("unknown_tracks.unknown_id", ondelete="SET NULL")),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.UniqueConstraint("meeting_id", "name", "start_ms", "end_ms"),
    )
    op.create_table(
        "transcription_segments",
        sa.Column("segment_id", sa.Integer(), primary_key=True),
        sa.Column("meeting_id", sa.Integer(), sa.ForeignKey("meetings.meeting_id"), nullable=False),
        sa.Column("speaker_label", sa.Text(), nullable=False),
        sa.Column("person_id", sa.Integer(), sa.ForeignKey("persons.person_id")),
        sa.Column("start_ms", sa.Integer(), nullable=False),
        sa.Column("end_ms", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("original_text", sa.Text()),
        sa.Column("confidence", sa.Float()),
        sa.Column("sentence_type", sa.Text()),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("word_timestamps", sa.Text()),
    )
    op.create_index("idx_transcription_meeting", "transcription_segments", ["meeting_id"])
    op.create_table(
        "refined_transcription_segments",
        sa.Column("refined_segment_id", sa.Integer(), primary_key=True),
        sa.Column("meeting_id", sa.Integer(), sa.ForeignKey("meetings.meeting_id", ondelete="CASCADE"), nullable=False),
        sa.Column("speaker_label", sa.Text(), nullable=False),
        sa.Column("person_id", sa.Integer(), sa.ForeignKey("persons.person_id")),
        sa.Column("start_ms", sa.Integer(), nullable=False),
        sa.Column("end_ms", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("source_segment_ids", sa.Text(), nullable=False),
        sa.Column("sentence_type", sa.Text()),
        sa.Column("refinement_method", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
    )
    op.create_index(
        "idx_refined_transcription_meeting",
        "refined_transcription_segments",
        ["meeting_id"],
    )
    op.create_table(
        "transcript_refinement_runs",
        sa.Column("meeting_id", sa.Integer(), sa.ForeignKey("meetings.meeting_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("method", sa.Text(), nullable=False),
        sa.Column("speaker_change_recommendations", sa.Text(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("error", sa.Text()),
        sa.Column("updated_at", sa.Text(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("transcript_refinement_runs")
    op.drop_index("idx_refined_transcription_meeting", table_name="refined_transcription_segments")
    op.drop_table("refined_transcription_segments")
    op.drop_index("idx_transcription_meeting", table_name="transcription_segments")
    op.drop_table("transcription_segments")
    op.drop_table("meeting_introductions")
    op.drop_index("idx_unknown_voice_sample_source_window", table_name="unknown_voice_samples")
    op.drop_table("unknown_voice_samples")
    op.drop_table("unknown_samples")
    op.drop_table("voice_embeddings")
    op.drop_table("recognition_logs")
    op.drop_table("face_embeddings")
    op.drop_table("audio_logs")
    op.drop_table("attendance_events")
    op.drop_table("unknown_voices")
    op.drop_table("unknown_tracks")
    op.drop_table("persons")
    op.drop_table("meetings")