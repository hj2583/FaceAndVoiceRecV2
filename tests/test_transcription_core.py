from pathlib import Path

import config
import database

from transcription_core import (
    TranscriptionSegment,
    apply_text_cleanup,
    process_meeting_transcription,
    save_transcripts,
)


def test_cleanup_normalizes_whitespace():
    assert apply_text_cleanup("  hello\n world  ") == "hello world"


def test_cleanup_none_preserves_text():
    assert apply_text_cleanup("  hello  ", "none") == "  hello  "


def test_save_transcripts_groups_by_speaker(tmp_path: Path):
    segments = [
        TranscriptionSegment("Speaker A", 65_000, 70_000, "later"),
        TranscriptionSegment("Speaker A", 1_000, 3_000, " first "),
        TranscriptionSegment("Speaker B", 4_000, 5_000, "second"),
    ]

    paths = save_transcripts(segments, tmp_path)

    assert set(paths) == {"Speaker A", "Speaker B"}
    assert paths["Speaker A"].read_text(encoding="utf-8").splitlines() == [
        "[00:01] first",
        "[01:05] later",
    ]


def test_process_meeting_transcription_persists_completed_segments(
    tmp_path: Path,
    monkeypatch,
):
    db_path = tmp_path / "meeting.db"
    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    database.init_db()

    video_path = tmp_path / "meeting.mp4"
    video_path.write_bytes(b"video")

    def fake_extract(_video_path, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"audio")
        return output_path

    monkeypatch.setattr(
        "transcription_core.extract_audio_from_video",
        fake_extract,
    )
    monkeypatch.setattr(
        "transcription_core.transcribe_with_diarization",
        lambda *_args, **_kwargs: [
            TranscriptionSegment("Speaker A", 0, 1500, " hello ", 0.8),
        ],
    )

    meeting_id = process_meeting_transcription(video_path)

    with database.get_conn() as connection:
        status = connection.execute(
            "SELECT transcription_status FROM meetings WHERE meeting_id=?",
            (meeting_id,),
        ).fetchone()[0]
        segments = connection.execute(
            "SELECT speaker_label, text FROM transcription_segments WHERE meeting_id=?",
            (meeting_id,),
        ).fetchall()

    assert status == "completed"
    assert segments == [("Speaker A", "hello")]
    assert (config.TRANSCRIPTS_DIR / str(meeting_id) / "speaker_a.txt").exists()