import sys
from pathlib import Path

import cv2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _NullVideoWriter:
    def isOpened(self):
        return True

    def write(self, _frame):
        return None

    def release(self):
        return None


@pytest.fixture(autouse=True)
def _no_disk_video_writes(monkeypatch):
    """Prevent tests from creating real video files via cv2.VideoWriter unless
    a test explicitly overrides it with its own monkeypatch."""

    monkeypatch.setattr(cv2, "VideoWriter", lambda *_args, **_kwargs: _NullVideoWriter())


@pytest.fixture(autouse=True)
def _isolate_voice_and_transcript_dirs(tmp_path, monkeypatch):
    """Keep unknown-voice files and regenerated transcripts out of the real project folders."""
    from face_voice_rec import config

    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(config, "UNKNOWN_VOICES_DIR", tmp_path / "unknown_voices")


@pytest.fixture(autouse=True)
def _isolate_database_path(tmp_path, monkeypatch):
    """Ensure tests never open the default project database path."""
    import database
    from face_voice_rec import config

    db_path = tmp_path / "db" / "faces.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(config, "DB_PATH", db_path)
    database.init_db()
