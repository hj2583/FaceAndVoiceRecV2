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
    import config

    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(config, "UNKNOWN_VOICES_DIR", tmp_path / "unknown_voices")
