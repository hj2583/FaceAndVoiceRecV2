from contextlib import nullcontext
import importlib
import sqlite3
import sys

import pytest

from face_voice_rec import config
from face_voice_rec.storage import database


class _FakeStreamlit:
    def __init__(self):
        self.button_labels = []
        self.captions = []
        self.infos = []
        self.warnings = []

    def header(self, _value):
        return None

    def subheader(self, _value):
        return None

    def info(self, value):
        self.infos.append(value)

    def warning(self, value):
        self.warnings.append(value)

    def caption(self, value):
        self.captions.append(value)

    def selectbox(self, _label, options, format_func=None, **_kwargs):
        options = list(options)
        value = options[0]
        if format_func is not None:
            _ = format_func(value)
        return value

    def button(self, label, **_kwargs):
        self.button_labels.append(label)
        return False

    def spinner(self, _label):
        return nullcontext()

    def rerun(self):
        return None

    def radio(self, _label, options, **_kwargs):
        return list(options)[0]

    def multiselect(self, _label, _options, **_kwargs):
        return []

    def container(self):
        return nullcontext()

    def markdown(self, _value, **_kwargs):
        return None

    def download_button(self, *_args, **_kwargs):
        return None


def _load_transcripts_module(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.delitem(sys.modules, "app.pages.transcripts", raising=False)
    module = importlib.import_module("app.pages.transcripts")
    monkeypatch.setattr(module, "st", fake_st)
    return module, fake_st


def _seed_minimal_meeting(db_path, *, refinement_status, refinement_method, refinement_error=None):
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO meetings(video_path, created_at, transcription_status) "
            "VALUES ('demo.mp4', 'now', 'completed')"
        )
        meeting_id = connection.execute("SELECT meeting_id FROM meetings").fetchone()[0]
        connection.execute(
            """
            INSERT INTO transcription_segments(
                meeting_id, speaker_label, start_ms, end_ms, text, original_text,
                confidence, sentence_type, word_timestamps, created_at
            ) VALUES (?, 'Alice', 0, 1000, 'Hello there.', 'Hello there.', 0.95, 'Comment', NULL, 'now')
            """,
            (meeting_id,),
        )
        connection.execute(
            """
            INSERT INTO refined_transcription_segments(
                meeting_id, speaker_label, person_id, start_ms, end_ms, text,
                source_segment_ids, sentence_type, refinement_method, created_at
            ) VALUES (?, 'Alice', NULL, 0, 1000, 'Hello there.', '[1]', 'Comment', ?, 'now')
            """,
            (meeting_id, refinement_method),
        )
        connection.execute(
            """
            INSERT INTO transcript_refinement_runs(
                meeting_id, status, method, speaker_change_recommendations, error, updated_at
            ) VALUES (?, ?, ?, '[]', ?, 'now')
            """,
            (meeting_id, refinement_status, refinement_method, refinement_error),
        )


@pytest.mark.parametrize(
    "status,method,expected",
    [
        ("complete", "foundry:test-deployment", "Cloud"),
        ("complete", "ollama:test-local", "Local Ollama"),
        ("complete_local_fallback", "ollama:test-local", "cloud fallback"),
        ("fallback", "deterministic", "Deterministic"),
        ("disabled", "deterministic", "Deterministic"),
    ],
)
def test_caption_uses_saved_provenance(monkeypatch, status, method, expected):
    module, _ = _load_transcripts_module(monkeypatch)

    assert expected in module._refinement_status_caption(status, method)


def test_render_uses_configured_provider_action_text_and_saved_fallback_details(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "meeting.db"
    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(database, "DB_PATH", db_path)
    database.init_db()
    _seed_minimal_meeting(
        db_path,
        refinement_status="complete_local_fallback",
        refinement_method="ollama:test-local",
        refinement_error="Foundry request failed (HTTP 429)",
    )

    module, fake_st = _load_transcripts_module(monkeypatch)
    monkeypatch.setattr(module, "DB_PATH", db_path)
    monkeypatch.setattr(module, "TRANSCRIPT_REFINEMENT_ENABLED", True)
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_USE_CLOUD", True)

    module.render()

    assert any("cloud" in label.lower() for label in fake_st.button_labels)
    assert any("cloud fallback" in caption.lower() for caption in fake_st.captions)
    assert any("Foundry request failed (HTTP 429)" in info for info in fake_st.infos)
    assert fake_st.warnings == []
