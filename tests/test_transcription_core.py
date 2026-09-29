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


def test_transcribe_with_diarization_only_transcribes_vad_speech_regions(tmp_path, monkeypatch):
    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(b"fake wav")

    monkeypatch.setattr(
        "transcription_core.detect_speech_segments",
        lambda *_a, **_k: [(0.0, 2.0), (5.0, 7.0)],
    )

    calls = []

    class FakeModel:
        def transcribe(self, path, verbose=False, **kwargs):
            calls.append(kwargs)
            return {
                "segments": [
                    {"start": 0.1, "end": 1.0, "text": "hello", "avg_logprob": -0.1},
                ]
            }

    fake_whisper = type("FakeWhisperModule", (), {"load_model": staticmethod(lambda *_a, **_k: FakeModel())})
    monkeypatch.setitem(__import__("sys").modules, "whisper", fake_whisper)

    from transcription_core import transcribe_with_diarization

    segments = transcribe_with_diarization(wav_path)

    assert segments[0].text == "hello"
    assert calls[0]["condition_on_previous_text"] is False
    assert calls[0]["word_timestamps"] is True
    assert calls[0]["task"] == "transcribe"


def test_fixed_language_mode_passes_configured_language(tmp_path, monkeypatch):
    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(b"fake wav")
    monkeypatch.setattr("transcription_core.detect_speech_segments", lambda *_a, **_k: [(0.0, 2.0)])
    monkeypatch.setattr(config, "WHISPER_LANGUAGE_MODE", "fixed")
    monkeypatch.setattr(config, "WHISPER_LANGUAGE", "en")

    calls = []

    class FakeModel:
        def transcribe(self, path, verbose=False, **kwargs):
            calls.append(kwargs)
            return {"segments": []}

    fake_whisper = type("FakeWhisperModule", (), {"load_model": staticmethod(lambda *_a, **_k: FakeModel())})
    monkeypatch.setitem(__import__("sys").modules, "whisper", fake_whisper)

    from transcription_core import transcribe_with_diarization

    transcribe_with_diarization(wav_path, diarize=lambda *_a: [])

    assert calls[0]["language"] == "en"


def test_transcribe_with_diarization_degrades_gracefully_when_vad_fails(tmp_path, monkeypatch):
    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(b"fake wav")

    def _raise(*_a, **_k):
        raise RuntimeError("silero-vad unavailable")

    monkeypatch.setattr("transcription_core.detect_speech_segments", _raise)

    class FakeModel:
        def transcribe(self, path, verbose=False, **kwargs):
            return {
                "segments": [
                    {"start": 0.1, "end": 1.0, "text": "hello", "avg_logprob": -0.1},
                ]
            }

    fake_whisper = type("FakeWhisperModule", (), {"load_model": staticmethod(lambda *_a, **_k: FakeModel())})
    monkeypatch.setitem(__import__("sys").modules, "whisper", fake_whisper)

    from transcription_core import transcribe_with_diarization

    segments = transcribe_with_diarization(wav_path)

    assert len(segments) == 1
    assert segments[0].text == "hello"
    assert segments[0].speaker_label == "Unknown Speaker"


def test_transcribe_with_diarization_picks_speaker_with_most_total_overlap(tmp_path, monkeypatch):
    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(b"fake wav")
    monkeypatch.setattr("transcription_core.detect_speech_segments", lambda *_a, **_k: [(0.0, 3.0)])

    class FakeModel:
        def transcribe(self, path, verbose=False, **kwargs):
            return {"segments": [{"start": 0.0, "end": 3.0, "text": "hello", "avg_logprob": -0.1}]}

    fake_whisper = type("FakeWhisperModule", (), {"load_model": staticmethod(lambda *_a, **_k: FakeModel())})
    monkeypatch.setitem(__import__("sys").modules, "whisper", fake_whisper)

    from transcription_core import transcribe_with_diarization

    # Overlapping sliding windows: each overlaps the segment by 1.5s, but B covers more in total.
    segments = transcribe_with_diarization(
        wav_path,
        diarize=lambda *_a: [("A", 0.0, 1.5), ("B", 0.75, 2.25), ("B", 1.5, 3.0)],
    )

    assert segments[0].speaker_label == "B"


def test_transcribe_with_diarization_splits_word_timestamps_at_speaker_change(tmp_path, monkeypatch):
    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(b"fake wav")
    monkeypatch.setattr("transcription_core.detect_speech_segments", lambda *_a, **_k: [(0.0, 4.0)])

    class FakeModel:
        def transcribe(self, path, verbose=False, **kwargs):
            return {
                "segments": [{
                    "start": 0.0,
                    "end": 4.0,
                    "text": " Hello Bob",
                    "avg_logprob": -0.1,
                    "words": [
                        {"word": " Hello", "start": 0.0, "end": 1.0},
                        {"word": " Bob", "start": 2.0, "end": 3.0},
                    ],
                }]
            }

    fake_whisper = type("FakeWhisperModule", (), {"load_model": staticmethod(lambda *_a, **_k: FakeModel())})
    monkeypatch.setitem(__import__("sys").modules, "whisper", fake_whisper)

    from transcription_core import transcribe_with_diarization

    segments = transcribe_with_diarization(
        wav_path,
        diarize=lambda *_a: [("Alice", 0.0, 1.5), ("Bob", 1.5, 4.0)],
    )

    assert [(segment.speaker_label, segment.text) for segment in segments] == [
        ("Alice", "Hello"),
        ("Bob", "Bob"),
    ]


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


def test_process_meeting_transcription_writes_optional_english_translation(
    tmp_path: Path,
    monkeypatch,
):
    db_path = tmp_path / "meeting.db"
    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(config, "WHISPER_TRANSLATION_ENABLED", True)
    database.init_db()

    video_path = tmp_path / "meeting.mp4"
    video_path.write_bytes(b"video")

    def fake_extract(_video_path, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"audio")
        return output_path

    monkeypatch.setattr("transcription_core.extract_audio_from_video", fake_extract)
    calls = []

    def fake_transcribe(*_args, **kwargs):
        calls.append(kwargs.get("task"))
        text = "Hello" if kwargs.get("task") == "translate" else "Hola"
        return [TranscriptionSegment("Alice", 0, 1000, text, 0.9)]

    monkeypatch.setattr("transcription_core.transcribe_with_diarization", fake_transcribe)

    meeting_id = process_meeting_transcription(video_path)

    meeting_dir = config.TRANSCRIPTS_DIR / str(meeting_id)
    assert calls == [None, "translate"]
    assert (meeting_dir / "alice.txt").read_text(encoding="utf-8") == "[00:00] Hola\n"
    assert (meeting_dir / "english_translation" / "alice.txt").read_text(encoding="utf-8") == "[00:00] Hello\n"


def test_process_meeting_transcription_uses_voice_core_diarizer_by_default(tmp_path, monkeypatch):
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

    monkeypatch.setattr("transcription_core.extract_audio_from_video", fake_extract)
    monkeypatch.setattr("transcription_core.detect_speech_segments", lambda *_a, **_k: [(0.0, 1.5)])

    diarize_calls = []

    def fake_diarize(wav_path, speech_regions):
        diarize_calls.append((wav_path, speech_regions))
        return [("Gina", 0.0, 1.5)]

    monkeypatch.setattr("voice_core.diarize_meeting_audio", fake_diarize)

    class FakeModel:
        def transcribe(self, path, verbose=False, **kwargs):
            return {"segments": [{"start": 0.1, "end": 1.0, "text": "hi", "avg_logprob": -0.1}]}

    fake_whisper = type("FakeWhisperModule", (), {"load_model": staticmethod(lambda *_a, **_k: FakeModel())})
    monkeypatch.setitem(__import__("sys").modules, "whisper", fake_whisper)

    meeting_id = process_meeting_transcription(video_path)

    assert diarize_calls  # the default diarizer was invoked
    with database.get_conn() as connection:
        row = connection.execute(
            "SELECT speaker_label FROM transcription_segments WHERE meeting_id=?",
            (meeting_id,),
        ).fetchone()
    assert row[0] == "Gina"


def test_process_meeting_transcription_populates_person_id_for_known_speaker(tmp_path, monkeypatch):
    db_path = tmp_path / "meeting.db"
    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    database.init_db()
    person_id = database.create_person("Henry")

    video_path = tmp_path / "meeting.mp4"
    video_path.write_bytes(b"video")

    def fake_extract(_video_path, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"audio")
        return output_path

    monkeypatch.setattr("transcription_core.extract_audio_from_video", fake_extract)
    monkeypatch.setattr(
        "transcription_core.transcribe_with_diarization",
        lambda *_a, **_k: [TranscriptionSegment("Henry", 0, 1000, "hello", 0.9)],
    )

    meeting_id = process_meeting_transcription(video_path)

    with database.get_conn() as connection:
        row = connection.execute(
            "SELECT speaker_label, person_id FROM transcription_segments WHERE meeting_id=?",
            (meeting_id,),
        ).fetchone()

    assert row == ("Henry", person_id)