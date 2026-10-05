from pathlib import Path

import config
import database

from transcription_core import (
    TranscriptionSegment,
    absorb_minor_speakers,
    apply_text_cleanup,
    build_face_event_diarizer,
    classify_segments,
    classify_sentence,
    group_transcript_turns,
    merge_speaker_sentences,
    process_meeting_transcription,
    save_transcripts,
)


def test_merge_joins_whole_same_speaker_turn():
    segments = [
        TranscriptionSegment("Alice", 0, 1000, "So the budget", 0.8),
        TranscriptionSegment("Alice", 1200, 2000, "is approved.", 0.6),
        TranscriptionSegment("Alice", 2100, 3000, "Next point", 0.9),
        TranscriptionSegment("Bob", 3100, 4000, "Okay", 0.9),
    ]

    merged = merge_speaker_sentences(segments, max_gap_ms=1500, max_chars=400)

    assert [(s.speaker_label, s.text) for s in merged] == [
        ("Alice", "So the budget is approved. Next point"),
        ("Bob", "Okay"),
    ]
    assert merged[0].start_ms == 0 and merged[0].end_ms == 3000


def test_merge_weights_confidence_by_duration():
    merged = merge_speaker_sentences([
        TranscriptionSegment("Alice", 0, 1000, "So the budget", 0.8),
        TranscriptionSegment("Alice", 1200, 2000, "is approved.", 0.6),
    ], max_gap_ms=1500, max_chars=400)

    assert abs(merged[0].confidence - (0.8 * 1000 + 0.6 * 800) / 1800) < 1e-6


def test_merge_long_turn_only_splits_at_sentence_boundary():
    segments = [
        TranscriptionSegment("Alice", 0, 1000, "a" * 10),
        TranscriptionSegment("Alice", 1000, 2000, "b" * 10),
        TranscriptionSegment("Alice", 2000, 3000, "end."),
        TranscriptionSegment("Alice", 3000, 4000, "next"),
    ]

    merged = merge_speaker_sentences(segments, max_gap_ms=1500, max_chars=15)

    assert [s.text for s in merged] == [f"{'a' * 10} {'b' * 10} end.", "next"]


def test_merge_does_not_join_across_long_pause():
    segments = [
        TranscriptionSegment("Alice", 0, 1000, "first part"),
        TranscriptionSegment("Alice", 5000, 6000, "second part"),
    ]

    assert len(merge_speaker_sentences(segments, max_gap_ms=1500, max_chars=400)) == 2


def test_group_transcript_turns_keeps_adjacent_same_speaker_entries_together():
    segments = [
        TranscriptionSegment("Alice", 0, 1000, "First sentence.", sentence_type="Comment"),
        TranscriptionSegment("Alice", 2000, 3000, "Second sentence.", sentence_type="Question"),
        TranscriptionSegment("Bob", 4000, 5000, "Reply.", sentence_type="Comment"),
        TranscriptionSegment("Alice", 20_000, 21_000, "Later turn.", sentence_type="Comment"),
    ]

    turns = group_transcript_turns(segments, max_gap_ms=10_000)

    assert [[segment.text for segment in turn] for turn in turns] == [
        ["First sentence.", "Second sentence."],
        ["Reply."],
        ["Later turn."],
    ]


def test_classify_sentence_rules():
    assert classify_sentence("What is the timeline?") == "Question"
    assert classify_sentence("Okay, so when do we start? I think May.") == "Question"
    assert classify_sentence("Bilakah projek ini siap") == "Question"
    assert classify_sentence("Moving on to the marketing plan.") == "Topic"
    assert classify_sentence("Welcome to PBE Tutorial. In this video, we will guide you.") == "Topic"
    assert classify_sentence("The sales grew ten percent last quarter.") == "Comment"
    assert classify_sentence("Okay.") == "Comment"
    assert classify_sentence("2.") == "Unknown"
    assert classify_sentence("Um, uh...") == "Unknown"


def test_absorb_minor_speakers_relabels_stray_unknown_fragments():
    segments = [
        TranscriptionSegment("Unknown Speaker 35", 0, 20_000, "Long intro"),
        TranscriptionSegment("Unknown Speaker", 20_500, 21_000, "1."),
        TranscriptionSegment("Unknown Speaker 36", 21_500, 23_000, "Display screen. The"),
        TranscriptionSegment("Unknown Speaker 35", 23_000, 40_000, "token switches off."),
        TranscriptionSegment("Alice", 41_000, 42_000, "Hi."),
    ]

    result = absorb_minor_speakers(segments, blip_ms=3000, minor_total_ms=8000)

    assert [s.speaker_label for s in result] == [
        "Unknown Speaker 35",
        "Unknown Speaker 35",
        "Unknown Speaker 35",
        "Unknown Speaker 35",
        "Alice",
    ]


def test_classify_segments_labels_each_turn():
    classified = classify_segments([
        TranscriptionSegment("Alice", 0, 1000, "When is the launch?"),
        TranscriptionSegment("Bob", 1000, 2000, "Early next month."),
    ])

    assert [s.sentence_type for s in classified] == ["Question", "Comment"]


def test_cleanup_normalizes_whitespace():
    assert apply_text_cleanup("  hello\n world  ") == "hello world"


def test_cleanup_none_preserves_text():
    assert apply_text_cleanup("  hello  ", "none") == "  hello  "


def test_face_event_diarizer_fills_uncovered_speech_with_unknown():
    diarize = build_face_event_diarizer([
        {"speaker": "Alice", "start_time": 1.0, "end_time": 2.0},
    ])

    assert diarize(Path("audio.wav"), [(0.0, 3.0)]) == [
        ("UNKNOWN", 0.0, 1.0),
        ("Alice", 1.0, 2.0),
        ("UNKNOWN", 2.0, 3.0),
    ]


def test_speaker_interval_requires_minimum_overlap_coverage():
    from transcription_core import _speaker_for_interval

    events = [("Alice", 0.0, 0.6), ("Bob", 0.6, 1.0)]

    assert _speaker_for_interval(0.0, 1.0, events, 0.8) == "UNKNOWN"
    assert _speaker_for_interval(0.0, 0.7, events, 0.8) == "Alice"


def test_empty_face_diarization_with_threshold_is_unknown():
    from transcription_core import _speaker_for_interval

    assert _speaker_for_interval(3.0, 4.0, [], 0.8) == "UNKNOWN"


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


def test_transcribe_with_diarization_preserves_segment_when_words_have_single_speaker_consensus(tmp_path, monkeypatch):
    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(b"fake wav")
    monkeypatch.setattr("transcription_core.detect_speech_segments", lambda *_a, **_k: [(0.0, 10.0)])

    class FakeModel:
        def transcribe(self, path, verbose=False, **kwargs):
            return {
                "segments": [{
                    "start": 0.0,
                    "end": 10.0,
                    "text": "Hello there everyone.",
                    "avg_logprob": -0.1,
                    "words": [
                        {"word": " Hello", "start": 2.0, "end": 2.4},
                        {"word": " there", "start": 2.4, "end": 2.8},
                    ],
                }]
            }

    fake_whisper = type("FakeWhisperModule", (), {"load_model": staticmethod(lambda *_a, **_k: FakeModel())})
    monkeypatch.setitem(__import__("sys").modules, "whisper", fake_whisper)

    from transcription_core import transcribe_with_diarization

    segments = transcribe_with_diarization(
        wav_path,
        diarize=lambda *_a: [("Alice", 2.0, 2.8)],
        minimum_speaker_overlap=0.8,
    )

    assert len(segments) == 1
    assert segments[0].speaker_label == "Alice"
    assert segments[0].start_ms == 0
    assert segments[0].end_ms == 10_000
    assert segments[0].text == "Hello there everyone."


def test_update_transcription_segments_saves_corrections_and_rewrites_files(tmp_path, monkeypatch):
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
    monkeypatch.setattr(
        "transcription_core.transcribe_with_diarization",
        lambda *_a, **_k: [TranscriptionSegment("Ivy", 0, 1000, "helo wrld.", 0.4)],
    )
    meeting_id = process_meeting_transcription(video_path)

    with database.get_conn() as connection:
        segment_id = connection.execute(
            "SELECT segment_id FROM transcription_segments WHERE meeting_id=?",
            (meeting_id,),
        ).fetchone()[0]

    database.update_transcription_segments(meeting_id, [{
        "segment_id": segment_id,
        "text": "Hello world.",
        "sentence_type": "Topic",
        "confidence": 1.0,
    }])

    with database.get_conn() as connection:
        row = connection.execute(
            "SELECT text, sentence_type, confidence FROM transcription_segments WHERE segment_id=?",
            (segment_id,),
        ).fetchone()
    assert row == ("Hello world.", "Topic", 1.0)
    assert (config.TRANSCRIPTS_DIR / str(meeting_id) / "ivy.txt").read_text(encoding="utf-8") == "[00:00] Hello world.\n"


def test_reanalyze_meeting_segments_merges_and_classifies_old_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "meeting.db"
    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    database.init_db()

    from transcription_core import reanalyze_meeting_segments

    with database.get_conn() as connection:
        connection.execute(
            "INSERT INTO meetings(video_path, created_at, transcription_status) VALUES ('v.mp4', 'now', 'completed')"
        )
        meeting_id = connection.execute("SELECT meeting_id FROM meetings").fetchone()[0]
        connection.executemany(
            """
            INSERT INTO transcription_segments(meeting_id, speaker_label, start_ms, end_ms, text, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'now')
            """,
            [
                (meeting_id, "Speaker 35", 6000, 9000, "video, we will guide you on how to use the token", 0.9),
                (meeting_id, "Speaker 35", 9000, 16000, "and perform signing. The token is a security device", 0.9),
                (meeting_id, "Speaker 35", 16000, 24000, "so never reveal the pin to anyone.", 0.7),
            ],
        )

    assert reanalyze_meeting_segments(meeting_id) == 1

    with database.get_conn() as connection:
        rows = connection.execute(
            "SELECT start_ms, end_ms, sentence_type FROM transcription_segments WHERE meeting_id=?",
            (meeting_id,),
        ).fetchall()
    assert rows == [(6000, 24000, "Topic")]


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


def test_process_meeting_transcription_does_not_run_english_translation(
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

    monkeypatch.setattr("transcription_core.extract_audio_from_video", fake_extract)
    calls = []

    def fake_transcribe(*_args, **kwargs):
        calls.append(kwargs.get("task"))
        text = "Hello" if kwargs.get("task") == "translate" else "Hola"
        return [TranscriptionSegment("Alice", 0, 1000, text, 0.9)]

    monkeypatch.setattr("transcription_core.transcribe_with_diarization", fake_transcribe)

    meeting_id = process_meeting_transcription(video_path)

    meeting_dir = config.TRANSCRIPTS_DIR / str(meeting_id)
    assert calls == [None]
    assert (meeting_dir / "alice.txt").read_text(encoding="utf-8") == "[00:00] Hola\n"
    assert not (meeting_dir / "english_translation").exists()


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


def test_subtitle_word_callback_receives_face_attributed_word_timestamps(tmp_path, monkeypatch):
    from transcription_core import SubtitleWord, transcribe_with_diarization

    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(b"fake wav")
    monkeypatch.setattr("transcription_core.detect_speech_segments", lambda *_a: [(0.0, 1.0)])

    class FakeModel:
        def transcribe(self, _path, **_kwargs):
            return {"segments": [{
                "start": 0.0, "end": 1.0, "text": " hello there", "avg_logprob": -0.1,
                "words": [
                    {"word": " hello", "start": 0.1, "end": 0.4},
                    {"word": " there", "start": 0.5, "end": 0.8},
                ],
            }]}

    fake_whisper = type("FakeWhisperModule", (), {
        "load_model": staticmethod(lambda *_a, **_k: FakeModel())
    })
    monkeypatch.setitem(__import__("sys").modules, "whisper", fake_whisper)
    observed = []

    segments = transcribe_with_diarization(
        wav_path,
        diarize=lambda *_a: [("Alice", 0.0, 0.45), ("Bob", 0.45, 1.0)],
        minimum_speaker_overlap=0.8,
        subtitle_word_callback=observed.append,
    )

    assert observed == [
        SubtitleWord("Alice", 100, 400, "hello"),
        SubtitleWord("Bob", 500, 800, "there"),
    ]
    assert [(segment.speaker_label, segment.text) for segment in segments] == [
        ("Alice", "hello"), ("Bob", "there"),
    ]


def test_subtitle_callback_estimates_word_timings_when_word_timings_are_missing(tmp_path, monkeypatch):
    from transcription_core import SubtitleWord, transcribe_with_diarization

    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(b"fake wav")
    monkeypatch.setattr("transcription_core.detect_speech_segments", lambda *_a: [(0.0, 1.0)])

    class FakeModel:
        def transcribe(self, _path, **_kwargs):
            return {"segments": [{
                "start": 0.0, "end": 1.0, "text": " hello world", "avg_logprob": -0.1,
            }]}

    fake_whisper = type("FakeWhisperModule", (), {
        "load_model": staticmethod(lambda *_a, **_k: FakeModel())
    })
    monkeypatch.setitem(__import__("sys").modules, "whisper", fake_whisper)
    observed = []

    transcribe_with_diarization(
        wav_path,
        diarize=lambda *_a: [("Alice", 0.0, 1.0)],
        subtitle_word_callback=observed.append,
    )

    assert observed == [
        SubtitleWord("Alice", 0, 500, "hello"),
        SubtitleWord("Alice", 500, 1000, "world"),
    ]


def test_process_meeting_transcription_forwards_exact_subtitle_callback(tmp_path, monkeypatch):
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

    captured = {}

    def fake_transcribe(_audio_path, **kwargs):
        captured["subtitle_word_callback"] = kwargs.get("subtitle_word_callback")
        return [TranscriptionSegment("Speaker A", 0, 1000, "hello", 0.9)]

    callback_events = []
    callback = callback_events.append

    monkeypatch.setattr("transcription_core.extract_audio_from_video", fake_extract)
    monkeypatch.setattr("transcription_core.transcribe_with_diarization", fake_transcribe)

    meeting_id = process_meeting_transcription(
        video_path,
        subtitle_word_callback=callback,
    )

    assert meeting_id > 0
    assert captured["subtitle_word_callback"] is callback