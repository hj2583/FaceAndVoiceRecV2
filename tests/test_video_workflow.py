import json
from pathlib import Path

import pytest

import config
from subtitle_renderer import SubtitleWord, VideoMuxError
import video_workflow


def test_workflow_passes_face_events_to_transcription(tmp_path, monkeypatch):
    video_path = tmp_path / "meeting.mp4"
    video_path.write_bytes(b"video")
    output_path = tmp_path / "tracked.mp4"
    log_path = tmp_path / "video.json"
    expected_events = [{
        "speaker": "Alice",
        "person_id": 7,
        "track_id": 3,
        "start_time": 1.0,
        "end_time": 2.0,
        "confidence": 0.9,
        "source": "video",
    }]

    def fake_process(_source, annotated_path, _log_path, **_kwargs):
        Path(annotated_path).write_bytes(b"annotated")
        log_path.write_text(json.dumps({"speech": expected_events}), encoding="utf-8")
        return True

    calls = []

    def fake_transcribe(path, diarize=None, minimum_speaker_overlap=0.0, subtitle_word_callback=None, **_kwargs):
        calls.append((path, diarize, minimum_speaker_overlap))
        if subtitle_word_callback is not None:
            subtitle_word_callback(SubtitleWord("Alice", 100, 300, "hello"))
        return 14

    def fake_mux(_annotated, _source, output, subtitle_path=None, **_kwargs):
        assert subtitle_path is not None
        Path(output).write_bytes(b"final")
        return Path(output)

    monkeypatch.setattr(video_workflow, "process_video_pipeline", fake_process)
    monkeypatch.setattr(video_workflow, "process_meeting_transcription", fake_transcribe)
    monkeypatch.setattr(video_workflow, "mux_tracked_video", fake_mux)

    meeting_id = video_workflow.process_video_and_transcribe(video_path, output_path, log_path)

    assert meeting_id == 14
    assert calls[0][0] == video_path
    assert calls[0][2] == config.SPEAKER_FACE_OVERLAP_THRESHOLD
    assert calls[0][1](tmp_path / "audio.wav", [(0.0, 3.0)]) == [
        ("UNKNOWN", 0.0, 1.0),
        ("Alice", 1.0, 2.0),
        ("UNKNOWN", 2.0, 3.0),
    ]


def test_video_stage_failure_does_not_start_transcription(tmp_path, monkeypatch):
    video_path = tmp_path / "meeting.mp4"
    video_path.write_bytes(b"video")
    output_path = tmp_path / "tracked.mp4"
    log_path = tmp_path / "video.json"

    monkeypatch.setattr(
        video_workflow,
        "process_video_pipeline",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("Video stage failed")),
    )

    transcribe_called = []

    def fake_transcribe(*_args, **_kwargs):
        transcribe_called.append(True)
        return 99

    monkeypatch.setattr(video_workflow, "process_meeting_transcription", fake_transcribe)

    try:
        video_workflow.process_video_and_transcribe(video_path, output_path, log_path)
    except RuntimeError as error:
        assert "Video stage failed" in str(error)
    else:
        raise AssertionError("expected RuntimeError")

    assert transcribe_called == []


def test_workflow_runs_one_transcription_then_burns_cues(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    final = tmp_path / "tracked.mp4"
    log = tmp_path / "events.json"
    calls = {"video": 0, "transcribe": 0, "mux": []}
    stage_messages = []
    seen = {"annotated": None, "subtitle": None}

    def fake_video(_source, annotated, log_path, **_kwargs):
        calls["video"] += 1
        seen["annotated"] = Path(annotated)
        Path(annotated).write_bytes(b"annotated")
        Path(log_path).write_text(json.dumps({"speech": [{
            "speaker": "Alice", "start_time": 0.0, "end_time": 1.0,
        }]}), encoding="utf-8")
        return True

    def fake_transcribe(_source, diarize, minimum_speaker_overlap, subtitle_word_callback, **_kwargs):
        calls["transcribe"] += 1
        assert minimum_speaker_overlap == config.SPEAKER_FACE_OVERLAP_THRESHOLD
        assert diarize(source, [(0.0, 1.0)]) == [("Alice", 0.0, 1.0)]
        subtitle_word_callback(SubtitleWord("Alice", 100, 400, "Hello."))
        return 7

    def fake_write(cues, path):
        assert cues[0].lines == ("Alice: Hello.",)
        seen["subtitle"] = Path(path)
        Path(path).write_text("ASS", encoding="utf-8")
        return Path(path)

    def fake_mux(annotated, original, output, subtitle_path=None, **_kwargs):
        calls["mux"].append((Path(annotated), original, Path(output), subtitle_path))
        Path(output).write_bytes(b"final")
        return Path(output)

    monkeypatch.setattr(video_workflow, "process_video_pipeline", fake_video)
    monkeypatch.setattr(video_workflow, "process_meeting_transcription", fake_transcribe)
    monkeypatch.setattr(video_workflow, "write_ass_subtitles", fake_write)
    monkeypatch.setattr(video_workflow, "mux_tracked_video", fake_mux)

    assert video_workflow.process_video_and_transcribe(
        source,
        final,
        log,
        stage_callback=stage_messages.append,
    ) == 7
    assert calls["video"] == calls["transcribe"] == len(calls["mux"]) == 1
    assert calls["mux"][0][1:3] == (source, final)
    assert calls["mux"][0][3] is not None
    assert final.read_bytes() == b"final"
    assert stage_messages == [
        "Processing faces and speaker events",
        "Transcribing and timing subtitles",
        "Burning subtitles and restoring audio",
    ]
    assert seen["annotated"] is not None
    assert seen["subtitle"] is not None
    assert not seen["annotated"].parent.exists()
    assert not seen["subtitle"].parent.exists()


def test_transcription_failure_preserves_audio_output_and_marks_stage(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    final = tmp_path / "tracked.mp4"
    log = tmp_path / "events.json"

    calls = {"transcribe": 0, "mux": []}
    stage_messages = []
    seen = {"annotated": None}

    def fake_video(_source, annotated, log_path, **_kwargs):
        seen["annotated"] = Path(annotated)
        Path(annotated).write_bytes(b"annotated")
        Path(log_path).write_text(json.dumps({"speech": []}), encoding="utf-8")
        return True

    def fake_transcribe(*_args, **_kwargs):
        calls["transcribe"] += 1
        raise RuntimeError("Whisper failed")

    def fake_mux(_annotated, _source, output, subtitle_path=None, **_kwargs):
        calls["mux"].append(subtitle_path)
        assert subtitle_path is None
        Path(output).write_bytes(b"audio-bearing")
        return Path(output)

    monkeypatch.setattr(video_workflow, "process_video_pipeline", fake_video)
    monkeypatch.setattr(video_workflow, "process_meeting_transcription", fake_transcribe)
    monkeypatch.setattr(video_workflow, "mux_tracked_video", fake_mux)

    with pytest.raises(video_workflow.TranscriptStageError) as excinfo:
        video_workflow.process_video_and_transcribe(
            source,
            final,
            log,
            stage_callback=stage_messages.append,
        )

    assert excinfo.value.stage == "transcription"
    assert excinfo.value.meeting_id is None
    assert excinfo.value.output_updated is True
    assert calls["transcribe"] == 1
    assert calls["mux"] == [None]
    assert final.read_bytes() == b"audio-bearing"
    assert stage_messages == [
        "Processing faces and speaker events",
        "Transcribing and timing subtitles",
    ]
    assert seen["annotated"] is not None
    assert not seen["annotated"].parent.exists()


def test_cue_generation_failure_preserves_audio_output_and_meeting_id(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    final = tmp_path / "tracked.mp4"
    log = tmp_path / "events.json"

    calls = {"transcribe": 0, "mux": []}

    def fake_video(_source, annotated, log_path, **_kwargs):
        Path(annotated).write_bytes(b"annotated")
        Path(log_path).write_text(json.dumps({"speech": []}), encoding="utf-8")
        return True

    def fake_transcribe(*_args, subtitle_word_callback=None, **_kwargs):
        calls["transcribe"] += 1
        if subtitle_word_callback is not None:
            subtitle_word_callback(SubtitleWord("Alice", 10, 30, "hello"))
        return 52

    def fake_build(_words):
        raise RuntimeError("cue generation failed")

    def fake_mux(_annotated, _source, output, subtitle_path=None, **_kwargs):
        calls["mux"].append(subtitle_path)
        assert subtitle_path is None
        Path(output).write_bytes(b"audio-bearing")
        return Path(output)

    monkeypatch.setattr(video_workflow, "process_video_pipeline", fake_video)
    monkeypatch.setattr(video_workflow, "process_meeting_transcription", fake_transcribe)
    monkeypatch.setattr(video_workflow, "build_subtitle_cues", fake_build)
    monkeypatch.setattr(video_workflow, "mux_tracked_video", fake_mux)

    with pytest.raises(video_workflow.TranscriptStageError) as excinfo:
        video_workflow.process_video_and_transcribe(source, final, log)

    assert excinfo.value.stage == "transcription"
    assert excinfo.value.meeting_id == 52
    assert excinfo.value.output_updated is True
    assert isinstance(excinfo.value.cause, RuntimeError)
    assert "cue generation failed" in str(excinfo.value.cause)
    assert calls["transcribe"] == 1
    assert calls["mux"] == [None]
    assert final.read_bytes() == b"audio-bearing"


def test_subtitle_failure_falls_back_to_audio_output_and_marks_stage(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    final = tmp_path / "tracked.mp4"
    log = tmp_path / "events.json"

    calls = {"transcribe": 0, "mux": []}
    seen = {"annotated": None}

    def fake_video(_source, annotated, log_path, **_kwargs):
        seen["annotated"] = Path(annotated)
        Path(annotated).write_bytes(b"annotated")
        Path(log_path).write_text(json.dumps({"speech": [{
            "speaker": "Alice", "start_time": 0.0, "end_time": 1.0,
        }]}), encoding="utf-8")
        return True

    def fake_transcribe(_source, diarize, minimum_speaker_overlap, subtitle_word_callback, **_kwargs):
        calls["transcribe"] += 1
        assert minimum_speaker_overlap == config.SPEAKER_FACE_OVERLAP_THRESHOLD
        assert diarize(source, [(0.0, 1.0)]) == [("Alice", 0.0, 1.0)]
        subtitle_word_callback(SubtitleWord("Alice", 0, 1000, "Hello"))
        return 8

    def fake_write(cues, path):
        assert cues[0].lines == ("Alice: Hello",)
        Path(path).write_text("ASS", encoding="utf-8")
        return Path(path)

    def fake_mux(_annotated, _source, output, subtitle_path=None, **_kwargs):
        calls["mux"].append(subtitle_path)
        if subtitle_path is not None:
            raise VideoMuxError("subtitle burn failed")
        Path(output).write_bytes(b"audio-fallback")
        return Path(output)

    monkeypatch.setattr(video_workflow, "process_video_pipeline", fake_video)
    monkeypatch.setattr(video_workflow, "process_meeting_transcription", fake_transcribe)
    monkeypatch.setattr(video_workflow, "write_ass_subtitles", fake_write)
    monkeypatch.setattr(video_workflow, "mux_tracked_video", fake_mux)

    with pytest.raises(video_workflow.TranscriptStageError) as excinfo:
        video_workflow.process_video_and_transcribe(source, final, log)

    assert excinfo.value.stage == "subtitle"
    assert excinfo.value.meeting_id == 8
    assert excinfo.value.output_updated is True
    assert isinstance(excinfo.value.cause, VideoMuxError)
    assert calls["transcribe"] == 1
    assert len(calls["mux"]) == 2
    assert calls["mux"][0] is not None
    assert calls["mux"][1] is None
    assert final.read_bytes() == b"audio-fallback"
    assert seen["annotated"] is not None
    assert not seen["annotated"].parent.exists()


def test_subtitle_and_audio_mux_failure_copies_annotated_and_reports_audio_mux(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    final = tmp_path / "tracked.mp4"
    final.write_bytes(b"existing-good-output")
    log = tmp_path / "events.json"

    calls = {"transcribe": 0, "mux": []}

    def fake_video(_source, annotated, log_path, **_kwargs):
        Path(annotated).write_bytes(b"annotated-silent")
        Path(log_path).write_text(json.dumps({"speech": [{
            "speaker": "Alice", "start_time": 0.0, "end_time": 1.0,
        }]}), encoding="utf-8")
        return True

    def fake_transcribe(_source, diarize, minimum_speaker_overlap, subtitle_word_callback, **_kwargs):
        calls["transcribe"] += 1
        assert minimum_speaker_overlap == config.SPEAKER_FACE_OVERLAP_THRESHOLD
        assert diarize(source, [(0.0, 1.0)]) == [("Alice", 0.0, 1.0)]
        subtitle_word_callback(SubtitleWord("Alice", 0, 1000, "Hello"))
        return 33

    def fake_write(_cues, path):
        Path(path).write_text("ASS", encoding="utf-8")
        return Path(path)

    def fake_mux(_annotated, _source, _output, subtitle_path=None, **_kwargs):
        calls["mux"].append(subtitle_path)
        if subtitle_path is not None:
            raise VideoMuxError("subtitle burn failed")
        raise RuntimeError("audio mux fallback failed")

    monkeypatch.setattr(video_workflow, "process_video_pipeline", fake_video)
    monkeypatch.setattr(video_workflow, "process_meeting_transcription", fake_transcribe)
    monkeypatch.setattr(video_workflow, "write_ass_subtitles", fake_write)
    monkeypatch.setattr(video_workflow, "mux_tracked_video", fake_mux)

    with pytest.raises(video_workflow.TranscriptStageError) as excinfo:
        video_workflow.process_video_and_transcribe(source, final, log)

    assert calls["transcribe"] == 1
    assert len(calls["mux"]) == 2
    assert calls["mux"][0] is not None
    assert calls["mux"][1] is None
    assert excinfo.value.stage == "audio_mux"
    assert excinfo.value.meeting_id == 33
    assert excinfo.value.output_updated is True
    assert "subtitle burn failed" in str(excinfo.value.cause)
    assert "audio mux fallback failed" in str(excinfo.value.cause)
    assert final.read_bytes() == b"annotated-silent"


def test_subtitle_audio_and_copy_failure_preserves_output_and_reports_all_errors(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    final = tmp_path / "tracked.mp4"
    final.write_bytes(b"existing-good-output")
    log = tmp_path / "events.json"

    calls = {"transcribe": 0, "mux": []}

    def fake_video(_source, annotated, log_path, **_kwargs):
        Path(annotated).write_bytes(b"annotated-silent")
        Path(log_path).write_text(json.dumps({"speech": [{
            "speaker": "Alice", "start_time": 0.0, "end_time": 1.0,
        }]}), encoding="utf-8")
        return True

    def fake_transcribe(_source, diarize, minimum_speaker_overlap, subtitle_word_callback, **_kwargs):
        calls["transcribe"] += 1
        assert minimum_speaker_overlap == config.SPEAKER_FACE_OVERLAP_THRESHOLD
        assert diarize(source, [(0.0, 1.0)]) == [("Alice", 0.0, 1.0)]
        subtitle_word_callback(SubtitleWord("Alice", 0, 1000, "Hello"))
        return 71

    def fake_write(_cues, path):
        Path(path).write_text("ASS", encoding="utf-8")
        return Path(path)

    def fake_mux(_annotated, _source, _output, subtitle_path=None, **_kwargs):
        calls["mux"].append(subtitle_path)
        if subtitle_path is not None:
            raise VideoMuxError("subtitle burn failed")
        raise RuntimeError("audio mux fallback failed")

    monkeypatch.setattr(video_workflow, "process_video_pipeline", fake_video)
    monkeypatch.setattr(video_workflow, "process_meeting_transcription", fake_transcribe)
    monkeypatch.setattr(video_workflow, "write_ass_subtitles", fake_write)
    monkeypatch.setattr(video_workflow, "mux_tracked_video", fake_mux)
    monkeypatch.setattr(video_workflow.shutil, "copy2", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("copy fallback failed")))

    with pytest.raises(video_workflow.TranscriptStageError) as excinfo:
        video_workflow.process_video_and_transcribe(source, final, log)

    assert calls["transcribe"] == 1
    assert len(calls["mux"]) == 2
    assert calls["mux"][0] is not None
    assert calls["mux"][1] is None
    assert excinfo.value.stage == "audio_mux"
    assert excinfo.value.meeting_id == 71
    assert excinfo.value.output_updated is False
    assert "subtitle burn failed" in str(excinfo.value.cause)
    assert "audio mux fallback failed" in str(excinfo.value.cause)
    assert "copy fallback failed" in str(excinfo.value.cause)
    assert final.read_bytes() == b"existing-good-output"


def test_transcription_audio_and_copy_failure_preserves_output_and_reports_all_errors(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    final = tmp_path / "tracked.mp4"
    final.write_bytes(b"existing-good-output")
    log = tmp_path / "events.json"

    calls = {"transcribe": 0, "mux": []}

    def fake_video(_source, annotated, log_path, **_kwargs):
        Path(annotated).write_bytes(b"annotated-silent")
        Path(log_path).write_text(json.dumps({"speech": []}), encoding="utf-8")
        return True

    def fake_transcribe(*_args, **_kwargs):
        calls["transcribe"] += 1
        raise RuntimeError("transcription failed")

    def fake_mux(_annotated, _source, _output, subtitle_path=None, **_kwargs):
        calls["mux"].append(subtitle_path)
        assert subtitle_path is None
        raise RuntimeError("audio mux fallback failed")

    monkeypatch.setattr(video_workflow, "process_video_pipeline", fake_video)
    monkeypatch.setattr(video_workflow, "process_meeting_transcription", fake_transcribe)
    monkeypatch.setattr(video_workflow, "mux_tracked_video", fake_mux)
    monkeypatch.setattr(video_workflow.shutil, "copy2", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("copy fallback failed")))

    with pytest.raises(video_workflow.TranscriptStageError) as excinfo:
        video_workflow.process_video_and_transcribe(source, final, log)

    assert calls["transcribe"] == 1
    assert calls["mux"] == [None]
    assert excinfo.value.stage == "audio_mux"
    assert excinfo.value.meeting_id is None
    assert excinfo.value.output_updated is False
    assert "transcription failed" in str(excinfo.value.cause)
    assert "audio mux fallback failed" in str(excinfo.value.cause)
    assert "copy fallback failed" in str(excinfo.value.cause)
    assert final.read_bytes() == b"existing-good-output"


def test_relative_output_path_uses_temp_outside_output_folder(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    log = tmp_path / "events.json"
    output_path = Path("tracked") / "result.mp4"
    seen = {"annotated": None, "output": None}

    def fake_video(_source, annotated, log_path, **_kwargs):
        seen["annotated"] = Path(annotated)
        Path(annotated).write_bytes(b"annotated")
        Path(log_path).write_text(json.dumps({"speech": []}), encoding="utf-8")
        return True

    def fake_transcribe(*_args, **_kwargs):
        return 99

    def fake_write(_cues, path):
        Path(path).write_text("ASS", encoding="utf-8")
        return Path(path)

    def fake_mux(_annotated, _source, output, subtitle_path=None, **_kwargs):
        seen["output"] = Path(output)
        Path(output).write_bytes(b"ok")
        return Path(output)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(video_workflow, "process_video_pipeline", fake_video)
    monkeypatch.setattr(video_workflow, "process_meeting_transcription", fake_transcribe)
    monkeypatch.setattr(video_workflow, "write_ass_subtitles", fake_write)
    monkeypatch.setattr(video_workflow, "mux_tracked_video", fake_mux)

    assert video_workflow.process_video_and_transcribe(source, output_path, log) == 99

    tracked_dir = (tmp_path / "tracked").resolve()
    expected_output = (tmp_path / output_path).resolve()
    assert seen["annotated"] is not None
    assert seen["annotated"].is_absolute()
    assert seen["output"] == expected_output
    assert seen["annotated"].parent.parent == tmp_path.resolve()
    assert tracked_dir not in seen["annotated"].parents


def test_workflow_success_replaces_uploaded_video_attendance_once(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    final = tmp_path / "tracked.mp4"
    log = tmp_path / "events.json"

    def fake_video(_source, annotated, log_path, **_kwargs):
        Path(annotated).write_bytes(b"annotated")
        Path(log_path).write_text(
            json.dumps(
                {
                    "recognition": [],
                    "speech": [],
                    "attendance": [
                        {
                            "person_id": 7,
                            "person_name": "Alice",
                            "track_id": 1,
                            "confidence": 0.94,
                            "media_offset_ms": 700,
                        },
                        {
                            "person_id": 8,
                            "person_name": "Bob",
                            "track_id": 2,
                            "confidence": 0.81,
                            "media_offset_ms": 1300,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return True

    def fake_transcribe(*_args, subtitle_word_callback=None, **_kwargs):
        if subtitle_word_callback is not None:
            subtitle_word_callback(SubtitleWord("Alice", 50, 150, "hi"))
        return 12

    def fake_write(_cues, path):
        Path(path).write_text("ASS", encoding="utf-8")
        return Path(path)

    def fake_mux(_annotated, _source, output, subtitle_path=None, **_kwargs):
        assert subtitle_path is not None
        Path(output).write_bytes(b"final")
        return Path(output)

    replacement_calls = []

    def fake_replace(source_ref, events):
        replacement_calls.append((source_ref, events))
        return len(events)

    monkeypatch.setattr(video_workflow, "process_video_pipeline", fake_video)
    monkeypatch.setattr(video_workflow, "process_meeting_transcription", fake_transcribe)
    monkeypatch.setattr(video_workflow, "write_ass_subtitles", fake_write)
    monkeypatch.setattr(video_workflow, "mux_tracked_video", fake_mux)
    monkeypatch.setattr(video_workflow.database, "replace_video_attendance", fake_replace)

    assert video_workflow.process_video_and_transcribe(source, final, log) == 12
    assert len(replacement_calls) == 1
    assert replacement_calls[0][0] == source.resolve().as_posix()
    assert replacement_calls[0][1] == [
        {
            "person_id": 7,
            "person_name": "Alice",
            "track_id": 1,
            "confidence": 0.94,
            "media_offset_ms": 700,
        },
        {
            "person_id": 8,
            "person_name": "Bob",
            "track_id": 2,
            "confidence": 0.81,
            "media_offset_ms": 1300,
        },
    ]


def test_workflow_success_with_no_candidates_clears_uploaded_video_attendance(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    final = tmp_path / "tracked.mp4"
    log = tmp_path / "events.json"

    def fake_video(_source, annotated, log_path, **_kwargs):
        Path(annotated).write_bytes(b"annotated")
        Path(log_path).write_text(
            json.dumps({"recognition": [], "speech": []}),
            encoding="utf-8",
        )
        return True

    def fake_transcribe(*_args, subtitle_word_callback=None, **_kwargs):
        if subtitle_word_callback is not None:
            subtitle_word_callback(SubtitleWord("Alice", 10, 20, "ok"))
        return 13

    def fake_write(_cues, path):
        Path(path).write_text("ASS", encoding="utf-8")
        return Path(path)

    def fake_mux(_annotated, _source, output, subtitle_path=None, **_kwargs):
        assert subtitle_path is not None
        Path(output).write_bytes(b"final")
        return Path(output)

    replacement_calls = []

    def fake_replace(source_ref, events):
        replacement_calls.append((source_ref, events))
        return 0

    monkeypatch.setattr(video_workflow, "process_video_pipeline", fake_video)
    monkeypatch.setattr(video_workflow, "process_meeting_transcription", fake_transcribe)
    monkeypatch.setattr(video_workflow, "write_ass_subtitles", fake_write)
    monkeypatch.setattr(video_workflow, "mux_tracked_video", fake_mux)
    monkeypatch.setattr(video_workflow.database, "replace_video_attendance", fake_replace)

    assert video_workflow.process_video_and_transcribe(source, final, log) == 13
    assert replacement_calls == [(source.resolve().as_posix(), [])]


def test_workflow_failure_does_not_replace_uploaded_video_attendance(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    final = tmp_path / "tracked.mp4"
    log = tmp_path / "events.json"

    def fake_video(_source, annotated, log_path, **_kwargs):
        Path(annotated).write_bytes(b"annotated")
        Path(log_path).write_text(
            json.dumps(
                {
                    "recognition": [],
                    "speech": [],
                    "attendance": [
                        {
                            "person_id": 7,
                            "person_name": "Alice",
                            "track_id": 1,
                            "confidence": 0.95,
                            "media_offset_ms": 800,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return True

    monkeypatch.setattr(video_workflow, "process_video_pipeline", fake_video)
    monkeypatch.setattr(
        video_workflow,
        "process_meeting_transcription",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("transcription failed")),
    )

    def fake_mux(_annotated, _source, output, subtitle_path=None, **_kwargs):
        assert subtitle_path is None
        Path(output).write_bytes(b"audio-fallback")
        return Path(output)

    replacement_calls = []
    monkeypatch.setattr(video_workflow, "mux_tracked_video", fake_mux)
    monkeypatch.setattr(video_workflow.database, "replace_video_attendance", lambda *args, **kwargs: replacement_calls.append((args, kwargs)))

    with pytest.raises(video_workflow.TranscriptStageError) as excinfo:
        video_workflow.process_video_and_transcribe(source, final, log)

    assert excinfo.value.stage == "transcription"
    assert replacement_calls == []
