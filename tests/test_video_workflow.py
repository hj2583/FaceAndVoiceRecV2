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

    def fake_video(_source, annotated, log_path, **_kwargs):
        calls["video"] += 1
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

    assert video_workflow.process_video_and_transcribe(source, final, log) == 7
    assert calls["video"] == calls["transcribe"] == len(calls["mux"]) == 1
    assert calls["mux"][0][1:3] == (source, final)
    assert calls["mux"][0][3] is not None
    assert final.read_bytes() == b"final"


def test_transcription_failure_preserves_audio_output_and_marks_stage(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    final = tmp_path / "tracked.mp4"
    log = tmp_path / "events.json"

    calls = {"transcribe": 0, "mux": []}

    def fake_video(_source, annotated, log_path, **_kwargs):
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
        video_workflow.process_video_and_transcribe(source, final, log)

    assert excinfo.value.stage == "transcription"
    assert calls["transcribe"] == 1
    assert calls["mux"] == [None]
    assert final.read_bytes() == b"audio-bearing"


def test_subtitle_failure_falls_back_to_audio_output_and_marks_stage(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    final = tmp_path / "tracked.mp4"
    log = tmp_path / "events.json"

    calls = {"transcribe": 0, "mux": []}

    def fake_video(_source, annotated, log_path, **_kwargs):
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
    assert isinstance(excinfo.value.cause, VideoMuxError)
    assert calls["transcribe"] == 1
    assert len(calls["mux"]) == 2
    assert calls["mux"][0] is not None
    assert calls["mux"][1] is None
    assert final.read_bytes() == b"audio-fallback"
