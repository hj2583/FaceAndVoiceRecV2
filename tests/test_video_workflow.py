import json

import config
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

    def fake_process(*_args, **_kwargs):
        log_path.write_text(json.dumps({"speech": expected_events}), encoding="utf-8")
        return True

    calls = []

    def fake_transcribe(path, diarize=None, minimum_speaker_overlap=0.0):
        calls.append((path, diarize, minimum_speaker_overlap))
        return 14

    monkeypatch.setattr(video_workflow, "process_video_pipeline", fake_process)
    monkeypatch.setattr(video_workflow, "process_meeting_transcription", fake_transcribe)

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


def test_transcript_stage_failure_preserves_video_output(tmp_path, monkeypatch):
    video_path = tmp_path / "meeting.mp4"
    video_path.write_bytes(b"video")
    output_path = tmp_path / "tracked.mp4"
    log_path = tmp_path / "video.json"

    def fake_process(*_args, **_kwargs):
        output_path.write_bytes(b"tracked video")
        log_path.write_text(json.dumps({"speech": []}), encoding="utf-8")
        return True

    monkeypatch.setattr(video_workflow, "process_video_pipeline", fake_process)
    monkeypatch.setattr(
        video_workflow,
        "process_meeting_transcription",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("Whisper failed")),
    )

    try:
        video_workflow.process_video_and_transcribe(video_path, output_path, log_path)
    except video_workflow.TranscriptStageError as error:
        assert error.video_output_path == output_path
        assert isinstance(error.cause, RuntimeError)
        assert "Whisper failed" in str(error)
    else:
        raise AssertionError("expected TranscriptStageError")

    assert output_path.read_bytes() == b"tracked video"
