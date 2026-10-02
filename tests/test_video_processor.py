import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import video_processor


class _FakeCapture:
    def __init__(self, frame_count=5, width=160, height=120, fps=30.0):
        self.frame_count = frame_count
        self.width = width
        self.height = height
        self.fps = fps
        self._read_calls = 0

    def isOpened(self):
        return True

    def get(self, prop):
        return {
            cv2.CAP_PROP_FRAME_WIDTH: self.width,
            cv2.CAP_PROP_FRAME_HEIGHT: self.height,
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FRAME_COUNT: self.frame_count,
        }[prop]

    def read(self):
        if self._read_calls >= self.frame_count:
            return False, None
        self._read_calls += 1
        frame = np.full((self.height, self.width, 3), self._read_calls, dtype=np.uint8)
        return True, frame

    def release(self):
        return None


class _FakeWriter:
    def __init__(self):
        self.frames = []
        self._opened = True

    def isOpened(self):
        return self._opened

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        return None


class _FakeLandmarker:
    def close(self):
        return None


def _patch_common(monkeypatch, capture, writer):
    def _make_writer(path, *_a, **_k):
        # Real code renames this temp file on disk after writing; touch it so
        # os.replace(temp_output, output_path) succeeds with the fake writer.
        Path(path).touch()
        return writer

    monkeypatch.setattr(video_processor.cv2, "VideoCapture", lambda *_a, **_k: capture)
    monkeypatch.setattr(video_processor.cv2, "VideoWriter", _make_writer)
    monkeypatch.setattr(video_processor.cv2, "VideoWriter_fourcc", lambda *_a: 0)
    monkeypatch.setattr(video_processor, "extract_audio_to_wav", lambda _p: None)
    monkeypatch.setattr(video_processor, "create_face_landmarker", lambda: _FakeLandmarker())
    monkeypatch.setattr(video_processor, "detect_faces_tiled", lambda *_a, **_k: [])
    monkeypatch.setattr(video_processor, "FaceIndex", lambda: object())
    monkeypatch.setattr(
        video_processor,
        "CentroidTracker",
        lambda: SimpleNamespace(update=lambda *_a, **_k: [], tracks={}),
    )
    monkeypatch.setattr(video_processor, "convert_h264", lambda *_a, **_k: False)


def test_process_video_pipeline_writes_one_frame_per_input_frame(monkeypatch, tmp_path):
    capture = _FakeCapture(frame_count=7)
    writer = _FakeWriter()
    _patch_common(monkeypatch, capture, writer)

    output_path = tmp_path / "out.mp4"
    log_path = tmp_path / "log.json"

    video_processor.process_video_pipeline(tmp_path / "in.mp4", output_path, log_path)

    assert len(writer.frames) == 7
    assert log_path.exists()
    data = json.loads(log_path.read_text(encoding="utf-8"))
    assert "recognition" in data


def test_process_video_pipeline_propagates_detection_errors(monkeypatch, tmp_path):
    capture = _FakeCapture(frame_count=3)
    writer = _FakeWriter()
    _patch_common(monkeypatch, capture, writer)

    def _boom(*_args, **_kwargs):
        raise ValueError("detector exploded")

    monkeypatch.setattr(video_processor, "detect_faces_tiled", _boom)

    with pytest.raises(ValueError, match="detector exploded"):
        video_processor.process_video_pipeline(
            tmp_path / "in.mp4", tmp_path / "out.mp4", tmp_path / "log.json"
        )


def test_process_video_pipeline_logs_unknown_speaker_events(tmp_path, monkeypatch):
    """An unknown-but-speaking track must now produce an audio_logs event,
    where the old `if speaker.person_id is not None` guard used to drop it."""
    capture = _FakeCapture(frame_count=3)
    writer = _FakeWriter()
    _patch_common(monkeypatch, capture, writer)
    monkeypatch.setattr(video_processor, "extract_audio_to_wav", lambda _p: tmp_path / "audio.wav")
    monkeypatch.setattr(video_processor, "detect_speech_segments", lambda *_a, **_k: [(0.0, 10.0)])
    monkeypatch.setattr(video_processor, "extract_embedding", lambda *_a, **_k: None)
    monkeypatch.setattr(video_processor, "lip_open_ratio", lambda _landmarks: 0.05)

    class _Landmarker:
        def detect_for_video(self, *_a, **_k):
            return SimpleNamespace(face_landmarks=[object()])

        def close(self):
            return None

    monkeypatch.setattr(video_processor, "create_face_landmarker", lambda: _Landmarker())

    from tracking import Track

    track = Track(track_id=1, center_x=30, center_y=30, area=3600)
    monkeypatch.setattr(
        video_processor, "CentroidTracker",
        lambda: SimpleNamespace(update=lambda dets, _n: [(dets[0], track)] if dets else [], tracks={1: track}),
    )
    # 70x70 box: large enough for landmarks (LANDMARK_MIN_FACE_SIZE=60).
    monkeypatch.setattr(video_processor, "detect_faces_tiled", lambda *_a, **_k: [(0, 0, 70, 70, 0.9)])

    logged = []
    monkeypatch.setattr(video_processor, "log_audio", lambda *args, **kwargs: logged.append((args, kwargs)))

    video_processor.process_video_pipeline(
        tmp_path / "in.mp4", tmp_path / "out.mp4", tmp_path / "log.json",
    )

    assert len(logged) == 1, "the open UNKNOWN run must be flushed once when the video ends"
    args, kwargs = logged[0]
    # log_audio(start, end, person_id, person_name, confidence, source, track_id=...)
    assert args[2] is None
    assert args[3] == "UNKNOWN"
    assert args[5] == "video"
    assert kwargs["track_id"] == 1
    speech = json.loads((tmp_path / "log.json").read_text(encoding="utf-8"))["speech"]
    assert len(speech) == 1
    assert speech[0]["person_id"] is None
    assert speech[0]["track_id"] == 1
    assert speech[0]["speaker"] == "UNKNOWN"
    assert [entry["person_name"] for entry in speech] == ["UNKNOWN"]


def test_process_video_pipeline_logs_recognized_speaker_with_evicted_track(tmp_path, monkeypatch):
    """A recognized speaker's event must carry person_id even if the tracker
    has already evicted the track by the time the run closes."""
    capture = _FakeCapture(frame_count=3)
    writer = _FakeWriter()
    _patch_common(monkeypatch, capture, writer)
    monkeypatch.setattr(video_processor, "extract_audio_to_wav", lambda _p: tmp_path / "audio.wav")
    monkeypatch.setattr(video_processor, "detect_speech_segments", lambda *_a, **_k: [(0.0, 10.0)])
    monkeypatch.setattr(video_processor, "extract_embedding", lambda *_a, **_k: None)
    monkeypatch.setattr(video_processor, "lip_open_ratio", lambda _landmarks: 0.05)

    class _Landmarker:
        def detect_for_video(self, *_a, **_k):
            return SimpleNamespace(face_landmarks=[object()])

        def close(self):
            return None

    monkeypatch.setattr(video_processor, "create_face_landmarker", lambda: _Landmarker())

    from tracking import Track

    track = Track(
        track_id=2, center_x=30, center_y=30, area=3600,
        person_id=7, person_name="Alice", confidence=0.95,
    )
    # Simulate the track already being evicted by the time the run closes.
    monkeypatch.setattr(
        video_processor, "CentroidTracker",
        lambda: SimpleNamespace(update=lambda dets, _n: [(dets[0], track)] if dets else [], tracks={}),
    )
    # 70x70 box: large enough for landmarks (LANDMARK_MIN_FACE_SIZE=60).
    monkeypatch.setattr(video_processor, "detect_faces_tiled", lambda *_a, **_k: [(0, 0, 70, 70, 0.9)])

    logged = []
    monkeypatch.setattr(video_processor, "log_audio", lambda *args, **kwargs: logged.append((args, kwargs)))

    video_processor.process_video_pipeline(
        tmp_path / "in.mp4", tmp_path / "out.mp4", tmp_path / "log.json",
    )

    assert len(logged) == 1, "the open recognized run must be flushed once when the video ends"
    args, kwargs = logged[0]
    # log_audio(start, end, person_id, person_name, confidence, source, track_id=...)
    assert args[2] == 7
    assert args[3] == "Alice"
    assert kwargs["track_id"] == 2
