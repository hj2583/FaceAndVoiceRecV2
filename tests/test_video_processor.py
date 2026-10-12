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


def test_uniface_landmarker_close_releases_model_sessions():
    detector = object()
    mesher = object()
    landmarker = video_processor._UniFaceLandmarker(detector, mesher)

    landmarker.close()

    assert landmarker._detector is None
    assert landmarker._mesher is None


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
    lip_ratios = iter((0.0, 0.06, 0.0))
    monkeypatch.setattr(
        video_processor,
        "lip_open_ratio",
        lambda _landmarks: next(lip_ratios, 0.0),
    )

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


def test_process_video_pipeline_logs_recognized_speaker_with_evicted_track(tmp_path, monkeypatch):
    """A recognized speaker's event must carry person_id even if the tracker
    has already evicted the track by the time the run closes."""
    capture = _FakeCapture(frame_count=3)
    writer = _FakeWriter()
    _patch_common(monkeypatch, capture, writer)
    monkeypatch.setattr(video_processor, "extract_audio_to_wav", lambda _p: tmp_path / "audio.wav")
    monkeypatch.setattr(video_processor, "detect_speech_segments", lambda *_a, **_k: [(0.0, 10.0)])
    monkeypatch.setattr(video_processor, "extract_embedding", lambda *_a, **_k: None)
    lip_ratios = iter((0.0, 0.06, 0.0))
    monkeypatch.setattr(
        video_processor,
        "lip_open_ratio",
        lambda _landmarks: next(lip_ratios, 0.0),
    )

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

    assert len(logged) == 2, "the uncertain startup frame and recognized run should both be recorded"
    args, kwargs = logged[-1]
    # log_audio(start, end, person_id, person_name, confidence, source, track_id=...)
    assert args[2] == 7
    assert args[3] == "Alice"
    assert kwargs["track_id"] == 2


def test_process_video_pipeline_attendance_candidates_first_seen_per_track_person(tmp_path, monkeypatch):
    capture = _FakeCapture(frame_count=3, fps=2.0)
    writer = _FakeWriter()
    _patch_common(monkeypatch, capture, writer)
    monkeypatch.setattr(video_processor, "extract_audio_to_wav", lambda _p: None)
    monkeypatch.setattr(video_processor, "extract_embedding", lambda *_a, **_k: None)

    class _Landmarker:
        def detect_for_video(self, *_a, **_k):
            return SimpleNamespace(face_landmarks=[])

        def close(self):
            return None

    monkeypatch.setattr(video_processor, "create_face_landmarker", lambda: _Landmarker())

    from tracking import Track

    known_primary = Track(track_id=1, center_x=30, center_y=30, area=3600)
    known_primary.person_id = 101
    known_primary.person_name = "Alice"
    known_primary.confidence = 0.91
    known_primary.embedding = np.array([0.1, 0.2], dtype=np.float32)

    known_high_conf = Track(track_id=2, center_x=50, center_y=50, area=3600)
    known_high_conf.person_id = 202
    known_high_conf.person_name = "Bob"
    known_high_conf.confidence = 1.4
    known_high_conf.embedding = np.array([0.3, 0.4], dtype=np.float32)

    unknown_track = Track(track_id=3, center_x=70, center_y=70, area=3600)
    unknown_track.person_id = None
    unknown_track.person_name = "Unknown"
    unknown_track.confidence = 0.99
    unknown_track.embedding = np.array([0.5, 0.6], dtype=np.float32)

    known_below_threshold = Track(track_id=4, center_x=90, center_y=90, area=3600)
    known_below_threshold.person_id = 303
    known_below_threshold.person_name = "Carol"
    known_below_threshold.confidence = max(0.0, video_processor.RECOGNITION_THRESHOLD - 0.01)
    known_below_threshold.embedding = np.array([0.7, 0.8], dtype=np.float32)

    class _Tracker:
        def __init__(self):
            self.call_count = 0
            self.tracks = {
                1: known_primary,
                2: known_high_conf,
                3: unknown_track,
                4: known_below_threshold,
            }

        def update(self, detections, _frame_no):
            self.call_count += 1
            if self.call_count == 1:
                return [
                    (detections[0], known_primary),
                    (detections[1], known_below_threshold),
                ]
            if self.call_count == 2:
                return [
                    (detections[0], known_primary),
                    (detections[2], known_high_conf),
                    (detections[3], unknown_track),
                ]
            return []

    monkeypatch.setattr(video_processor, "CentroidTracker", _Tracker)
    monkeypatch.setattr(
        video_processor,
        "detect_faces_tiled",
        lambda *_a, **_k: [
            (0, 0, 70, 70, 0.95),
            (75, 0, 70, 70, 0.95),
            (0, 75, 70, 70, 0.95),
            (75, 75, 70, 70, 0.95),
        ],
    )

    log_path = tmp_path / "video_log.json"
    video_processor.process_video_pipeline(tmp_path / "in.mp4", tmp_path / "out.mp4", log_path)

    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert isinstance(payload.get("recognition"), list)
    assert isinstance(payload.get("speech"), list)

    attendance = payload["attendance"]
    assert attendance == [
        {
            "person_id": 101,
            "person_name": "Alice",
            "track_id": 1,
            "confidence": 0.91,
            "media_offset_ms": 500,
        },
        {
            "person_id": 202,
            "person_name": "Bob",
            "track_id": 2,
            "confidence": 1.0,
            "media_offset_ms": 1000,
        },
    ]
