import threading
import time
import uuid
from datetime import datetime, timezone
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from realtime import start_realtime_vad


def test_start_realtime_vad_returns_available_started_instance():
    vad = Mock()
    factory = Mock(return_value=vad)

    result, available = start_realtime_vad(Mock(), vad_factory=factory)

    assert result is vad
    assert available is True
    vad.start.assert_called_once_with()


def test_start_realtime_vad_contains_startup_failure():
    vad = Mock()
    vad.start.side_effect = RuntimeError("no microphone")

    result, available = start_realtime_vad(Mock(), vad_factory=Mock(return_value=vad))

    assert result is None
    assert available is False


def test_run_detects_vad_becoming_unavailable(monkeypatch):
    import realtime

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    class _Cap:
        def __init__(self):
            self.calls = 0

        def isOpened(self):
            return True

        def set(self, *_args, **_kwargs):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 160.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 120.0
            return 0.0

        def read(self):
            self.calls += 1
            if self.calls <= 2:
                return True, frame.copy()
            return False, None

        def release(self):
            return None

    class _Mesh:
        def detect_for_video(self, *_args, **_kwargs):
            return SimpleNamespace(face_landmarks=[])

        def close(self):
            return None

    class _Tracker:
        def __init__(self):
            self.tracks = {}

        def update(self, detections, frame_no):
            return []

    class _Vad:
        def __init__(self):
            self.available = True

        def stop(self):
            return None

    vad = _Vad()
    mic_lines = []
    read_counter = {"count": 0}

    original_read = _Cap.read

    def _read_and_drop(self):
        ok, image = original_read(self)
        if ok:
            read_counter["count"] += 1
            if read_counter["count"] == 1:
                vad.available = False
        return ok, image

    def _put_text(_frame, text, origin, *_args, **_kwargs):
        if str(text).startswith("MIC:"):
            mic_lines.append(str(text))
        return None

    def _start_realtime_vad(callback):
        callback(True)
        return vad, True

    monkeypatch.setattr(_Cap, "read", _read_and_drop)
    monkeypatch.setattr(realtime.cv2, "VideoCapture", lambda *_args, **_kwargs: _Cap())
    monkeypatch.setattr(realtime, "FaceIndex", lambda: object())
    monkeypatch.setattr(realtime, "CentroidTracker", lambda: _Tracker())
    monkeypatch.setattr(realtime, "start_realtime_vad", _start_realtime_vad)
    monkeypatch.setattr(realtime, "create_face_landmarker", lambda: _Mesh())
    monkeypatch.setattr(realtime, "detect_faces_realtime", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(realtime, "log_recognition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "log_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "cvtColor", lambda image, _code: image)
    monkeypatch.setattr(realtime.cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "waitKey", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(realtime.cv2, "getWindowProperty", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(realtime.cv2, "rectangle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "putText", _put_text)
    monkeypatch.setattr(realtime.cv2, "destroyAllWindows", lambda: None)

    dummy_mp = SimpleNamespace(
        ImageFormat=SimpleNamespace(SRGB=1),
        Image=lambda image_format, data: data,
    )
    import sys
    fake_video_processor = ModuleType("video_processor")
    fake_video_processor.lip_open_ratio = lambda _landmarks: 0.0
    monkeypatch.setitem(sys.modules, "video_processor", fake_video_processor)
    monkeypatch.setitem(sys.modules, "mediapipe", dummy_mp)

    realtime.run(camera=0, width=160, height=120)

    assert "MIC: UNAVAILABLE" in mic_lines


def test_run_continues_cleanup_when_vad_stop_raises(monkeypatch):
    import realtime

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    class _Cap:
        def __init__(self):
            self.read_calls = 0
            self.released = False

        def isOpened(self):
            return True

        def set(self, *_args, **_kwargs):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 160.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 120.0
            return 0.0

        def read(self):
            self.read_calls += 1
            if self.read_calls == 1:
                return True, frame.copy()
            return False, None

        def release(self):
            self.released = True

    cap = _Cap()

    class _Mesh:
        def __init__(self):
            self.closed = False

        def detect_for_video(self, *_args, **_kwargs):
            return SimpleNamespace(face_landmarks=[SimpleNamespace()])

        def close(self):
            self.closed = True

    mesh = _Mesh()

    class _Track:
        def __init__(self):
            self.track_id = 1
            self.person_id = None
            self.person_name = "Unknown"
            self.confidence = 0.0
            self.embedding = None
            self.center_x = 40
            self.center_y = 50
            self.lip_open = 0.0

    class _Tracker:
        def __init__(self):
            self.tracks = {}

        def update(self, detections, frame_no):
            if not detections:
                return []
            return [(detections[0], _Track())]

    class _Vad:
        def __init__(self):
            self.available = True

        def stop(self):
            raise RuntimeError("stop failed")

    destroy_called = {"value": False}

    monkeypatch.setattr(realtime.cv2, "VideoCapture", lambda *_args, **_kwargs: cap)
    monkeypatch.setattr(realtime, "FaceIndex", lambda: object())
    monkeypatch.setattr(realtime, "CentroidTracker", lambda: _Tracker())
    def _start_realtime_vad(callback):
        callback(True)
        return _Vad(), True

    monkeypatch.setattr(realtime, "start_realtime_vad", _start_realtime_vad)
    monkeypatch.setattr(realtime, "create_face_landmarker", lambda: mesh)
    monkeypatch.setattr(realtime, "detect_faces_realtime", lambda *_args, **_kwargs: [(10, 10, 40, 40, 0.9)])
    monkeypatch.setattr(realtime, "extract_embedding", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "face_quality", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(realtime, "log_recognition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "log_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "cvtColor", lambda image, _code: image)
    monkeypatch.setattr(realtime.cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "waitKey", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(realtime.cv2, "getWindowProperty", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(realtime.cv2, "rectangle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "putText", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "imwrite", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(realtime, "register_unknown", lambda **_kwargs: 1)
    monkeypatch.setattr(realtime, "find_matching_unknown", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "update_unknown_image", lambda **_kwargs: None)
    monkeypatch.setattr(realtime, "add_unknown_sample", lambda **_kwargs: None)
    monkeypatch.setattr(realtime.np, "save", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "destroyAllWindows", lambda: destroy_called.__setitem__("value", True))

    dummy_mp = SimpleNamespace(
        ImageFormat=SimpleNamespace(SRGB=1),
        Image=lambda image_format, data: data,
    )
    import sys
    fake_video_processor = ModuleType("video_processor")
    fake_video_processor.lip_open_ratio = lambda _landmarks: 0.0
    monkeypatch.setitem(sys.modules, "video_processor", fake_video_processor)
    monkeypatch.setitem(sys.modules, "mediapipe", dummy_mp)

    realtime.run(camera=0, width=160, height=120)

    assert cap.released is True
    assert mesh.closed is True
    assert destroy_called["value"] is True


def test_speaking_banner_drawn_below_fps_line(monkeypatch):
    import realtime

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    class _Cap:
        def __init__(self):
            self.calls = 0

        def isOpened(self):
            return True

        def set(self, *_args, **_kwargs):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 160.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 120.0
            return 0.0

        def read(self):
            self.calls += 1
            if self.calls <= 2:
                return True, frame.copy()
            return False, None

        def release(self):
            return None

    class _Track:
        def __init__(self):
            self.track_id = 1
            self.person_id = 7
            self.person_name = "Alice"
            self.confidence = 0.99
            self.embedding = None
            self.center_x = 60
            self.center_y = 60
            self.lip_open = 1.0

    class _Tracker:
        def __init__(self):
            self.tracks = {1: _Track()}

        def update(self, detections, frame_no):
            return [(detections[0], self.tracks[1])] if detections else []

    class _Mesh:
        def detect_for_video(self, *_args, **_kwargs):
            return SimpleNamespace(face_landmarks=[SimpleNamespace()])

        def close(self):
            return None

    class _Vad:
        def __init__(self):
            self.available = True

        def stop(self):
            return None

    put_text_calls = []

    def _put_text(_frame, text, origin, *_args, **_kwargs):
        put_text_calls.append((text, origin))
        return None

    monkeypatch.setattr(realtime.cv2, "VideoCapture", lambda *_args, **_kwargs: _Cap())
    monkeypatch.setattr(realtime, "FaceIndex", lambda: SimpleNamespace(search=lambda *_args, **_kwargs: None))
    monkeypatch.setattr(realtime, "CentroidTracker", lambda: _Tracker())
    def _start_realtime_vad(callback):
        callback(True)
        return _Vad(), True

    monkeypatch.setattr(realtime, "start_realtime_vad", _start_realtime_vad)
    monkeypatch.setattr(realtime, "create_face_landmarker", lambda: _Mesh())
    monkeypatch.setattr(realtime, "detect_faces_realtime", lambda *_args, **_kwargs: [(10, 10, 60, 60, 0.9)])
    monkeypatch.setattr(realtime, "extract_embedding", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "face_quality", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(realtime, "log_recognition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "log_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "cvtColor", lambda image, _code: image)
    monkeypatch.setattr(realtime.cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "waitKey", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(realtime.cv2, "getWindowProperty", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(realtime.cv2, "rectangle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "putText", _put_text)
    monkeypatch.setattr(realtime.cv2, "destroyAllWindows", lambda: None)

    dummy_mp = SimpleNamespace(
        ImageFormat=SimpleNamespace(SRGB=1),
        Image=lambda image_format, data: data,
    )
    import sys
    fake_video_processor = ModuleType("video_processor")
    lip_ratios = iter((0.0, 0.06, 0.0))
    fake_video_processor.lip_open_ratio = lambda _landmarks: next(lip_ratios, 0.0)
    monkeypatch.setitem(sys.modules, "video_processor", fake_video_processor)
    monkeypatch.setitem(sys.modules, "mediapipe", dummy_mp)

    realtime.run(camera=0, width=160, height=120)

    fps_y = next(origin[1] for text, origin in put_text_calls if str(text).startswith("FPS:"))
    speaking_y = next(origin[1] for text, origin in put_text_calls if str(text).startswith("SPEAKING:"))
    assert speaking_y > fps_y


def test_run_detection_scheduler_calls_detector_twice_for_four_frames_interval_three(monkeypatch):
    import time

    import realtime

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    class _Cap:
        def __init__(self):
            self.calls = 0

        def isOpened(self):
            return True

        def set(self, *_args, **_kwargs):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 160.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 120.0
            return 0.0

        def read(self):
            # Slow the reader down relative to the (near-instant, mocked)
            # compute stage so the drop-oldest queue never actually drops a
            # frame here, keeping the frame_no sequence this test depends on
            # deterministic.
            time.sleep(0.01)
            self.calls += 1
            if self.calls <= 4:
                return True, frame.copy()
            return False, None

        def release(self):
            return None

    class _Mesh:
        def detect_for_video(self, *_args, **_kwargs):
            return SimpleNamespace(face_landmarks=[])

        def close(self):
            return None

    class _Tracker:
        def __init__(self):
            self.tracks = {}

        def update(self, detections, frame_no):
            return []

    class _Vad:
        def __init__(self):
            self.available = True

        def stop(self):
            return None

    detector_calls = {"count": 0}

    def _detector(*_args, **_kwargs):
        detector_calls["count"] += 1
        return []

    monkeypatch.setattr(realtime, "REALTIME_DETECTION_INTERVAL", 3)
    monkeypatch.setattr(realtime.cv2, "VideoCapture", lambda *_args, **_kwargs: _Cap())
    monkeypatch.setattr(realtime, "FaceIndex", lambda: object())
    monkeypatch.setattr(realtime, "CentroidTracker", lambda: _Tracker())

    def _start_realtime_vad(callback):
        callback(True)
        return _Vad(), True

    monkeypatch.setattr(realtime, "start_realtime_vad", _start_realtime_vad)
    monkeypatch.setattr(realtime, "create_face_landmarker", lambda: _Mesh())
    monkeypatch.setattr(realtime, "detect_faces_realtime", _detector)
    monkeypatch.setattr(realtime, "log_recognition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "log_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "cvtColor", lambda image, _code: image)
    monkeypatch.setattr(realtime.cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "waitKey", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(realtime.cv2, "getWindowProperty", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(realtime.cv2, "rectangle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "putText", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "destroyAllWindows", lambda: None)

    dummy_mp = SimpleNamespace(
        ImageFormat=SimpleNamespace(SRGB=1),
        Image=lambda image_format, data: data,
    )
    import sys
    fake_video_processor = ModuleType("video_processor")
    fake_video_processor.lip_open_ratio = lambda _landmarks: 0.0
    monkeypatch.setitem(sys.modules, "video_processor", fake_video_processor)
    monkeypatch.setitem(sys.modules, "mediapipe", dummy_mp)

    realtime.run(camera=0, width=160, height=120)

    assert detector_calls["count"] == 2


def test_run_uses_increasing_landmark_timestamps_for_multiple_faces(monkeypatch):
    import realtime

    frame = np.zeros((160, 240, 3), dtype=np.uint8)

    class _Cap:
        def __init__(self):
            self.calls = 0

        def isOpened(self):
            return True

        def set(self, *_args, **_kwargs):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 240.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 160.0
            return 0.0

        def read(self):
            self.calls += 1
            if self.calls == 1:
                return True, frame.copy()
            return False, None

        def release(self):
            return None

    class _Mesh:
        def __init__(self):
            self.timestamps = []

        def detect_for_video(self, _image, timestamp_ms):
            self.timestamps.append(timestamp_ms)
            return SimpleNamespace(face_landmarks=[])

        def close(self):
            return None

    class _Tracker:
        def update(self, _detections, _frame_no):
            return []

    class _Vad:
        available = True

        def stop(self):
            return None

    mesh = _Mesh()
    monkeypatch.setattr(realtime.cv2, "VideoCapture", lambda *_args, **_kwargs: _Cap())
    monkeypatch.setattr(realtime, "FaceIndex", lambda: object())
    monkeypatch.setattr(realtime, "CentroidTracker", lambda: _Tracker())
    monkeypatch.setattr(realtime, "start_realtime_vad", lambda callback: (_Vad(), True))
    monkeypatch.setattr(realtime, "create_face_landmarker", lambda: mesh)
    monkeypatch.setattr(
        realtime,
        "detect_faces_realtime",
        lambda *_args, **_kwargs: [
            (10, 10, 70, 70, 0.9),
            (130, 10, 70, 70, 0.9),
        ],
    )
    monkeypatch.setattr(realtime.cv2, "cvtColor", lambda image, _code: image)
    monkeypatch.setattr(realtime.cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "waitKey", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(realtime.cv2, "getWindowProperty", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(realtime.cv2, "rectangle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "putText", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(realtime, "ensure_opencv_face_detector_available", lambda: None)

    dummy_mp = SimpleNamespace(
        ImageFormat=SimpleNamespace(SRGB=1),
        Image=lambda image_format, data: data,
    )
    import sys
    fake_video_processor = ModuleType("video_processor")
    fake_video_processor.lip_open_ratio = lambda _landmarks: 0.0
    monkeypatch.setitem(sys.modules, "video_processor", fake_video_processor)
    monkeypatch.setitem(sys.modules, "mediapipe", dummy_mp)

    realtime.run(camera=0, width=240, height=160)

    assert len(mesh.timestamps) == 2
    assert mesh.timestamps[0] < mesh.timestamps[1]


def test_run_processes_every_captured_frame_via_threaded_pipeline(monkeypatch):
    import realtime

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    class _Cap:
        def __init__(self):
            self.calls = 0

        def isOpened(self):
            return True

        def set(self, *_args, **_kwargs):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 160.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 120.0
            return 0.0

        def read(self):
            self.calls += 1
            if self.calls <= 3:
                return True, frame.copy()
            return False, None

        def release(self):
            return None

    class _Mesh:
        def detect_for_video(self, *_args, **_kwargs):
            return SimpleNamespace(face_landmarks=[])

        def close(self):
            return None

    class _Tracker:
        def __init__(self):
            self.tracks = {}

        def update(self, detections, frame_no):
            return []

    cap = _Cap()

    monkeypatch.setattr(realtime.cv2, "VideoCapture", lambda *_args, **_kwargs: cap)
    monkeypatch.setattr(realtime, "FaceIndex", lambda: object())
    monkeypatch.setattr(realtime, "CentroidTracker", lambda: _Tracker())
    monkeypatch.setattr(realtime, "start_realtime_vad", lambda callback: (None, False))
    monkeypatch.setattr(realtime, "create_face_landmarker", lambda: _Mesh())
    monkeypatch.setattr(realtime, "detect_faces_realtime", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(realtime, "log_recognition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "log_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "cvtColor", lambda image, _code: image)
    monkeypatch.setattr(realtime.cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "waitKey", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(realtime.cv2, "getWindowProperty", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(realtime.cv2, "rectangle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "putText", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "destroyAllWindows", lambda: None)

    import sys
    fake_video_processor = ModuleType("video_processor")
    fake_video_processor.lip_open_ratio = lambda _landmarks: 0.0
    monkeypatch.setitem(sys.modules, "video_processor", fake_video_processor)
    monkeypatch.setitem(
        sys.modules,
        "mediapipe",
        SimpleNamespace(ImageFormat=SimpleNamespace(SRGB=1), Image=lambda image_format, data: data),
    )

    realtime.run(camera=0, width=160, height=120)

    assert cap.calls == 4


def test_run_quits_cleanly_on_q_keypress(monkeypatch):
    """The q/Esc quit path must return normally (PipelineStop), release the
    camera, and leave no frame-pipeline-* threads running afterward."""
    import realtime

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    class _Cap:
        def __init__(self):
            self.calls = 0
            self.released = 0

        def isOpened(self):
            return True

        def set(self, *_args, **_kwargs):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 160.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 120.0
            return 0.0

        def read(self):
            self.calls += 1
            return True, frame.copy()

        def release(self):
            self.released += 1

    class _Mesh:
        def detect_for_video(self, *_args, **_kwargs):
            return SimpleNamespace(face_landmarks=[])

        def close(self):
            return None

    class _Tracker:
        def __init__(self):
            self.tracks = {}

        def update(self, detections, frame_no):
            return []

    cap = _Cap()
    displayed = {"count": 0}

    def _fake_wait_key(*_args, **_kwargs):
        displayed["count"] += 1
        return ord("q") if displayed["count"] >= 3 else 0

    monkeypatch.setattr(realtime.cv2, "VideoCapture", lambda *_args, **_kwargs: cap)
    monkeypatch.setattr(realtime, "FaceIndex", lambda: object())
    monkeypatch.setattr(realtime, "CentroidTracker", lambda: _Tracker())
    monkeypatch.setattr(realtime, "start_realtime_vad", lambda callback: (None, False))
    monkeypatch.setattr(realtime, "create_face_landmarker", lambda: _Mesh())
    monkeypatch.setattr(realtime, "detect_faces_realtime", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(realtime, "log_recognition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "log_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "cvtColor", lambda image, _code: image)
    monkeypatch.setattr(realtime.cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "waitKey", _fake_wait_key)
    monkeypatch.setattr(realtime.cv2, "getWindowProperty", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(realtime.cv2, "rectangle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "putText", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "destroyAllWindows", lambda: None)

    import sys
    fake_video_processor = ModuleType("video_processor")
    fake_video_processor.lip_open_ratio = lambda _landmarks: 0.0
    monkeypatch.setitem(sys.modules, "video_processor", fake_video_processor)
    monkeypatch.setitem(
        sys.modules,
        "mediapipe",
        SimpleNamespace(ImageFormat=SimpleNamespace(SRGB=1), Image=lambda image_format, data: data),
    )

    realtime.run(camera=0, width=160, height=120)

    assert displayed["count"] >= 3
    assert cap.released >= 1

    time.sleep(0.1)
    active_names = {t.name for t in threading.enumerate()}
    assert not any(name.startswith("frame-pipeline-") for name in active_names)


def _voice_fallback_track():
    from tracking import Track

    return Track(track_id=1, center_x=60, center_y=60, area=100)


class _PcmVad:
    def get_recent_pcm(self, seconds):
        return b"\x00\x01" * 8000


def test_voice_fallback_skips_embedding_when_no_voiceprints_enrolled(monkeypatch):
    import realtime

    extract = Mock()
    monkeypatch.setattr(realtime.voice_core, "has_enrolled_voiceprints", lambda: False)
    monkeypatch.setattr(realtime.voice_core, "extract_voice_embedding", extract)

    assert realtime.voice_fallback_match(_voice_fallback_track(), 1, _PcmVad()) is None
    extract.assert_not_called()


def test_voice_fallback_is_throttled_per_track_and_leaves_face_identity_untouched(monkeypatch):
    import realtime

    match = {"person_id": 5, "person_name": "Vera", "similarity": 0.8}
    extract = Mock(return_value=np.ones(4, dtype=np.float32))
    monkeypatch.setattr(realtime, "VOICE_FALLBACK_INTERVAL_FRAMES", 30)
    monkeypatch.setattr(realtime.voice_core, "has_enrolled_voiceprints", lambda: True)
    monkeypatch.setattr(realtime.voice_core, "extract_voice_embedding", extract)
    monkeypatch.setattr(realtime.voice_core, "match_voice_embedding", lambda *_a, **_k: match)
    track = _voice_fallback_track()

    assert realtime.voice_fallback_match(track, 10, _PcmVad()) == match
    assert realtime.voice_fallback_match(track, 39, _PcmVad()) == match
    assert extract.call_count == 1

    assert realtime.voice_fallback_match(track, 40, _PcmVad()) == match
    assert extract.call_count == 2

    assert (track.person_id, track.person_name, track.confidence) == (None, "Unknown", 0.0)


def test_run_uses_voice_identity_for_display_and_log_without_mutating_track(monkeypatch):
    import realtime
    from tracking import Track

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    class _Cap:
        def __init__(self):
            self.calls = 0

        def isOpened(self):
            return True

        def set(self, *_args, **_kwargs):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 160.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 120.0
            return 0.0

        def read(self):
            self.calls += 1
            if self.calls <= 3:
                return True, frame.copy()
            return False, None

        def release(self):
            return None

    track = Track(track_id=1, center_x=60, center_y=60, area=2500)
    track.lip_open = 1.0

    class _Tracker:
        def __init__(self):
            self.tracks = {1: track}

        def update(self, detections, frame_no):
            return [(detections[0], track)] if detections else []

    class _Mesh:
        def detect_for_video(self, *_args, **_kwargs):
            return SimpleNamespace(face_landmarks=[SimpleNamespace()])

        def close(self):
            return None

    class _Vad(_PcmVad):
        available = True

        def stop(self):
            return None

    speaking_lines = []
    audio_logs = []
    extract = Mock(return_value=np.ones(4, dtype=np.float32))

    def _put_text(_frame, text, *_args, **_kwargs):
        if str(text).startswith("SPEAKING:"):
            speaking_lines.append(str(text))

    def _start_realtime_vad(callback):
        callback(True)
        return _Vad(), True

    monkeypatch.setattr(realtime.cv2, "VideoCapture", lambda *_args, **_kwargs: _Cap())
    monkeypatch.setattr(realtime, "FaceIndex", lambda: SimpleNamespace(search=lambda *_args, **_kwargs: None))
    monkeypatch.setattr(realtime, "CentroidTracker", lambda: _Tracker())
    monkeypatch.setattr(realtime, "start_realtime_vad", _start_realtime_vad)
    monkeypatch.setattr(realtime, "create_face_landmarker", lambda: _Mesh())
    monkeypatch.setattr(realtime, "detect_faces_realtime", lambda *_args, **_kwargs: [(10, 10, 60, 60, 0.9)])
    monkeypatch.setattr(realtime, "extract_embedding", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "face_quality", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(realtime, "log_recognition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "log_audio", lambda *args, **_kwargs: audio_logs.append(args))
    monkeypatch.setattr(realtime, "register_unknown", lambda **_kwargs: 1)
    monkeypatch.setattr(realtime, "find_matching_unknown", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "update_unknown_image", lambda **_kwargs: None)
    monkeypatch.setattr(realtime, "add_unknown_sample", lambda **_kwargs: None)
    monkeypatch.setattr(realtime.np, "save", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "imwrite", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(realtime.voice_core, "has_enrolled_voiceprints", lambda: True)
    monkeypatch.setattr(realtime.voice_core, "extract_voice_embedding", extract)
    monkeypatch.setattr(
        realtime.voice_core,
        "match_voice_embedding",
        lambda *_a, **_k: {"person_id": 5, "person_name": "Vera", "similarity": 0.8},
    )
    monkeypatch.setattr(realtime.cv2, "cvtColor", lambda image, _code: image)
    monkeypatch.setattr(realtime.cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "waitKey", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(realtime.cv2, "getWindowProperty", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(realtime.cv2, "rectangle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "putText", _put_text)
    monkeypatch.setattr(realtime.cv2, "destroyAllWindows", lambda: None)

    import sys
    fake_video_processor = ModuleType("video_processor")
    lip_ratios = iter((0.0, 0.06, 0.0))
    fake_video_processor.lip_open_ratio = lambda _landmarks: next(lip_ratios, 0.0)
    monkeypatch.setitem(sys.modules, "video_processor", fake_video_processor)
    monkeypatch.setitem(
        sys.modules,
        "mediapipe",
        SimpleNamespace(ImageFormat=SimpleNamespace(SRGB=1), Image=lambda image_format, data: data),
    )

    realtime.run(camera=0, width=160, height=120)

    assert speaking_lines == ["SPEAKING: Vera"] * 2
    assert extract.call_count == 1
    assert (track.person_id, track.person_name, track.confidence) == (None, "Unknown", 0.0)
    assert (5, "Vera", 0.8, "realtime") in [
        (log[2], log[3], log[4], log[5]) for log in audio_logs
    ]


def test_run_low_confidence_named_face_uses_voice_fallback_for_display_and_log(monkeypatch):
    import realtime
    from tracking import Track

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    class _Cap:
        def __init__(self):
            self.calls = 0

        def isOpened(self):
            return True

        def set(self, *_args, **_kwargs):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 160.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 120.0
            return 0.0

        def read(self):
            self.calls += 1
            if self.calls <= 3:
                return True, frame.copy()
            return False, None

        def release(self):
            return None

    track = Track(track_id=1, center_x=60, center_y=60, area=2500, person_id=7, person_name="Alice", confidence=0.2)
    track.lip_open = 1.0

    class _Tracker:
        def __init__(self):
            self.tracks = {1: track}

        def update(self, detections, frame_no):
            return [(detections[0], track)] if detections else []

    class _Mesh:
        def detect_for_video(self, *_args, **_kwargs):
            return SimpleNamespace(face_landmarks=[SimpleNamespace()])

        def close(self):
            return None

    class _Vad(_PcmVad):
        available = True

        def stop(self):
            return None

    speaking_lines = []
    audio_logs = []
    extract = Mock(return_value=np.ones(4, dtype=np.float32))

    def _put_text(_frame, text, *_args, **_kwargs):
        if str(text).startswith("SPEAKING:"):
            speaking_lines.append(str(text))

    def _start_realtime_vad(callback):
        callback(True)
        return _Vad(), True

    monkeypatch.setattr(realtime.cv2, "VideoCapture", lambda *_args, **_kwargs: _Cap())
    monkeypatch.setattr(realtime, "FaceIndex", lambda: SimpleNamespace(search=lambda *_args, **_kwargs: None))
    monkeypatch.setattr(realtime, "CentroidTracker", lambda: _Tracker())
    monkeypatch.setattr(realtime, "start_realtime_vad", _start_realtime_vad)
    monkeypatch.setattr(realtime, "create_face_landmarker", lambda: _Mesh())
    monkeypatch.setattr(realtime, "detect_faces_realtime", lambda *_args, **_kwargs: [(10, 10, 60, 60, 0.9)])
    monkeypatch.setattr(realtime, "extract_embedding", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "face_quality", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(realtime, "log_recognition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "log_audio", lambda *args, **_kwargs: audio_logs.append(args))
    monkeypatch.setattr(realtime, "register_unknown", lambda **_kwargs: 1)
    monkeypatch.setattr(realtime, "find_matching_unknown", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "update_unknown_image", lambda **_kwargs: None)
    monkeypatch.setattr(realtime, "add_unknown_sample", lambda **_kwargs: None)
    monkeypatch.setattr(realtime.np, "save", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "imwrite", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(realtime.voice_core, "has_enrolled_voiceprints", lambda: True)
    monkeypatch.setattr(realtime.voice_core, "extract_voice_embedding", extract)
    monkeypatch.setattr(
        realtime.voice_core,
        "match_voice_embedding",
        lambda *_a, **_k: {"person_id": 5, "person_name": "Vera", "similarity": 0.8},
    )
    monkeypatch.setattr(realtime.cv2, "cvtColor", lambda image, _code: image)
    monkeypatch.setattr(realtime.cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "waitKey", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(realtime.cv2, "getWindowProperty", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(realtime.cv2, "rectangle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "putText", _put_text)
    monkeypatch.setattr(realtime.cv2, "destroyAllWindows", lambda: None)

    import sys
    fake_video_processor = ModuleType("video_processor")
    lip_ratios = iter((0.0, 0.06, 0.0))
    fake_video_processor.lip_open_ratio = lambda _landmarks: next(lip_ratios, 0.0)
    monkeypatch.setitem(sys.modules, "video_processor", fake_video_processor)
    monkeypatch.setitem(
        sys.modules,
        "mediapipe",
        SimpleNamespace(ImageFormat=SimpleNamespace(SRGB=1), Image=lambda image_format, data: data),
    )

    realtime.run(camera=0, width=160, height=120)

    assert speaking_lines == ["SPEAKING: Vera"] * 2
    assert extract.call_count == 1
    # Voice fallback must not mutate track identity.
    assert (track.person_id, track.person_name, track.confidence) == (7, "Alice", 0.2)
    # Logged speaker must match displayed fallback identity.
    assert (5, "Vera", 0.8, "realtime") in [
        (log[2], log[3], log[4], log[5]) for log in audio_logs
    ]


def test_run_uses_callback_monotonic_audio_samples_and_logs_wall_clock_event_times(monkeypatch):
    import realtime

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    class _Cap:
        def __init__(self):
            self.calls = 0

        def isOpened(self):
            return True

        def set(self, *_args, **_kwargs):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 160.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 120.0
            return 0.0

        def read(self):
            self.calls += 1
            if self.calls == 1:
                return True, frame.copy()
            return False, None

        def release(self):
            return None

    class _Mesh:
        def detect_for_video(self, *_args, **_kwargs):
            return SimpleNamespace(face_landmarks=[])

        def close(self):
            return None

    class _Tracker:
        def __init__(self):
            self.tracks = {}

        def update(self, detections, frame_no):
            return []

    class _Vad:
        available = True

        def stop(self):
            return None

    class _FakeAttributor:
        def __init__(self):
            self.audio_updates = []
            self._finish_called = False

        def update_audio(self, timestamp, voice_active, voice_confidence=None):
            self.audio_updates.append((timestamp, voice_active, voice_confidence))

        def update_faces(self, timestamp, face_observations):
            return {"timestamp": timestamp, "active_speaker": None}

        def pop_closed_events(self):
            if not self._finish_called:
                return []
            return [{
                "speaker": "UNKNOWN",
                "track_id": None,
                "person_id": None,
                "start_time": 100.1,
                "end_time": 100.3,
                "confidence": 0.6,
            }]

        def finish(self):
            self._finish_called = True

    fake_attributor = _FakeAttributor()
    logged = []

    monotonic_values = iter([100.0, 100.1, 100.2, 100.25, 100.3, 100.35, 100.4])

    def _mono():
        return next(monotonic_values)

    monkeypatch.setattr(realtime.time, "monotonic", _mono)
    monkeypatch.setattr(realtime.time, "time", lambda: 1000.0)
    monkeypatch.setattr(realtime, "SpeakerAttributor", lambda: fake_attributor)
    monkeypatch.setattr(realtime.cv2, "VideoCapture", lambda *_args, **_kwargs: _Cap())
    monkeypatch.setattr(realtime, "FaceIndex", lambda: object())
    monkeypatch.setattr(realtime, "CentroidTracker", lambda: _Tracker())
    monkeypatch.setattr(realtime, "create_face_landmarker", lambda: _Mesh())
    monkeypatch.setattr(realtime, "detect_faces_realtime", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(realtime, "log_recognition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "log_audio", lambda *args, **_kwargs: logged.append(args))
    monkeypatch.setattr(realtime.cv2, "cvtColor", lambda image, _code: image)
    monkeypatch.setattr(realtime.cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "waitKey", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(realtime.cv2, "getWindowProperty", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(realtime.cv2, "rectangle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "putText", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "destroyAllWindows", lambda: None)

    def _start_realtime_vad(callback):
        callback(True)
        return _Vad(), True

    monkeypatch.setattr(realtime, "start_realtime_vad", _start_realtime_vad)

    import sys
    fake_video_processor = ModuleType("video_processor")
    fake_video_processor.lip_open_ratio = lambda _landmarks: 0.0
    monkeypatch.setitem(sys.modules, "video_processor", fake_video_processor)
    monkeypatch.setitem(
        sys.modules,
        "mediapipe",
        SimpleNamespace(ImageFormat=SimpleNamespace(SRGB=1), Image=lambda image_format, data: data),
    )

    realtime.run(camera=0, width=160, height=120)

    # Audio samples must come from callback timestamps, not per-frame polling.
    assert len(fake_attributor.audio_updates) == 1
    assert fake_attributor.audio_updates[0][0] == 100.1
    assert fake_attributor.audio_updates[0][1] is True

    # Logged times must be wall-clock epoch seconds after monotonic->epoch conversion.
    assert len(logged) == 1
    start_time, end_time = logged[0][0], logged[0][1]
    assert abs(start_time - 1000.1) < 1e-6
    assert abs(end_time - 1000.3) < 1e-6


def test_run_logs_unknown_speaking_track(monkeypatch):
    """Mirrors test_speaking_banner_drawn_below_fps_line but with an
    unresolved track, asserting the new code path still logs it."""
    import realtime

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    class _Cap:
        def __init__(self):
            self.calls = 0

        def isOpened(self):
            return True

        def set(self, *_args, **_kwargs):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 160.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 120.0
            return 0.0

        def read(self):
            self.calls += 1
            if self.calls == 1:
                return True, frame.copy()
            return False, None

        def release(self):
            return None

    from tracking import Track

    track = Track(track_id=9, center_x=60, center_y=60, area=2500)
    track.lip_open = 1.0

    class _Tracker:
        def __init__(self):
            self.tracks = {9: track}

        def update(self, detections, frame_no):
            return [(detections[0], track)] if detections else []

    class _Mesh:
        def detect_for_video(self, *_args, **_kwargs):
            return SimpleNamespace(face_landmarks=[SimpleNamespace()])

        def close(self):
            return None

    class _Vad:
        def __init__(self):
            self.available = True

        def stop(self):
            return None

    clock = {"t": 500.0}

    def _monotonic():
        clock["t"] += 0.01
        return clock["t"]

    class _FakeAttributor:
        def __init__(self):
            self._finished = False

        def update_audio(self, *_args, **_kwargs):
            return None

        def update_faces(self, timestamp, _face_observations):
            return {
                "timestamp": timestamp,
                "active_speaker": "UNKNOWN",
                "track_id": 9,
                "confidence": 0.5,
                "run_start_time": 500.10,
                "reason": {},
            }

        def pop_closed_events(self):
            if not self._finished:
                return []
            return [{
                "speaker": "UNKNOWN",
                "track_id": 9,
                "person_id": None,
                "start_time": 500.10,
                "end_time": 500.20,
                "confidence": 0.5,
            }]

        def finish(self):
            self._finished = True

    logged = []

    monkeypatch.setattr(realtime.cv2, "VideoCapture", lambda *_args, **_kwargs: _Cap())
    monkeypatch.setattr(realtime.time, "monotonic", _monotonic)
    monkeypatch.setattr(realtime, "SpeakerAttributor", lambda: _FakeAttributor())
    monkeypatch.setattr(realtime, "FaceIndex", lambda: SimpleNamespace(search=lambda *_args, **_kwargs: None))
    monkeypatch.setattr(realtime, "CentroidTracker", lambda: _Tracker())

    def _start_realtime_vad(callback):
        callback(True)
        return _Vad(), True

    monkeypatch.setattr(realtime, "start_realtime_vad", _start_realtime_vad)
    monkeypatch.setattr(realtime, "create_face_landmarker", lambda: _Mesh())
    monkeypatch.setattr(realtime, "detect_faces_realtime", lambda *_args, **_kwargs: [(10, 10, 60, 60, 0.9)])
    monkeypatch.setattr(realtime, "extract_embedding", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "face_quality", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(realtime, "log_recognition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "log_audio", lambda *args, **kwargs: logged.append((args, kwargs)))
    monkeypatch.setattr(realtime.voice_core, "has_enrolled_voiceprints", lambda: False)
    monkeypatch.setattr(realtime.cv2, "cvtColor", lambda image, _code: image)
    monkeypatch.setattr(realtime.cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "waitKey", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(realtime.cv2, "getWindowProperty", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(realtime.cv2, "rectangle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "putText", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "destroyAllWindows", lambda: None)

    dummy_mp = SimpleNamespace(ImageFormat=SimpleNamespace(SRGB=1), Image=lambda image_format, data: data)
    import sys
    fake_video_processor = ModuleType("video_processor")
    fake_video_processor.lip_open_ratio = lambda _landmarks: 1.0
    monkeypatch.setitem(sys.modules, "video_processor", fake_video_processor)
    monkeypatch.setitem(sys.modules, "mediapipe", dummy_mp)

    realtime.run(camera=0, width=160, height=120)

    assert len(logged) == 1, "an UNKNOWN speaking track should be logged once on realtime exit, not skipped"
    args, kwargs = logged[0]
    # log_audio(start, end, person_id, person_name, confidence, source, track_id=...)
    assert args[2] is None
    assert args[3] == "UNKNOWN"
    assert args[5] == "realtime"
    assert kwargs["track_id"] == 9


def test_run_records_realtime_attendance_once_per_track_person_pair(monkeypatch):
    import realtime

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    class _Cap:
        def __init__(self):
            self.calls = 0

        def isOpened(self):
            return True

        def set(self, *_args, **_kwargs):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 160.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 120.0
            return 0.0

        def read(self):
            self.calls += 1
            if self.calls <= 3:
                return True, frame.copy()
            return False, None

        def release(self):
            return None

    class _Mesh:
        def detect_for_video(self, *_args, **_kwargs):
            return SimpleNamespace(face_landmarks=[])

        def close(self):
            return None

    class _Track:
        def __init__(self, track_id, person_id, person_name, confidence):
            self.track_id = track_id
            self.person_id = person_id
            self.person_name = person_name
            self.confidence = confidence
            self.embedding = None
            self.center_x = 80
            self.center_y = 60
            self.lip_open = None

    def _det(track_width):
        return {"bbox": (10, 10, 10 + track_width, 70), "lip_open": None}

    track_high = _Track(1, 11, "Alice", 0.95)
    track_low = _Track(2, 12, "Bob", 0.20)
    track_unknown = _Track(3, None, "Unknown", 0.0)
    track_new = _Track(4, 11, "Alice", 0.99)

    class _Tracker:
        def __init__(self):
            self.tracks = {
                1: track_high,
                2: track_low,
                3: track_unknown,
                4: track_new,
            }
            self._frame_no = 0

        def update(self, _detections, _frame_no):
            self._frame_no += 1
            if self._frame_no == 1:
                return [
                    (_det(80), track_high),
                    (_det(80), track_low),
                    (_det(10), track_unknown),
                ]
            if self._frame_no == 2:
                return [(_det(80), track_high)]
            if self._frame_no == 3:
                return [(_det(80), track_new)]
            return []

    class _MonoClock:
        def __init__(self):
            self.value = 100.0

        def __call__(self):
            current = self.value
            self.value += 1.0
            return current

    attendance_calls = []
    recognition_calls = []

    monkeypatch.setattr(realtime.time, "monotonic", _MonoClock())
    monkeypatch.setattr(realtime.time, "time", lambda: 1000.0)
    monkeypatch.setattr(realtime.cv2, "VideoCapture", lambda *_args, **_kwargs: _Cap())
    monkeypatch.setattr(realtime, "FaceIndex", lambda: SimpleNamespace(search=lambda *_args, **_kwargs: None))
    monkeypatch.setattr(realtime, "CentroidTracker", lambda: _Tracker())
    monkeypatch.setattr(realtime, "start_realtime_vad", lambda callback: (None, False))
    monkeypatch.setattr(realtime, "create_face_landmarker", lambda: _Mesh())
    monkeypatch.setattr(realtime, "detect_faces_realtime", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(realtime, "extract_embedding", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "log_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        realtime,
        "log_recognition",
        lambda *args, **kwargs: recognition_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        realtime,
        "record_attendance_event",
        lambda **kwargs: attendance_calls.append(kwargs),
        raising=False,
    )
    monkeypatch.setattr(realtime.cv2, "cvtColor", lambda image, _code: image)
    monkeypatch.setattr(realtime.cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "waitKey", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(realtime.cv2, "getWindowProperty", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(realtime.cv2, "rectangle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "putText", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "destroyAllWindows", lambda: None)

    dummy_mp = SimpleNamespace(
        ImageFormat=SimpleNamespace(SRGB=1),
        Image=lambda image_format, data: data,
    )
    import sys
    fake_video_processor = ModuleType("video_processor")
    fake_video_processor.lip_open_ratio = lambda _landmarks: 0.0
    monkeypatch.setitem(sys.modules, "video_processor", fake_video_processor)
    monkeypatch.setitem(sys.modules, "mediapipe", dummy_mp)

    realtime.run(camera=0, width=160, height=120)

    assert len(attendance_calls) == 2
    assert {(c["track_id"], c["person_id"]) for c in attendance_calls} == {(1, 11), (4, 11)}

    for call in attendance_calls:
        assert call["source"] == "realtime"
        assert call["person_name"] == "Alice"
        assert call["confidence"] >= realtime.RECOGNITION_THRESHOLD
        assert uuid.UUID(call["source_ref"])  # raises if not UUID
        observed = datetime.fromisoformat(call["observed_at_utc"])
        assert observed.tzinfo == timezone.utc

    assert attendance_calls[0]["source_ref"] == attendance_calls[1]["source_ref"]

    expected_first = datetime.fromtimestamp(1001.0, timezone.utc).isoformat()
    assert attendance_calls[0]["observed_at_utc"] == expected_first

    # Recognition logging cadence remains active while attendance is deduped.
    assert recognition_calls


def test_run_retries_transient_attendance_failure_with_throttle(monkeypatch):
    import realtime

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    class _Cap:
        def __init__(self):
            self.calls = 0

        def isOpened(self):
            return True

        def set(self, *_args, **_kwargs):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 160.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 120.0
            return 0.0

        def read(self):
            self.calls += 1
            if self.calls <= 4:
                return True, frame.copy()
            return False, None

        def release(self):
            return None

    class _Mesh:
        def detect_for_video(self, *_args, **_kwargs):
            return SimpleNamespace(face_landmarks=[])

        def close(self):
            return None

    class _Track:
        def __init__(self):
            self.track_id = 7
            self.person_id = 77
            self.person_name = "Alice"
            self.confidence = 0.95
            self.embedding = None
            self.center_x = 80
            self.center_y = 60
            self.lip_open = None

    track = _Track()

    class _Tracker:
        def __init__(self):
            self.tracks = {7: track}

        def update(self, _detections, _frame_no):
            return [({"bbox": (10, 10, 90, 90), "lip_open": None}, track)]

    class _MonoClock:
        def __init__(self):
            self.values = [200.0, 200.4, 200.8, 201.2, 201.6]
            self.index = 0

        def __call__(self):
            if self.index < len(self.values):
                value = self.values[self.index]
                self.index += 1
                return value
            self.values.append(self.values[-1] + 0.4)
            self.index += 1
            return self.values[-1]

    attendance_attempts = []
    successful_events = []

    def _record_attendance_event(**kwargs):
        attendance_attempts.append(kwargs)
        if len(attendance_attempts) == 1:
            raise RuntimeError("transient write failure")
        successful_events.append(kwargs)
        return True

    monkeypatch.setattr(realtime.time, "monotonic", _MonoClock())
    monkeypatch.setattr(realtime.time, "time", lambda: 1200.0)
    monkeypatch.setattr(realtime.cv2, "VideoCapture", lambda *_args, **_kwargs: _Cap())
    monkeypatch.setattr(realtime, "FaceIndex", lambda: SimpleNamespace(search=lambda *_args, **_kwargs: None))
    monkeypatch.setattr(realtime, "CentroidTracker", lambda: _Tracker())
    monkeypatch.setattr(realtime, "start_realtime_vad", lambda callback: (None, False))
    monkeypatch.setattr(realtime, "create_face_landmarker", lambda: _Mesh())
    monkeypatch.setattr(realtime, "detect_faces_realtime", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(realtime, "extract_embedding", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "log_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "log_recognition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "record_attendance_event", _record_attendance_event, raising=False)
    monkeypatch.setattr(realtime.cv2, "cvtColor", lambda image, _code: image)
    monkeypatch.setattr(realtime.cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "waitKey", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(realtime.cv2, "getWindowProperty", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(realtime.cv2, "rectangle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "putText", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "destroyAllWindows", lambda: None)

    dummy_mp = SimpleNamespace(
        ImageFormat=SimpleNamespace(SRGB=1),
        Image=lambda image_format, data: data,
    )
    import sys
    fake_video_processor = ModuleType("video_processor")
    fake_video_processor.lip_open_ratio = lambda _landmarks: 0.0
    monkeypatch.setitem(sys.modules, "video_processor", fake_video_processor)
    monkeypatch.setitem(sys.modules, "mediapipe", dummy_mp)

    realtime.run(camera=0, width=160, height=120)

    assert len(attendance_attempts) == 2
    assert len(successful_events) == 1
    assert successful_events[0]["track_id"] == 7
    assert successful_events[0]["person_id"] == 77


def test_run_clamps_attendance_confidence_to_one(monkeypatch):
    import realtime

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    class _Cap:
        def __init__(self):
            self.calls = 0

        def isOpened(self):
            return True

        def set(self, *_args, **_kwargs):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 160.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 120.0
            return 0.0

        def read(self):
            self.calls += 1
            if self.calls <= 1:
                return True, frame.copy()
            return False, None

        def release(self):
            return None

    class _Mesh:
        def detect_for_video(self, *_args, **_kwargs):
            return SimpleNamespace(face_landmarks=[])

        def close(self):
            return None

    class _Track:
        def __init__(self):
            self.track_id = 2
            self.person_id = 21
            self.person_name = "Alice"
            self.confidence = 1.0000001
            self.embedding = None
            self.center_x = 80
            self.center_y = 60
            self.lip_open = None

    track = _Track()

    class _Tracker:
        def __init__(self):
            self.tracks = {2: track}

        def update(self, _detections, _frame_no):
            return [({"bbox": (10, 10, 90, 90), "lip_open": None}, track)]

    attendance_calls = []

    monkeypatch.setattr(realtime.cv2, "VideoCapture", lambda *_args, **_kwargs: _Cap())
    monkeypatch.setattr(realtime, "FaceIndex", lambda: SimpleNamespace(search=lambda *_args, **_kwargs: None))
    monkeypatch.setattr(realtime, "CentroidTracker", lambda: _Tracker())
    monkeypatch.setattr(realtime, "start_realtime_vad", lambda callback: (None, False))
    monkeypatch.setattr(realtime, "create_face_landmarker", lambda: _Mesh())
    monkeypatch.setattr(realtime, "detect_faces_realtime", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(realtime, "extract_embedding", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "log_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "log_recognition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        realtime,
        "record_attendance_event",
        lambda **kwargs: attendance_calls.append(kwargs),
        raising=False,
    )
    monkeypatch.setattr(realtime.cv2, "cvtColor", lambda image, _code: image)
    monkeypatch.setattr(realtime.cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "waitKey", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(realtime.cv2, "getWindowProperty", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(realtime.cv2, "rectangle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "putText", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "destroyAllWindows", lambda: None)

    dummy_mp = SimpleNamespace(
        ImageFormat=SimpleNamespace(SRGB=1),
        Image=lambda image_format, data: data,
    )
    import sys
    fake_video_processor = ModuleType("video_processor")
    fake_video_processor.lip_open_ratio = lambda _landmarks: 0.0
    monkeypatch.setitem(sys.modules, "video_processor", fake_video_processor)
    monkeypatch.setitem(sys.modules, "mediapipe", dummy_mp)

    realtime.run(camera=0, width=160, height=120)

    assert len(attendance_calls) == 1
    assert attendance_calls[0]["confidence"] == 1.0


def test_run_initializes_database_before_processing(monkeypatch):
    import realtime

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    class _Cap:
        def __init__(self):
            self.calls = 0

        def isOpened(self):
            return True

        def set(self, *_args, **_kwargs):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 160.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 120.0
            return 0.0

        def read(self):
            self.calls += 1
            if self.calls <= 1:
                return True, frame.copy()
            return False, None

        def release(self):
            return None

    class _Mesh:
        def detect_for_video(self, *_args, **_kwargs):
            return SimpleNamespace(face_landmarks=[])

        def close(self):
            return None

    class _Tracker:
        def __init__(self):
            self.tracks = {}

        def update(self, _detections, _frame_no):
            return []

    init_calls = []

    monkeypatch.setattr(realtime.cv2, "VideoCapture", lambda *_args, **_kwargs: _Cap())
    monkeypatch.setattr(realtime, "FaceIndex", lambda: SimpleNamespace(search=lambda *_args, **_kwargs: None))
    monkeypatch.setattr(realtime, "CentroidTracker", lambda: _Tracker())
    monkeypatch.setattr(realtime, "start_realtime_vad", lambda callback: (None, False))
    monkeypatch.setattr(realtime, "create_face_landmarker", lambda: _Mesh())
    monkeypatch.setattr(realtime, "detect_faces_realtime", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(realtime, "log_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "log_recognition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime, "init_db", lambda: init_calls.append(True), raising=False)
    monkeypatch.setattr(realtime.cv2, "cvtColor", lambda image, _code: image)
    monkeypatch.setattr(realtime.cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "waitKey", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(realtime.cv2, "getWindowProperty", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(realtime.cv2, "rectangle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "putText", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(realtime.cv2, "destroyAllWindows", lambda: None)

    dummy_mp = SimpleNamespace(
        ImageFormat=SimpleNamespace(SRGB=1),
        Image=lambda image_format, data: data,
    )
    import sys
    fake_video_processor = ModuleType("video_processor")
    fake_video_processor.lip_open_ratio = lambda _landmarks: 0.0
    monkeypatch.setitem(sys.modules, "video_processor", fake_video_processor)
    monkeypatch.setitem(sys.modules, "mediapipe", dummy_mp)

    realtime.run(camera=0, width=160, height=120)

    assert init_calls == [True]


def test_run_raises_when_database_initialization_fails(monkeypatch):
    import realtime

    class _Cap:
        def isOpened(self):
            return True

        def set(self, *_args, **_kwargs):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 160.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 120.0
            return 0.0

        def read(self):
            return False, None

        def release(self):
            return None

    monkeypatch.setattr(realtime.cv2, "VideoCapture", lambda *_args, **_kwargs: _Cap())
    monkeypatch.setattr(
        realtime,
        "init_db",
        Mock(side_effect=RuntimeError("init failed")),
        raising=False,
    )
    monkeypatch.setattr(realtime.cv2, "destroyAllWindows", lambda: None)

    with pytest.raises(RuntimeError, match="init failed"):
        realtime.run(camera=0, width=160, height=120)
