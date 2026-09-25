import threading
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

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
            if self.calls == 1:
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
    fake_video_processor.lip_open_ratio = lambda _landmarks: 1.0
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
    fake_video_processor.lip_open_ratio = lambda _landmarks: 1.0
    monkeypatch.setitem(sys.modules, "video_processor", fake_video_processor)
    monkeypatch.setitem(
        sys.modules,
        "mediapipe",
        SimpleNamespace(ImageFormat=SimpleNamespace(SRGB=1), Image=lambda image_format, data: data),
    )

    realtime.run(camera=0, width=160, height=120)

    assert speaking_lines == ["SPEAKING: Vera"] * 3
    assert extract.call_count == 1
    assert (track.person_id, track.person_name, track.confidence) == (None, "Unknown", 0.0)
    assert [(log[2], log[3], log[4], log[5]) for log in audio_logs] == [(5, "Vera", 0.8, "realtime")]
