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


def test_audio_state_switches_to_unavailable_when_worker_dies():
    class _DeadVad:
        def __init__(self):
            self.available = False

    vad = _DeadVad()
    audio_available = True

    if vad is not None and audio_available and not getattr(vad, "available", False):
        audio_available = False

    assert audio_available is False


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
    monkeypatch.setattr(realtime, "detect_faces_opencv", lambda *_args, **_kwargs: [])
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


def test_cleanup_continues_when_vad_stop_raises():
    cleanup_steps = []
    vad = Mock()
    vad.stop.side_effect = RuntimeError("stop failed")
    cap = Mock()
    mesh = Mock()

    try:
        try:
            vad.stop()
        except Exception:
            cleanup_steps.append("vad_error")

        try:
            cap.release()
            cleanup_steps.append("cap_released")
        except Exception:
            cleanup_steps.append("cap_error")

        try:
            mesh.close()
            cleanup_steps.append("mesh_closed")
        except Exception:
            cleanup_steps.append("mesh_error")
    finally:
        pass

    assert "vad_error" in cleanup_steps
    assert "cap_released" in cleanup_steps
    assert "mesh_closed" in cleanup_steps


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
    monkeypatch.setattr(realtime, "detect_faces_opencv", lambda *_args, **_kwargs: [(10, 10, 40, 40, 0.9)])
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
    monkeypatch.setattr(realtime, "detect_faces_opencv", lambda *_args, **_kwargs: [(10, 10, 60, 60, 0.9)])
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
