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
