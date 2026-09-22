from unittest.mock import Mock

import numpy as np

import face_backend


def _make_backend(detector, recognizer):
    backend = face_backend.UniFaceBackend.__new__(face_backend.UniFaceBackend)
    backend._detector = detector
    backend._recognizer = recognizer
    return backend


def test_detect_faces_converts_uniface_faces_to_detection_tuples():
    fake_face = Mock(bbox=np.array([10.0, 20.0, 30.0, 50.0]), confidence=0.87)
    detector = Mock()
    detector.detect.return_value = [fake_face]
    backend = _make_backend(detector, recognizer=Mock())

    detections = backend.detect_faces(np.zeros((100, 100, 3), dtype=np.uint8))

    assert detections == [(10.0, 20.0, 20.0, 30.0, 0.87)]


def test_detect_faces_returns_empty_list_for_invalid_image():
    backend = _make_backend(detector=Mock(), recognizer=Mock())
    assert backend.detect_faces(None) == []
    assert backend.detect_faces(np.empty((0, 0, 3), dtype=np.uint8)) == []


def test_extract_embedding_aligns_using_redetected_landmarks():
    landmarks = np.zeros((5, 2), dtype=np.float32)
    fake_face = Mock(landmarks=landmarks, confidence=0.9)
    detector = Mock()
    detector.detect.return_value = [fake_face]

    raw_embedding = np.ones(512, dtype=np.float32)
    recognizer = Mock()
    recognizer.get_normalized_embedding.return_value = raw_embedding

    backend = _make_backend(detector, recognizer)
    face_crop = np.zeros((112, 112, 3), dtype=np.uint8)

    embedding = backend.extract_embedding(face_crop)

    recognizer.get_normalized_embedding.assert_called_once_with(face_crop, landmarks)
    assert embedding is not None
    assert embedding.shape == (512,)
    assert np.isclose(np.linalg.norm(embedding), 1.0)


def test_extract_embedding_uses_alignment_template_when_no_face_found_in_crop():
    detector = Mock()
    detector.detect.return_value = []
    recognizer = Mock()
    recognizer.get_normalized_embedding.return_value = np.ones(512, dtype=np.float32)
    backend = _make_backend(detector, recognizer=recognizer)

    embedding = backend.extract_embedding(np.zeros((112, 112, 3), dtype=np.uint8))

    assert embedding is not None
    assert embedding.shape == (512,)
    landmarks = recognizer.get_normalized_embedding.call_args.args[1]
    assert landmarks.shape == (5, 2)


def test_extract_embedding_returns_none_for_empty_crop():
    backend = _make_backend(detector=Mock(), recognizer=Mock())
    assert backend.extract_embedding(None) is None
    assert backend.extract_embedding(np.empty((0, 0, 3), dtype=np.uint8)) is None


def test_get_face_backend_returns_singleton(monkeypatch):
    monkeypatch.setattr(face_backend, "_backend_instance", None)
    created = []

    class FakeBackend:
        def __init__(self):
            created.append(self)

    monkeypatch.setattr(face_backend, "UniFaceBackend", FakeBackend)

    first = face_backend.get_face_backend()
    second = face_backend.get_face_backend()

    assert first is second
    assert len(created) == 1