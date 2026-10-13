"""Smoke tests for package-qualified storage and face-domain imports."""

import inspect

from face_voice_rec.domain.face.detection_core import detect_faces_tiled
from face_voice_rec.domain.face.face_backend import get_face_backend
from face_voice_rec.domain.face.face_core import FaceIndex
from face_voice_rec.domain.face.tracking import CentroidTracker
from face_voice_rec.storage.database import init_db


def test_storage_and_face_package_imports() -> None:
    """Verify representative storage and face symbols are importable via package paths."""
    assert inspect.isfunction(init_db)
    assert inspect.isclass(FaceIndex)
    assert inspect.isfunction(get_face_backend)
    assert inspect.isfunction(detect_faces_tiled)
    assert inspect.isclass(CentroidTracker)
