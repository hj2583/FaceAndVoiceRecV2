import logging
import os
import sys
import threading
from pathlib import Path

import numpy as np

from config import EMBEDDING_DIM

logger = logging.getLogger(__name__)


def _configure_cuda_dlls():
    if os.name != "nt":
        return

    site_packages = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    dll_dirs = [
        site_packages / "cu13" / "bin",
        site_packages / "cu13" / "bin" / "x86_64",
        site_packages / "cudnn" / "bin",
    ]

    for dll_dir in dll_dirs:
        if not dll_dir.exists():
            continue
        os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(dll_dir))


def _cuda_providers():
    _configure_cuda_dlls()
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is unavailable")
    return ["CUDAExecutionProvider"]


def get_cuda_providers():
    return _cuda_providers()


def _require_cuda_session(session, component):
    active_provider = session.get_providers()[0]
    if active_provider != "CUDAExecutionProvider":
        raise RuntimeError(
            f"{component} is running on {active_provider}; "
            "CUDAExecutionProvider is required"
        )


def _normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        return None
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return (vector / norm).astype(np.float32)


def _fallback_landmarks(image):
    template = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    height, width = image.shape[:2]
    return template * np.array(
        [width / 112.0, height / 112.0],
        dtype=np.float32,
    )


class UniFaceBackend:
    """Face detection + recognition backed entirely by UniFace (SCRFD + ArcFace)."""

    name = "uniface"

    def __init__(self):
        from uniface.constants import SCRFDWeights
        from uniface.detection import SCRFD
        from uniface.recognition import ArcFace

        providers = _cuda_providers()
        self._detector = SCRFD(
            model_name=SCRFDWeights.SCRFD_10G_KPS,
            confidence_threshold=0.3,
            providers=providers,
        )
        _require_cuda_session(self._detector.session, "SCRFD detector")
        self._realtime_detector = SCRFD(
            model_name=SCRFDWeights.SCRFD_500M_KPS,
            confidence_threshold=0.4,
            providers=providers,
        )
        _require_cuda_session(self._realtime_detector.session, "realtime SCRFD detector")
        self._recognizer = ArcFace(providers=providers)
        _require_cuda_session(self._recognizer.session, "ArcFace recognizer")

    def detect_faces_realtime(self, image):
        if image is None or getattr(image, "size", 0) == 0:
            return []

        faces = self._realtime_detector.detect(image)
        detections = []
        for face in faces:
            bbox = np.asarray(face.bbox, dtype=np.float32)
            if bbox.shape[0] < 4:
                continue
            x1, y1, x2, y2 = map(float, bbox[:4])
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            if width > 0 and height > 0:
                detections.append((x1, y1, width, height, float(face.confidence)))
        return detections

    def detect_faces(self, image):
        if image is None or getattr(image, "size", 0) == 0:
            return []

        faces = self._detector.detect(image)
        detections = []
        for face in faces:
            bbox = np.asarray(face.bbox, dtype=np.float32)
            if bbox.shape[0] < 4:
                continue
            x1, y1, x2, y2 = map(float, bbox[:4])
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            if width > 0 and height > 0:
                detections.append((x1, y1, width, height, float(face.confidence)))
        return detections

    def extract_embedding(self, face_crop):
        if face_crop is None or face_crop.size == 0:
            return None

        try:
            # ArcFace needs 5-point alignment landmarks in the same coordinate
            # space as the image, so re-detect on the (already cropped) face.
            faces = self._detector.detect(face_crop)
            landmarks = (
                max(faces, key=lambda f: f.confidence).landmarks
                if faces
                else _fallback_landmarks(face_crop)
            )
            embedding = np.asarray(
                self._recognizer.get_normalized_embedding(face_crop, landmarks),
                dtype=np.float32,
            ).reshape(-1)

            normalized = _normalize(embedding)
            if normalized is None:
                return None
            if normalized.shape[0] != EMBEDDING_DIM:
                logger.error(
                    "Unexpected UniFace embedding size: %s (expected %s)",
                    normalized.shape[0],
                    EMBEDDING_DIM,
                )
                return None
            return normalized
        except Exception:
            logger.exception("UniFace embedding extraction failed")
            return None

    def build_model(self, model_name=None):
        # UniFace lazily downloads/caches its ONNX weights on first use, so
        # there is nothing extra to warm up beyond constructing the backend.
        return None


_backend_lock = threading.Lock()
_backend_instance = None


def get_face_backend():
    global _backend_instance
    if _backend_instance is None:
        with _backend_lock:
            if _backend_instance is None:
                _backend_instance = UniFaceBackend()
    return _backend_instance
