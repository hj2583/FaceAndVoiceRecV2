import importlib
import logging
import os
from typing import Any, Iterable, Optional

import numpy as np

from config import EMBEDDING_DIM, FACE_BACKEND, FACE_BACKEND_FALLBACK

logger = logging.getLogger(__name__)


def _normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm <= 1e-8:
        return None
    return (vector / norm).astype(np.float32)


class BaseFaceBackend:
    name = "base"

    def detect_faces(self, image):
        raise NotImplementedError

    def extract_embedding(self, face_crop):
        raise NotImplementedError

    def build_model(self, model_name):
        return None


class DeepFaceBackend(BaseFaceBackend):
    name = "deepface"

    def __init__(self):
        from deepface import DeepFace

        self._deepface = DeepFace

    def detect_faces(self, image):
        faces = self._deepface.extract_faces(
            img_path=image,
            detector_backend="retinaface",
            enforce_detection=False,
            expand_percentage=0,
        )
        detections = []
        for face in faces:
            area = face.get("facial_area", {})
            x = float(area.get("x", 0))
            y = float(area.get("y", 0))
            width = float(area.get("w", 0))
            height = float(area.get("h", 0))
            confidence = float(face.get("confidence", 0.0))
            if width > 0 and height > 0 and confidence >= 0.0:
                detections.append((x, y, width, height, confidence))
        return detections

    def extract_embedding(self, face_crop):
        if face_crop is None or face_crop.size == 0:
            return None

        result = self._deepface.represent(
            img_path=face_crop,
            model_name="ArcFace",
            detector_backend="skip",
            enforce_detection=False,
            align=True,
        )
        if not result:
            return None

        embedding = _normalize(result[0]["embedding"])
        if embedding is None:
            return None

        if embedding.shape[0] != EMBEDDING_DIM:
            logger.error(
                "Unexpected ArcFace embedding size: %s (expected %s)",
                embedding.shape[0],
                EMBEDDING_DIM,
            )
            return None

        return embedding

    def build_model(self, model_name):
        return self._deepface.build_model(model_name)


class UniFaceBackend(BaseFaceBackend):
    name = "uniface"

    def __init__(self):
        self._impl = self._resolve_impl()

    def _resolve_impl(self):
        try:
            import uniface
            from uniface import FaceAnalyzer
            from uniface.detection import SCRFD
            from uniface.constants import SCRFDWeights
            from uniface.recognition import ArcFace
        except Exception as exc:
            raise RuntimeError("UniFace package is not installed or misconfigured") from exc

        analyzer = FaceAnalyzer(
            detector=SCRFD(model_name=SCRFDWeights.SCRFD_500M_KPS),
            recognizer=ArcFace(),
        )
        return analyzer

    def detect_faces(self, image):
        if image is None or image.size == 0:
            return []

        faces = self._impl.analyze(image)
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
            detected = self._impl.analyze(face_crop)
            if not detected:
                return None
            embedding = np.asarray(detected[0].embedding, dtype=np.float32).reshape(-1)
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

    def build_model(self, model_name):
        return None


class FaceBackendFactory:
    @staticmethod
    def create(preferred: Optional[str] = None, fallback: Optional[str] = None):
        preferred_name = preferred or os.getenv("FACE_BACKEND") or FACE_BACKEND
        fallback_name = fallback or FACE_BACKEND_FALLBACK or "deepface"
        for backend_name in (preferred_name, fallback_name):
            try:
                backend = FaceBackendFactory._build_backend(backend_name)
                if backend is not None:
                    logger.info("Using face backend: %s", backend_name)
                    return backend
            except Exception as exc:
                logger.warning("Face backend %s unavailable: %s", backend_name, exc)

        raise RuntimeError("No usable face backend is available")

    @staticmethod
    def _build_backend(backend_name: str):
        backend_name = (backend_name or "deepface").lower()
        if backend_name == "deepface":
            return DeepFaceBackend()
        if backend_name == "uniface":
            return UniFaceBackend()
        raise ValueError(f"Unsupported face backend: {backend_name}")


def get_face_backend(preferred: Optional[str] = None, fallback: Optional[str] = None):
    return FaceBackendFactory.create(preferred=preferred, fallback=fallback)
