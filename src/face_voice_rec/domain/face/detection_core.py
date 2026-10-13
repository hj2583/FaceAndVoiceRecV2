"""Long-range face detection helpers.

The geometry and NMS functions are dependency-light so they can be tested
without loading the RetinaFace model.  Model inference is lazy and failures
in one tile are isolated from the remaining tiles.
"""

import logging
import threading
from typing import List, Tuple

import cv2
import os
import numpy as np


logger = logging.getLogger(__name__)

Detection = Tuple[float, float, float, float, float]


# Lightweight cached OpenCV cascade for fallback/realtime detection
_OPENCV_CASCADE = None
_OPENCV_CASCADE_LOCK = threading.Lock()


def _get_opencv_cascade():
    global _OPENCV_CASCADE
    if _OPENCV_CASCADE is None:
        with _OPENCV_CASCADE_LOCK:
            if _OPENCV_CASCADE is None:
                # Build the cascade path portably so tests can override cv2.data.haarcascades
                cascade_filename = 'haarcascade_frontalface_default.xml'
                base = getattr(cv2, "data", None)
                # Ensure we always pass a string to os.path.join. Some OpenCV
                # builds (or test doubles) may provide `haarcascades = None`;
                # coerce falsy/None values to an empty string so join() never
                # receives None as an argument.
                haar_dir = (getattr(base, "haarcascades", "") or "") if base is not None else ""
                cascade_path = os.path.join(haar_dir, cascade_filename)
                _OPENCV_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _OPENCV_CASCADE


def ensure_opencv_face_detector_available():
    cascade = _get_opencv_cascade()
    if cascade is None or cascade.empty():
        raise RuntimeError(
            "OpenCV Haar cascade data is unavailable. "
            "Install a supported OpenCV wheel with: "
            "python -m pip install \"opencv-contrib-python>=4.10,<4.12\""
        )


def _starts(length: int, tile_length: int, count: int) -> List[int]:
    if count <= 1:
        return [0]
    if tile_length >= length:
        return [0] * count

    step = (length - tile_length) / (count - 1)
    return [int(round(index * step)) for index in range(count)]


def tile_frame(
    frame: np.ndarray,
    grid: Tuple[int, int],
    overlap_ratio: float,
) -> List[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]]:
    """Split a frame into overlapping tiles that cover its full extent."""
    if frame is None or frame.size == 0:
        return []

    cols, rows = grid
    if cols < 1 or rows < 1:
        raise ValueError("grid must contain positive column and row counts")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("overlap_ratio must be in the range [0.0, 1.0)")

    height, width = frame.shape[:2]
    base_width = int(np.ceil(width / cols))
    base_height = int(np.ceil(height / rows))
    tile_width = min(width, int(np.ceil(base_width / (1.0 - overlap_ratio))))
    tile_height = min(height, int(np.ceil(base_height / (1.0 - overlap_ratio))))
    x_starts = _starts(width, tile_width, cols)
    y_starts = _starts(height, tile_height, rows)

    tiles = []
    for row, y_start in enumerate(y_starts):
        for col, x_start in enumerate(x_starts):
            y_end = min(height, y_start + tile_height)
            x_end = min(width, x_start + tile_width)
            tiles.append(
                (frame[y_start:y_end, x_start:x_end].copy(),
                 (col, row),
                 (x_start, y_start))
            )
    return tiles


def upscale_tile(tile: np.ndarray, scale_factor: float) -> np.ndarray:
    """Upscale a tile while preserving its aspect ratio."""
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive")
    if scale_factor == 1.0:
        return tile

    height, width = tile.shape[:2]
    return cv2.resize(
        tile,
        (max(1, int(round(width * scale_factor))),
         max(1, int(round(height * scale_factor)))),
        interpolation=cv2.INTER_CUBIC if scale_factor > 1 else cv2.INTER_AREA,
    )


def remap_tile_detections(
    detections: List[Detection],
    tile_origin: Tuple[int, int],
    frame_shape: Tuple[int, int],
    upscale_factor: float,
) -> List[Detection]:
    """Map boxes from an upscaled tile into full-frame coordinates."""
    if upscale_factor <= 0:
        raise ValueError("upscale_factor must be positive")

    height, width = frame_shape[:2]
    origin_x, origin_y = tile_origin
    remapped = []

    for x, y, box_width, box_height, confidence in detections:
        x = origin_x + x / upscale_factor
        y = origin_y + y / upscale_factor
        box_width /= upscale_factor
        box_height /= upscale_factor

        x = max(0.0, min(float(width), x))
        y = max(0.0, min(float(height), y))
        box_width = min(box_width, width - x)
        box_height = min(box_height, height - y)

        if box_width > 0 and box_height > 0:
            remapped.append((x, y, box_width, box_height, confidence))

    return remapped


def _calculate_iou(box1: Detection, box2: Detection) -> float:
    x1, y1, width1, height1, _ = box1
    x2, y2, width2, height2, _ = box2
    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + width1, x2 + width2)
    bottom = min(y1 + height1, y2 + height2)
    if right <= left or bottom <= top:
        return 0.0

    intersection = (right - left) * (bottom - top)
    union = width1 * height1 + width2 * height2 - intersection
    return intersection / union if union > 0 else 0.0


def nms_merge(
    detections: List[Detection],
    iou_threshold: float = 0.4,
) -> List[Detection]:
    """Remove lower-confidence boxes that duplicate stronger detections."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in the range [0.0, 1.0]")

    remaining = sorted(detections, key=lambda detection: detection[4], reverse=True)
    kept = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [
            detection
            for detection in remaining
            if _calculate_iou(best, detection) <= iou_threshold
        ]
    return kept


def _detect_faces_opencv(tile: np.ndarray) -> List[Detection]:
    """Fallback detector using OpenCV cascade classifier."""
    try:
        cascade = _get_opencv_cascade()
        if cascade is None or cascade.empty():
            return []

        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
        )

        detections = []
        for x, y, w, h in faces:
            # OpenCV cascade doesn't provide confidence, so use a moderate default
            detections.append((float(x), float(y), float(w), float(h), 0.7))
        return detections
    except Exception as error:
        logger.warning("OpenCV fallback detection failed: %s", error)
        return []


def detect_faces_opencv(
    frame: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: tuple[int, int] = (20, 20),
) -> List[Detection]:
    """Lightweight public OpenCV-backed detector with caching.

    Returns a list of Detection tuples `(x, y, width, height, confidence)`.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return []

    try:
        cascade = _get_opencv_cascade()
        if cascade is None or cascade.empty():
            logger.warning("OpenCV face cascade is unavailable")
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )

        detections: List[Detection] = []
        for x, y, w, h in faces:
            detections.append((float(x), float(y), float(w), float(h), 0.7))
        return detections
    except Exception as error:
        logger.warning("OpenCV face detection failed: %s", error)
        return []


def _detect_faces_uniface(tile: np.ndarray) -> List[Detection]:
    """Run the UniFace SCRFD detector lazily and return tile-space boxes.
    Falls back to OpenCV cascade if UniFace fails.
    """
    try:
        from face_backend import get_face_backend

        detections = get_face_backend().detect_faces(tile)
    except Exception as error:
        logger.warning("UniFace detection failed, falling back to OpenCV: %s", error)
        return _detect_faces_opencv(tile)

    # Filter out very low-confidence detections
    return [d for d in detections if d[4] >= 0.5]


def detect_faces_tiled(
    frame: np.ndarray,
    tile_grid: Tuple[int, int] = (3, 2),
    overlap_ratio: float = 0.2,
    upscale_factor: float = 2.0,
    nms_iou_threshold: float = 0.4,
    min_confidence: float = 0.5,
) -> List[Detection]:
    """Detect faces in overlapping upscaled tiles and merge duplicate boxes.
    
    Includes automatic fallback from UniFace to OpenCV if primary detector fails.
    Filters detections by minimum confidence threshold.
    """
    all_detections = []

    try:
        full_frame_detections = _detect_faces_uniface(frame)
        # Apply confidence filtering to full-frame detections as well
        full_frame_detections = [d for d in full_frame_detections if d[4] >= min_confidence]
        all_detections.extend(
            remap_tile_detections(
                full_frame_detections,
                (0, 0),
                frame.shape[:2],
                1.0,
            )
        )
    except Exception as error:
        logger.warning("Skipping failed full-frame detection: %s", error)

    for tile, _tile_index, origin in tile_frame(frame, tile_grid, overlap_ratio):
        try:
            upscaled = upscale_tile(tile, upscale_factor)
            detections = _detect_faces_uniface(upscaled)
            # Filter by confidence before remapping
            detections = [d for d in detections if d[4] >= min_confidence]
            all_detections.extend(
                remap_tile_detections(
                    detections,
                    origin,
                    frame.shape[:2],
                    upscale_factor,
                )
            )
        except Exception as error:
            logger.warning("Skipping failed detection tile at %s: %s", origin, error)

    return nms_merge(all_detections, nms_iou_threshold)