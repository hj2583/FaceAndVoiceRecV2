# Long-Range Face Detection (Tiling + RetinaFace) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `video_processor.py` (offline video pipeline) reliably detect attendees up to ~15m by replacing the close-range-tuned MediaPipe detector with a tiled RetinaFace detection pass, while keeping the existing tracker/recognition/logging pipeline unchanged.

**Architecture:** A new `detection_core.py` module splits each frame into overlapping upscaled tiles, runs RetinaFace (via DeepFace) on each tile plus one full-frame pass, remaps all boxes to full-frame coordinates, and merges duplicates with NMS. `video_processor.py`'s per-frame loop calls this module for face boxes instead of relying on MediaPipe's detector; MediaPipe FaceLandmarker is still used, but only on crops large enough to yield reliable lip landmarks (for active-speaker detection).

**Tech Stack:** Python, OpenCV, DeepFace (RetinaFace backend, TensorFlow), MediaPipe (FaceLandmarker, used for landmarks only), NumPy, pytest (new dev dependency).

## Global Constraints

- Offline (`video_processor.py`) only — `realtime.py` is explicitly out of scope for this plan.
- No camera hardware changes; must work with a standard fixed wide-lens 1080p webcam.
- GPU acceleration relies on TensorFlow auto-detecting CUDA if present; no CPU-only fallback toggle is required (per user decision).
- Recognition confidence naturally degrades at range and should fall back to "Unknown" — do not loosen `RECOGNITION_THRESHOLD` or `AMBIGUITY_MARGIN`.
- Detection failures on individual tiles must not abort frame processing — treat as "no face in this tile".

---

### Task 1: Test infrastructure + new config values

**Files:**
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`
- Modify: `config.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: config constants `TILE_GRID`, `TILE_OVERLAP`, `TILE_UPSCALE`, `TILED_DETECTION_INTERVAL`, `NMS_IOU_THRESHOLD`, `LANDMARK_MIN_FACE_SIZE`, `FACE_UPSCALE_TARGET_SIZE`, and updated `MIN_RECOGNITION_FACE_SIZE`, consumed by all later tasks.

- [ ] **Step 1: Install pytest and add it to requirements.txt**

Run:
```powershell
& "d:\Git\FaceAndVoiceRecV2\3.12venv\Scripts\python.exe" -m pip install pytest
```

Add a new line at the end of `requirements.txt`:
```
pytest>=8.0
```

- [ ] **Step 2: Create test package files**

Create `tests/__init__.py` with empty content.

Create `tests/conftest.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

- [ ] **Step 3: Write the failing config test**

Create `tests/test_config.py`:
```python
import config


def test_tile_grid_is_two_dimensional():
    cols, rows = config.TILE_GRID
    assert cols >= 1
    assert rows >= 1


def test_tile_overlap_is_a_fraction():
    assert 0.0 <= config.TILE_OVERLAP < 1.0


def test_tile_upscale_is_positive():
    assert config.TILE_UPSCALE > 0


def test_nms_iou_threshold_is_a_fraction():
    assert 0.0 <= config.NMS_IOU_THRESHOLD <= 1.0


def test_landmark_min_face_size_not_smaller_than_min_face_size():
    assert config.LANDMARK_MIN_FACE_SIZE >= config.MIN_FACE_SIZE


def test_min_recognition_face_size_lowered_for_long_range():
    assert config.MIN_RECOGNITION_FACE_SIZE <= 15


def test_face_upscale_target_size_matches_arcface_input():
    assert config.FACE_UPSCALE_TARGET_SIZE == 112
```

- [ ] **Step 4: Run tests to verify they fail**

Run:
```powershell
& "d:\Git\FaceAndVoiceRecV2\3.12venv\Scripts\python.exe" -m pytest tests/test_config.py -v
```
Expected: FAIL with `AttributeError: module 'config' has no attribute 'TILE_GRID'` (and similar).

- [ ] **Step 5: Add the new config values**

In `config.py`, change:
```python
# Minimum face size before attempting ArcFace recognition.
MIN_RECOGNITION_FACE_SIZE = 30
```
to:
```python
# Minimum face size before attempting ArcFace recognition.
#
# Lowered from 30 to allow best-effort recognition attempts on
# small/distant faces detected via the long-range tiled pipeline.
# Low-confidence matches still fall back to "Unknown" via
# RECOGNITION_THRESHOLD / AMBIGUITY_MARGIN, unchanged.
MIN_RECOGNITION_FACE_SIZE = 15
```

Then, immediately after the existing `MAX_FACES = 8` line and before `# Tracker settings.`, insert:
```python

# Minimum face size (pixels) required to attempt MediaPipe lip-landmark
# extraction for active-speaker detection. Smaller/distant faces skip
# lip-based speaker candidacy rather than using unreliable landmarks.
LANDMARK_MIN_FACE_SIZE = 60
```

Then, immediately after the existing `FACE_UPSCALE_FACTOR = 2.0` line, insert:
```python

# Target minimum dimension (pixels) small faces are upscaled to before
# ArcFace embedding extraction, matching ArcFace's expected ~112x112 input.
FACE_UPSCALE_TARGET_SIZE = 112
```

Then, at the end of the `# Face detection / tracking` section content (after `DEBUG_FACE_SIZE = True`), add a new section:
```python

# ============================================================
# Long-range tiled detection (video_processor.py only)
# ============================================================

# Tile grid used to split a frame for long-range detection: (columns, rows).
TILE_GRID = (3, 2)

# Fractional overlap between adjacent tiles, so faces near tile
# borders aren't cut off.
TILE_OVERLAP = 0.2

# Upscale factor applied to each tile before running detection, to
# increase effective pixel density on distant faces.
TILE_UPSCALE = 2.0

# Run the full tiled detection pass every N frames; rely on tracker
# continuity in between (mirrors RECOGNITION_INTERVAL).
TILED_DETECTION_INTERVAL = 5

# IoU threshold for de-duplicating detections seen in overlapping tiles.
NMS_IOU_THRESHOLD = 0.4
```

- [ ] **Step 6: Run tests to verify they pass**

Run:
```powershell
& "d:\Git\FaceAndVoiceRecV2\3.12venv\Scripts\python.exe" -m pytest tests/test_config.py -v
```
Expected: PASS (7 passed).

- [ ] **Step 7: Commit**

```powershell
git add tests/__init__.py tests/conftest.py tests/test_config.py config.py requirements.txt
git commit -m "Add long-range detection config values and pytest infra"
```

---

### Task 2: `tile_frame()` — split a frame into overlapping tiles

**Files:**
- Create: `detection_core.py`
- Create: `tests/test_detection_core.py`

**Interfaces:**
- Consumes: `config.TILE_GRID`, `config.TILE_OVERLAP`, `config.TILE_UPSCALE`
- Produces: `tile_frame(frame) -> list[dict]`, each dict with keys `"image"` (np.ndarray), `"offset_x"` (int), `"offset_y"` (int), `"scale"` (float). Consumed by Task 5's `detect_faces_tiled`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_detection_core.py`:
```python
import numpy as np

import config
from detection_core import tile_frame


def test_tile_frame_produces_one_tile_per_grid_cell():
    frame = np.zeros((240, 360, 3), dtype=np.uint8)

    tiles = tile_frame(frame)

    cols, rows = config.TILE_GRID
    assert len(tiles) == cols * rows


def test_tile_frame_tiles_cover_frame_bounds():
    frame = np.zeros((240, 360, 3), dtype=np.uint8)
    height, width = frame.shape[:2]

    tiles = tile_frame(frame)

    for tile in tiles:
        assert 0 <= tile["offset_x"] < width
        assert 0 <= tile["offset_y"] < height
        assert tile["scale"] == config.TILE_UPSCALE


def test_tile_frame_upscales_tile_images():
    frame = np.zeros((240, 360, 3), dtype=np.uint8)

    tiles = tile_frame(frame)

    cols, rows = config.TILE_GRID
    approx_tile_w = (360 / cols) * (1 + config.TILE_OVERLAP)
    approx_tile_h = (240 / rows) * (1 + config.TILE_OVERLAP)

    first_tile = tiles[0]
    tile_h, tile_w = first_tile["image"].shape[:2]

    assert tile_w >= approx_tile_w * config.TILE_UPSCALE * 0.5
    assert tile_h >= approx_tile_h * config.TILE_UPSCALE * 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
& "d:\Git\FaceAndVoiceRecV2\3.12venv\Scripts\python.exe" -m pytest tests/test_detection_core.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'detection_core'`.

- [ ] **Step 3: Create `detection_core.py` with `tile_frame()`**

```python
"""Tiled long-range face detection helpers for offline video processing."""

import logging

import cv2

from config import (
    TILE_GRID,
    TILE_OVERLAP,
    TILE_UPSCALE,
)

logger = logging.getLogger(__name__)


def tile_frame(frame):
    """
    Split a frame into overlapping tiles for long-range face detection.

    Returns a list of dicts:
        {
            "image": np.ndarray,  # upscaled tile image
            "offset_x": int,      # tile's top-left x in the original frame
            "offset_y": int,      # tile's top-left y in the original frame
            "scale": float,       # upscale factor applied to this tile
        }
    """
    height, width = frame.shape[:2]
    cols, rows = TILE_GRID

    tile_w = width / cols
    tile_h = height / rows

    overlap_x = tile_w * TILE_OVERLAP
    overlap_y = tile_h * TILE_OVERLAP

    tiles = []

    for row in range(rows):
        for col in range(cols):
            x0 = max(0, int(col * tile_w - overlap_x))
            y0 = max(0, int(row * tile_h - overlap_y))
            x1 = min(width, int((col + 1) * tile_w + overlap_x))
            y1 = min(height, int((row + 1) * tile_h + overlap_y))

            crop = frame[y0:y1, x0:x1]

            if crop.size == 0:
                continue

            upscaled = cv2.resize(
                crop,
                None,
                fx=TILE_UPSCALE,
                fy=TILE_UPSCALE,
                interpolation=cv2.INTER_LINEAR,
            )

            tiles.append(
                {
                    "image": upscaled,
                    "offset_x": x0,
                    "offset_y": y0,
                    "scale": TILE_UPSCALE,
                }
            )

    return tiles
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```powershell
& "d:\Git\FaceAndVoiceRecV2\3.12venv\Scripts\python.exe" -m pytest tests/test_detection_core.py -v
```
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```powershell
git add detection_core.py tests/test_detection_core.py
git commit -m "Add tile_frame() for long-range face detection tiling"
```

---

### Task 3: `merge_detections()` — remap + NMS across tiles

**Files:**
- Modify: `detection_core.py`
- Modify: `tests/test_detection_core.py`

**Interfaces:**
- Consumes: `config.NMS_IOU_THRESHOLD`
- Produces: `merge_detections(tile_results, iou_threshold=NMS_IOU_THRESHOLD) -> list[tuple[float, float, float, float, float]]` where each tuple is `(x0, y0, x1, y1, confidence)` in full-frame coordinates. `tile_results` is `list[tuple[dict, list[tuple]]]` — pairs of (tile metadata dict from `tile_frame()`, boxes list from Task 4's `detect_faces_retinaface()`). Consumed by Task 5's `detect_faces_tiled`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_detection_core.py`:
```python
from detection_core import merge_detections


def _tile_meta(offset_x=0, offset_y=0, scale=1.0):
    return {"offset_x": offset_x, "offset_y": offset_y, "scale": scale}


def test_merge_detections_remaps_tile_local_coordinates():
    tile_results = [
        (_tile_meta(offset_x=100, offset_y=50, scale=2.0), [(20, 20, 60, 60, 0.9)]),
    ]

    merged = merge_detections(tile_results)

    assert len(merged) == 1
    x0, y0, x1, y1, confidence = merged[0]
    assert (x0, y0, x1, y1) == (110.0, 60.0, 130.0, 80.0)
    assert confidence == 0.9


def test_merge_detections_removes_duplicate_overlapping_boxes():
    tile_results = [
        (_tile_meta(), [(10, 10, 50, 50, 0.95)]),
        (_tile_meta(), [(12, 12, 52, 52, 0.80)]),
    ]

    merged = merge_detections(tile_results, iou_threshold=0.4)

    assert len(merged) == 1
    assert merged[0][4] == 0.95


def test_merge_detections_keeps_non_overlapping_boxes():
    tile_results = [
        (_tile_meta(), [(10, 10, 50, 50, 0.9), (200, 200, 240, 240, 0.9)]),
    ]

    merged = merge_detections(tile_results, iou_threshold=0.4)

    assert len(merged) == 2


def test_merge_detections_returns_empty_list_for_no_boxes():
    assert merge_detections([]) == []
    assert merge_detections([(_tile_meta(), [])]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
& "d:\Git\FaceAndVoiceRecV2\3.12venv\Scripts\python.exe" -m pytest tests/test_detection_core.py -v
```
Expected: FAIL with `ImportError: cannot import name 'merge_detections'`.

- [ ] **Step 3: Add `merge_detections()` and NMS helpers to `detection_core.py`**

Append to `detection_core.py`:
```python


def merge_detections(tile_results, iou_threshold=None):
    """
    Remap tile-local detections to full-frame coordinates and merge
    duplicate detections of the same face across overlapping tiles
    using Non-Maximum Suppression (NMS).

    tile_results: list of (tile_meta, boxes) where tile_meta is a dict
    with "offset_x", "offset_y", "scale" (as produced by tile_frame(),
    or {"offset_x": 0, "offset_y": 0, "scale": 1.0} for a full-frame
    pass), and boxes is a list of (x0, y0, x1, y1, confidence) tuples
    in tile-local pixel coordinates.

    Returns a list of (x0, y0, x1, y1, confidence) tuples in full-frame
    pixel coordinates, deduplicated.
    """
    if iou_threshold is None:
        iou_threshold = NMS_IOU_THRESHOLD

    all_boxes = []

    for tile_meta, boxes in tile_results:
        scale = tile_meta["scale"]
        offset_x = tile_meta["offset_x"]
        offset_y = tile_meta["offset_y"]

        for x0, y0, x1, y1, confidence in boxes:
            all_boxes.append(
                (
                    offset_x + x0 / scale,
                    offset_y + y0 / scale,
                    offset_x + x1 / scale,
                    offset_y + y1 / scale,
                    confidence,
                )
            )

    if not all_boxes:
        return []

    return _non_max_suppress(all_boxes, iou_threshold)


def _intersection_over_union(box_a, box_b):
    ax0, ay0, ax1, ay1 = box_a[:4]
    bx0, by0, bx1, by1 = box_b[:4]

    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)

    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def _non_max_suppress(boxes, iou_threshold):
    ordered = sorted(boxes, key=lambda box: box[4], reverse=True)

    kept = []

    while ordered:
        current = ordered.pop(0)
        kept.append(current)

        ordered = [
            box
            for box in ordered
            if _intersection_over_union(current, box) < iou_threshold
        ]

    return kept
```

Also update the top-level import block in `detection_core.py` to include `NMS_IOU_THRESHOLD`:
```python
from config import (
    TILE_GRID,
    TILE_OVERLAP,
    TILE_UPSCALE,
    NMS_IOU_THRESHOLD,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```powershell
& "d:\Git\FaceAndVoiceRecV2\3.12venv\Scripts\python.exe" -m pytest tests/test_detection_core.py -v
```
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```powershell
git add detection_core.py tests/test_detection_core.py
git commit -m "Add merge_detections() NMS logic for tiled face detection"
```

---

### Task 4: `detect_faces_retinaface()` — DeepFace RetinaFace wrapper

**Files:**
- Modify: `detection_core.py`
- Modify: `tests/test_detection_core.py`

**Interfaces:**
- Consumes: `deepface.DeepFace.extract_faces`
- Produces: `detect_faces_retinaface(image) -> list[tuple[float, float, float, float, float]]`, each tuple `(x0, y0, x1, y1, confidence)` in image-local pixel coordinates. Consumed by Task 5's `detect_faces_tiled`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_detection_core.py`:
```python
from unittest.mock import patch

from detection_core import detect_faces_retinaface


def test_detect_faces_retinaface_converts_deepface_output_to_boxes():
    fake_result = [
        {
            "facial_area": {"x": 10, "y": 20, "w": 30, "h": 40},
            "confidence": 0.87,
        },
    ]

    with patch("detection_core.DeepFace.extract_faces", return_value=fake_result):
        boxes = detect_faces_retinaface("fake_image")

    assert boxes == [(10, 20, 40, 60, 0.87)]


def test_detect_faces_retinaface_skips_zero_confidence_results():
    fake_result = [
        {
            "facial_area": {"x": 0, "y": 0, "w": 100, "h": 100},
            "confidence": 0.0,
        },
    ]

    with patch("detection_core.DeepFace.extract_faces", return_value=fake_result):
        boxes = detect_faces_retinaface("fake_image")

    assert boxes == []


def test_detect_faces_retinaface_returns_empty_list_on_exception():
    with patch(
        "detection_core.DeepFace.extract_faces",
        side_effect=ValueError("backend failure"),
    ):
        boxes = detect_faces_retinaface("fake_image")

    assert boxes == []
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
& "d:\Git\FaceAndVoiceRecV2\3.12venv\Scripts\python.exe" -m pytest tests/test_detection_core.py -v
```
Expected: FAIL with `ImportError: cannot import name 'detect_faces_retinaface'`.

- [ ] **Step 3: Add `detect_faces_retinaface()` to `detection_core.py`**

Add this import at the top of `detection_core.py` (after the `cv2` import):
```python
from deepface import DeepFace
```

Append the function to `detection_core.py`:
```python


def detect_faces_retinaface(image):
    """
    Run RetinaFace face detection (via DeepFace) on a single image
    (a full frame or a tile).

    Returns a list of bounding boxes in image-local pixel coordinates:
        [(x0, y0, x1, y1, confidence), ...]

    Detection errors (e.g. backend failure) and zero-confidence
    "whole image" fallbacks (returned by DeepFace when
    enforce_detection=False finds no face) are treated as "no faces
    found" rather than raised or reported as detections.
    """
    try:
        faces = DeepFace.extract_faces(
            img_path=image,
            detector_backend="retinaface",
            enforce_detection=False,
            align=False,
        )
    except Exception:
        logger.warning("RetinaFace detection failed for an image", exc_info=True)
        return []

    boxes = []

    for face in faces:
        region = face.get("facial_area")
        confidence = float(face.get("confidence", 0.0))

        if not region or confidence <= 0.0:
            continue

        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("w", 0)
        h = region.get("h", 0)

        if w <= 0 or h <= 0:
            continue

        boxes.append((x, y, x + w, y + h, confidence))

    return boxes
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```powershell
& "d:\Git\FaceAndVoiceRecV2\3.12venv\Scripts\python.exe" -m pytest tests/test_detection_core.py -v
```
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```powershell
git add detection_core.py tests/test_detection_core.py
git commit -m "Add detect_faces_retinaface() DeepFace wrapper"
```

---

### Task 5: `detect_faces_tiled()` orchestrator + GPU status logging

**Files:**
- Modify: `detection_core.py`
- Modify: `tests/test_detection_core.py`

**Interfaces:**
- Consumes: `tile_frame()`, `detect_faces_retinaface()`, `merge_detections()`
- Produces: `detect_faces_tiled(frame) -> list[tuple[float, float, float, float, float]]` in full-frame coordinates — this is the main entry point Task 6 (`video_processor.py`) calls. Also produces `log_gpu_status() -> None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_detection_core.py`:
```python
import numpy as np

from detection_core import detect_faces_tiled


def test_detect_faces_tiled_merges_tile_and_full_frame_detections():
    frame = np.zeros((240, 360, 3), dtype=np.uint8)

    def fake_detect(image):
        h, w = image.shape[:2]
        return [(0, 0, w // 2, h // 2, 0.5)]

    with patch("detection_core.detect_faces_retinaface", side_effect=fake_detect):
        boxes = detect_faces_tiled(frame)

    assert len(boxes) >= 1
    for x0, y0, x1, y1, confidence in boxes:
        assert 0 <= x0 < x1 <= 360
        assert 0 <= y0 < y1 <= 240


def test_detect_faces_tiled_returns_empty_list_when_nothing_detected():
    frame = np.zeros((240, 360, 3), dtype=np.uint8)

    with patch("detection_core.detect_faces_retinaface", return_value=[]):
        boxes = detect_faces_tiled(frame)

    assert boxes == []
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
& "d:\Git\FaceAndVoiceRecV2\3.12venv\Scripts\python.exe" -m pytest tests/test_detection_core.py -v
```
Expected: FAIL with `ImportError: cannot import name 'detect_faces_tiled'`.

- [ ] **Step 3: Add `detect_faces_tiled()` and `log_gpu_status()` to `detection_core.py`**

Append to `detection_core.py`:
```python


def detect_faces_tiled(frame):
    """
    Detect faces across a full frame using tiling for long-range recall
    plus one full-frame pass for cheap near-range detection.

    Returns a list of (x0, y0, x1, y1, confidence) tuples in
    full-frame pixel coordinates.
    """
    log_gpu_status()

    tile_results = []

    for tile in tile_frame(frame):
        boxes = detect_faces_retinaface(tile["image"])
        tile_results.append((tile, boxes))

    full_frame_meta = {"offset_x": 0, "offset_y": 0, "scale": 1.0}
    full_frame_boxes = detect_faces_retinaface(frame)
    tile_results.append((full_frame_meta, full_frame_boxes))

    return merge_detections(tile_results)


_gpu_status_logged = False


def log_gpu_status():
    """
    Log once whether TensorFlow can see a GPU. RetinaFace (via DeepFace)
    runs on TensorFlow, and tiling multiplies detector calls per frame,
    so GPU acceleration matters a lot for offline processing time.
    """
    global _gpu_status_logged

    if _gpu_status_logged:
        return

    _gpu_status_logged = True

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")

        if gpus:
            logger.info(
                "Long-range detection: %d GPU(s) found, using GPU acceleration: %s",
                len(gpus),
                gpus,
            )
        else:
            logger.warning(
                "Long-range detection: no GPU found, running RetinaFace on CPU. "
                "Tiled detection will be significantly slower."
            )

    except Exception:
        logger.warning(
            "Could not query TensorFlow GPU devices for long-range detection.",
            exc_info=True,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```powershell
& "d:\Git\FaceAndVoiceRecV2\3.12venv\Scripts\python.exe" -m pytest tests/test_detection_core.py -v
```
Expected: PASS (12 passed).

- [ ] **Step 5: Commit**

```powershell
git add detection_core.py tests/test_detection_core.py
git commit -m "Add detect_faces_tiled() orchestrator and GPU status logging"
```

---

### Task 6: Extend `prepare_recognition_crop()` for very small distant faces

**Files:**
- Modify: `video_processor.py`
- Create: `tests/test_video_processor.py`

**Interfaces:**
- Consumes: `config.FACE_UPSCALE_TARGET_SIZE`, `config.FACE_UPSCALE_THRESHOLD`, `config.MIN_RECOGNITION_FACE_SIZE`
- Produces: updated `prepare_recognition_crop(crop)` behavior (same signature, same module, same function name) — consumed by the existing recognition call site in `process_video_pipeline` (unchanged call site).

- [ ] **Step 1: Write the failing test**

Create `tests/test_video_processor.py`:
```python
import numpy as np

from video_processor import prepare_recognition_crop
import config


def test_prepare_recognition_crop_rejects_faces_below_minimum_size():
    tiny = np.zeros((10, 10, 3), dtype=np.uint8)
    assert prepare_recognition_crop(tiny) is None


def test_prepare_recognition_crop_upscales_small_faces_to_target_size():
    small = np.zeros((20, 20, 3), dtype=np.uint8)

    result = prepare_recognition_crop(small)

    assert result is not None
    height, width = result.shape[:2]
    assert min(height, width) >= config.FACE_UPSCALE_TARGET_SIZE


def test_prepare_recognition_crop_leaves_large_faces_unchanged():
    large = np.zeros((150, 150, 3), dtype=np.uint8)

    result = prepare_recognition_crop(large)

    assert result.shape == large.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
& "d:\Git\FaceAndVoiceRecV2\3.12venv\Scripts\python.exe" -m pytest tests/test_video_processor.py -v
```
Expected: FAIL on `test_prepare_recognition_crop_upscales_small_faces_to_target_size` — current fixed `FACE_UPSCALE_FACTOR = 2.0` only produces a 40x40 crop, below the 112 target.

- [ ] **Step 3: Update `prepare_recognition_crop()` in `video_processor.py`**

Replace:
```python
    recognition_crop = crop

    # --------------------------------------------------------
    # Upscale smaller but usable faces.
    # --------------------------------------------------------

    if (
        face_width < FACE_UPSCALE_THRESHOLD
        or face_height < FACE_UPSCALE_THRESHOLD
    ):

        recognition_crop = cv2.resize(
            crop,
            None,
            fx=FACE_UPSCALE_FACTOR,
            fy=FACE_UPSCALE_FACTOR,
            interpolation=cv2.INTER_CUBIC,
        )

    return recognition_crop
```
with:
```python
    recognition_crop = crop

    # --------------------------------------------------------
    # Upscale smaller but usable faces up to a target size that
    # matches ArcFace's expected ~112x112 input, instead of a
    # fixed factor. This matters for the small/distant faces the
    # long-range tiled detection pipeline now allows through.
    # --------------------------------------------------------

    if (
        face_width < FACE_UPSCALE_THRESHOLD
        or face_height < FACE_UPSCALE_THRESHOLD
    ):

        scale = max(
            1.0,
            FACE_UPSCALE_TARGET_SIZE / min(face_width, face_height),
        )

        recognition_crop = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LANCZOS4,
        )

    return recognition_crop
```

Update the import block near the top of `video_processor.py` — replace:
```python
    FACE_UPSCALE_THRESHOLD,
    FACE_UPSCALE_FACTOR,
```
with:
```python
    FACE_UPSCALE_THRESHOLD,
    FACE_UPSCALE_TARGET_SIZE,
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```powershell
& "d:\Git\FaceAndVoiceRecV2\3.12venv\Scripts\python.exe" -m pytest tests/test_video_processor.py -v
```
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```powershell
git add video_processor.py tests/test_video_processor.py
git commit -m "Upscale small recognition crops to a target size instead of a fixed factor"
```

---

### Task 7: Integrate tiled detection into `process_video_pipeline`

**Files:**
- Modify: `video_processor.py`

**Interfaces:**
- Consumes: `detection_core.detect_faces_tiled(frame) -> list[tuple[float, float, float, float, float]]`, `config.TILED_DETECTION_INTERVAL`, `config.LANDMARK_MIN_FACE_SIZE`
- Produces: same `detections` list shape consumed by the existing tracker (`{"bbox": (x0,y0,x1,y1), "landmarks": landmarks_or_None, "lip_open": float_or_None}`), and a guarded `speaking_candidates` append that skips `None` lip data.

This task has no isolated unit test — it modifies the main per-frame loop of `process_video_pipeline`, which depends on video I/O, the MediaPipe model file, and DeepFace/TensorFlow, and is validated end-to-end in Task 8's manual test instead.

- [ ] **Step 1: Add the `detection_core` import**

In `video_processor.py`, after the existing:
```python
from tracking import CentroidTracker
```
add:
```python

from detection_core import detect_faces_tiled
```

- [ ] **Step 2: Add `TILED_DETECTION_INTERVAL` and `LANDMARK_MIN_FACE_SIZE` to the config import block**

Replace:
```python
    MIN_FACE_SIZE,
    MIN_RECOGNITION_FACE_SIZE,
```
with:
```python
    MIN_FACE_SIZE,
    MIN_RECOGNITION_FACE_SIZE,
    LANDMARK_MIN_FACE_SIZE,
    TILED_DETECTION_INTERVAL,
```

- [ ] **Step 3: Replace the per-frame MediaPipe-only detection block**

Replace this whole block (from the `mp_image = mp.Image(` line through the end of the `detections.append(...)` loop, i.e. everything between the `# BGR → RGB` section and the `# Tracking` section):
```python
            # =================================================
            # MediaPipe image
            # =================================================

            mp_image = mp.Image(
                image_format=(
                    mp.ImageFormat.SRGB
                ),
                data=rgb,
            )

            # =================================================
            # MediaPipe timestamp
            # =================================================

            timestamp_ms = int(
                timestamp * 1000
            )

            result = (
                face_landmarker
                .detect_for_video(
                    mp_image,
                    timestamp_ms,
                )
            )

            detections = []

            # =================================================
            # Face detections
            # =================================================

            for landmarks in result.face_landmarks:

                if not landmarks:
                    continue

                bbox = bbox_from_landmarks(
                    landmarks,
                    frame.shape,
                )

                x0, y0, x1, y1 = bbox

                face_width = x1 - x0
                face_height = y1 - y0

                # ------------------------------------------------
                # Detection/tracking filter.
                #
                # This is intentionally separate from
                # MIN_RECOGNITION_FACE_SIZE.
                # ------------------------------------------------

                if (
                    face_width < MIN_FACE_SIZE
                    or face_height < MIN_FACE_SIZE
                ):
                    continue

                lip_open = (
                    lip_open_ratio(
                        landmarks
                    )
                )

                detections.append(
                    {
                        "bbox": bbox,
                        "landmarks": landmarks,
                        "lip_open": lip_open,
                    }
                )
```
with:
```python
            # =================================================
            # MediaPipe timestamp (also used to space out
            # per-face landmark calls within this frame)
            # =================================================

            timestamp_ms = int(
                timestamp * 1000
            )

            # =================================================
            # Long-range tiled face detection
            # =================================================

            if frame_no % TILED_DETECTION_INTERVAL == 0:
                raw_boxes = detect_faces_tiled(frame)
            else:
                raw_boxes = []

            detections = []
            landmark_call_offset = 0

            for x0, y0, x1, y1, _confidence in raw_boxes:

                x0, y0, x1, y1 = (
                    int(x0),
                    int(y0),
                    int(x1),
                    int(y1),
                )

                face_width = x1 - x0
                face_height = y1 - y0

                # ------------------------------------------------
                # Detection/tracking filter.
                #
                # This is intentionally separate from
                # MIN_RECOGNITION_FACE_SIZE.
                # ------------------------------------------------

                if (
                    face_width < MIN_FACE_SIZE
                    or face_height < MIN_FACE_SIZE
                ):
                    continue

                landmarks = None
                lip_open = None

                # ------------------------------------------------
                # Only attempt lip-landmark extraction on faces
                # large enough for reliable landmarks. Smaller/
                # distant faces are still detected and tracked,
                # but skip active-speaker candidacy.
                # ------------------------------------------------

                if (
                    face_width >= LANDMARK_MIN_FACE_SIZE
                    and face_height >= LANDMARK_MIN_FACE_SIZE
                ):

                    crop_rgb = rgb[y0:y1, x0:x1]

                    if crop_rgb.size:

                        crop_mp_image = mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=crop_rgb,
                        )

                        landmark_call_offset += 1

                        crop_result = (
                            face_landmarker
                            .detect_for_video(
                                crop_mp_image,
                                timestamp_ms + landmark_call_offset,
                            )
                        )

                        if crop_result.face_landmarks:

                            landmarks = (
                                crop_result.face_landmarks[0]
                            )

                            lip_open = (
                                lip_open_ratio(
                                    landmarks
                                )
                            )

                detections.append(
                    {
                        "bbox": (x0, y0, x1, y1),
                        "landmarks": landmarks,
                        "lip_open": lip_open,
                    }
                )
```

- [ ] **Step 4: Guard the active-speaker candidate check against missing lip data**

Replace:
```python
                # =================================================
                # Active speaker candidates
                # =================================================

                if (
                    audio_is_speech
                    and
                    track.lip_open
                    >= LIP_OPEN_THRESHOLD
                ):

                    speaking_candidates.append(
                        track
                    )
```
with:
```python
                # =================================================
                # Active speaker candidates
                #
                # track.lip_open is None for faces too small/distant
                # for reliable landmarks (see LANDMARK_MIN_FACE_SIZE
                # gate above) — they are excluded from candidacy.
                # =================================================

                if (
                    audio_is_speech
                    and
                    track.lip_open is not None
                    and
                    track.lip_open
                    >= LIP_OPEN_THRESHOLD
                ):

                    speaking_candidates.append(
                        track
                    )
```

- [ ] **Step 5: Run the full existing test suite to confirm no regressions**

Run:
```powershell
& "d:\Git\FaceAndVoiceRecV2\3.12venv\Scripts\python.exe" -m pytest tests/ -v
```
Expected: all tests still PASS (this task doesn't change any tested unit, only the untested main loop).

- [ ] **Step 6: Commit**

```powershell
git add video_processor.py
git commit -m "Integrate tiled long-range face detection into process_video_pipeline"
```

---

### Task 8: Manual validation with a real distance test video

**Files:** none (manual validation task, no code changes)

- [ ] **Step 1: Record or obtain a test video**

Using the fixed webcam setup, record (or use an existing) video with attendees standing/sitting at marked distances: 2m, 5m, 10m, and 15m from the camera, each facing the camera for at least a few seconds.

- [ ] **Step 2: Run the video through the updated pipeline**

Start the app:
```powershell
& "d:\Git\FaceAndVoiceRecV2\3.12venv\Scripts\python.exe" -m streamlit run app.py
```
Upload the test video via the video-processing tab and let it process to completion.

- [ ] **Step 3: Check detection recall at each distance**

Open the output tracked video and the JSON log. For each of the 2m/5m/10m/15m markers, confirm a bounding box is drawn around the attendee at that distance (detection recall is the priority per the design — a box appearing, even labeled "Unknown"/"Recognition Pending", counts as success).

- [ ] **Step 4: Note recognition behavior at range**

Record which distances still produce a correct name match vs. fall back to "Unknown" — this is expected to degrade at range per the design's accepted trade-off, not a bug.

- [ ] **Step 5: Check processing time**

Note total processing time for the video versus its length, to confirm it's acceptable for offline use. If unacceptably slow, the first things to tune are `TILE_GRID` (fewer tiles), `TILE_UPSCALE` (lower factor), and `TILED_DETECTION_INTERVAL` (larger interval) in `config.py`.

- [ ] **Step 6: Record findings**

Write a short note (in the PR description or a follow-up comment) summarizing detection recall by distance and processing time, so thresholds can be tuned in a follow-up if needed.
