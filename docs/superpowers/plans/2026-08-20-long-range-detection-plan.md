# Long-Range Face Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace MediaPipe's 2m face detector with a tiled RetinaFace pipeline to detect attendees up to 15m range while maintaining tracker/recognition compatibility.

**Architecture:** Introduce `detection_core.py` (new module) with tiling, upscaling, RetinaFace detection, coordinate remapping, and NMS. Modify `video_processor.py` to call `detect_faces_tiled()` instead of MediaPipe detector. Add face upscaling to `face_core.py` before ArcFace embedding. Update `config.py` with tile parameters.

**Tech Stack:** Python 3.12, OpenCV, DeepFace (RetinaFace backend, TensorFlow), MediaPipe (landmarks only), NumPy, pytest

## Global Constraints

- Offline video processing only; `realtime.py` explicitly out of scope
- No camera hardware changes; works with standard 1080p fixed-lens webcam
- GPU acceleration relies on TensorFlow auto-detection; no CPU-only fallback toggle
- Recognition confidence degrades naturally at range; do not loosen `RECOGNITION_THRESHOLD` or `AMBIGUITY_MARGIN`
- Detection failures on individual tiles must not abort frame processing
- Detect faces at ~15m range (vs. current 2m limit)
- Handle multiple faces per frame robustly
- Maintain 100% backward compatibility with tracker, recognition, logging pipeline

---

## File Structure

**New files:**
- `detection_core.py` — Tiling, upscaling, RetinaFace detection, coordinate remapping, NMS
- `tests/test_detection_core.py` — Unit tests for tile operations, coordinate math, NMS
- `tests/test_detection_integration.py` — End-to-end pipeline tests

**Modified files:**
- `config.py` — Add tile parameters, face upscaling thresholds, NMS settings
- `face_core.py` — Add face upscaling before ArcFace embedding
- `video_processor.py` — Replace MediaPipe detector with tiled detector, adjust detection interval
- `requirements.txt` — Verify DeepFace dependency present

**Unchanged:**
- Tracker, recognition, logging pipelines remain unchanged
- Test infrastructure (`tests/conftest.py`) already in place

---

## Task 1: Config & Dependencies

**Files:**
- Modify: `config.py`
- Modify: `requirements.txt`
- Test: (manually verify imports work)

**Interfaces:**
- Produces: Config constants `TILE_GRID`, `TILE_OVERLAP`, `TILE_UPSCALE`, `TILED_DETECTION_INTERVAL`, `NMS_IOU_THRESHOLD`, `FACE_UPSCALE_THRESHOLD`, `FACE_UPSCALE_TARGET_SIZE`, `MIN_RECOGNITION_FACE_SIZE`, consumed by all later tasks

- [ ] **Step 1: Add tile and upscaling config to config.py**

After the existing face detection section, add:

```python
# ============================================================
# Long-range tiled detection (video_processor.py only)
# ============================================================

# Tile grid used to split a frame: (columns, rows).
TILE_GRID = (3, 2)

# Fractional overlap between adjacent tiles.
TILE_OVERLAP = 0.2

# Upscale factor applied to each tile before detection.
TILE_UPSCALE = 2.0

# Run tiled detection every N frames and rely on tracker continuity between.
TILED_DETECTION_INTERVAL = 5

# IoU threshold for de-duplicating overlapping tile detections.
NMS_IOU_THRESHOLD = 0.4
```

Also modify existing `MIN_RECOGNITION_FACE_SIZE` (should already be 15, verify it is).

- [ ] **Step 2: Verify DeepFace in requirements.txt**

Check that `deepface>=0.0.87` or similar is in `requirements.txt`. If not, add it:

```
deepface>=0.0.87
```

Run in terminal:
```powershell
& "d:\Git\FaceAndVoiceRecV2\3.12venv\Scripts\python.exe" -c "import deepface; print(deepface.__version__)"
```

Should print version without error. If import fails, run:
```powershell
& "d:\Git\FaceAndVoiceRecV2\3.12venv\Scripts\python.exe" -m pip install deepface
```

---

## Task 2: Create detection_core.py with Tiling & Tile Remapping

**Files:**
- Create: `detection_core.py`
- Test: Manual validation (will be unit tested in Task 4)

**Interfaces:**
- Produces: `tile_frame()`, `remap_tile_detections()` (called by Task 5)

- [ ] **Step 1: Create detection_core.py skeleton**

Create `d:\Git\FaceAndVoiceRecV2\detection_core.py`:

```python
"""
Long-range face detection via tiled RetinaFace.

This module replaces MediaPipe's close-range detector with a tiled approach:
1. Split frame into overlapping tiles
2. Upscale each tile to enlarge small faces
3. Run RetinaFace (via DeepFace) on each tile
4. Remap detections to full-frame coordinates
5. Apply NMS to remove duplicates
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Type alias: (x, y, width, height, confidence)
Detection = Tuple[float, float, float, float, float]


def tile_frame(
    frame: np.ndarray,
    grid: Tuple[int, int],
    overlap_ratio: float,
) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Split frame into overlapping tiles.
    
    Args:
        frame: Input frame (H×W×3 or H×W)
        grid: (cols, rows) tuple, e.g., (3, 2)
        overlap_ratio: Fractional overlap, e.g., 0.2 for 20%
        
    Yields:
        Tuple of (tile_image, (col_idx, row_idx))
        
    Example:
        >>> frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        >>> tiles = list(tile_frame(frame, (3, 2), 0.2))
        >>> len(tiles)
        6
    """
    h, w = frame.shape[:2]
    cols, rows = grid
    
    # Calculate tile dimensions
    tile_h = h // rows
    tile_w = w // cols
    
    # Calculate step size accounting for overlap
    step_h = int(tile_h * (1 - overlap_ratio))
    step_w = int(tile_w * (1 - overlap_ratio))
    
    tiles = []
    for row in range(rows):
        for col in range(cols):
            y_start = row * step_h
            x_start = col * step_w
            
            # Clamp to frame boundaries
            y_end = min(y_start + tile_h, h)
            x_end = min(x_start + tile_w, w)
            
            tile = frame[y_start:y_end, x_start:x_end].copy()
            tiles.append((tile, (col, row)))
    
    return tiles


def remap_tile_detections(
    detections: List[Detection],
    tile_idx: Tuple[int, int],
    frame_shape: Tuple[int, int],
    tile_grid: Tuple[int, int],
    overlap_ratio: float,
    upscale_factor: float,
) -> List[Detection]:
    """
    Convert detection boxes from upscaled tile space to full-frame space.
    
    Algorithm:
    1. Unscale box coordinates (divide by upscale_factor)
    2. Calculate tile offset accounting for overlap
    3. Apply offset and clamp to frame boundaries
    
    Args:
        detections: List of (x, y, w, h, confidence) in upscaled tile space
        tile_idx: (col, row) position in grid
        frame_shape: (height, width) of original frame
        tile_grid: (cols, rows)
        overlap_ratio: Fractional overlap used in tiling
        upscale_factor: Scale factor applied to tile
        
    Returns:
        Detections remapped to full-frame coordinates
        
    Example:
        >>> det = [(10, 20, 30, 40, 0.95)]
        >>> remapped = remap_tile_detections(det, (1, 0), (1080, 1920), (3, 2), 0.2, 2.0)
        >>> # remapped[0] is now in full-frame coordinates
    """
    h, w = frame_shape
    cols, rows = tile_grid
    
    # Calculate tile dimensions
    tile_h = h // rows
    tile_w = w // cols
    
    # Calculate step size
    step_h = int(tile_h * (1 - overlap_ratio))
    step_w = int(tile_w * (1 - overlap_ratio))
    
    col_idx, row_idx = tile_idx
    
    # Tile origin in full frame
    tile_x_start = col_idx * step_w
    tile_y_start = row_idx * step_h
    
    remapped = []
    for x, y, bw, bh, conf in detections:
        # Step 1: Unscale from upscaled tile space
        x_unscaled = x / upscale_factor
        y_unscaled = y / upscale_factor
        bw_unscaled = bw / upscale_factor
        bh_unscaled = bh / upscale_factor
        
        # Step 2: Offset to full frame
        x_frame = x_unscaled + tile_x_start
        y_frame = y_unscaled + tile_y_start
        
        # Step 3: Clamp to frame boundaries
        x_frame = max(0, min(x_frame, w - 1))
        y_frame = max(0, min(y_frame, h - 1))
        
        # Ensure box doesn't extend beyond frame
        bw_frame = min(bw_unscaled, w - x_frame)
        bh_frame = min(bh_unscaled, h - y_frame)
        
        # Filter invalid boxes
        if bw_frame > 0 and bh_frame > 0:
            remapped.append((x_frame, y_frame, bw_frame, bh_frame, conf))
    
    return remapped
```

- [ ] **Step 2: Add NMS function to detection_core.py**

Append to `detection_core.py`:

```python
def nms_merge(
    detections: List[Detection],
    iou_threshold: float = 0.4,
) -> List[Detection]:
    """
    Non-Maximum Suppression to remove duplicate detections.
    
    Strategy:
    1. Sort by confidence (descending)
    2. For each box, remove lower-confidence boxes with IoU > threshold
    
    Args:
        detections: List of (x, y, w, h, confidence)
        iou_threshold: Minimum IoU to consider boxes as duplicates
        
    Returns:
        Deduplicated list
    """
    if not detections:
        return []
    
    # Sort by confidence (descending)
    sorted_dets = sorted(detections, key=lambda d: d[4], reverse=True)
    
    keep = []
    remove = set()
    
    for i, det_i in enumerate(sorted_dets):
        if i in remove:
            continue
        
        keep.append(det_i)
        
        # Compare with all lower-confidence boxes
        for j in range(i + 1, len(sorted_dets)):
            if j in remove:
                continue
            
            det_j = sorted_dets[j]
            iou = _calculate_iou(det_i, det_j)
            
            if iou > iou_threshold:
                remove.add(j)
    
    return keep


def _calculate_iou(box1: Detection, box2: Detection) -> float:
    """
    Calculate Intersection over Union for two boxes.
    
    Args:
        box1, box2: (x, y, w, h, conf)
        
    Returns:
        IoU value (0.0 to 1.0)
    """
    x1, y1, w1, h1, _ = box1
    x2, y2, w2, h2, _ = box2
    
    # Convert to (x_min, y_min, x_max, y_max)
    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2
    
    # Intersection
    xi_min = max(x1, x2)
    yi_min = max(y1, y2)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    if xi_max <= xi_min or yi_max <= yi_min:
        return 0.0  # No intersection
    
    intersection = (xi_max - xi_min) * (yi_max - yi_min)
    
    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union
```

---

## Task 3: Add DeepFace Integration to detection_core.py

**Files:**
- Modify: `detection_core.py`

**Interfaces:**
- Produces: `detect_faces_tiled()` main function (called by video_processor.py in Task 5)

- [ ] **Step 1: Add DeepFace detector initialization**

Append to `detection_core.py`:

```python
def _get_retinaface_detector():
    """
    Lazy-load RetinaFace detector (DeepFace backend).
    
    Uses DeepFace's built-in RetinaFace via TensorFlow.
    """
    try:
        from deepface import DeepFace
        # DeepFace.extract_faces() uses RetinaFace by default
        # We'll use the lower-level detection later
        return "retinaface"
    except ImportError:
        logger.error("DeepFace not installed. Fallback to MediaPipe.")
        return None
```

- [ ] **Step 2: Add face detection function**

Append to `detection_core.py`:

```python
def _detect_faces_retinaface(tile: np.ndarray) -> List[Detection]:
    """
    Run RetinaFace detection on a tile (from DeepFace).
    
    Args:
        tile: Image array (H×W×3), uint8
        
    Returns:
        List of (x, y, w, h, confidence)
    """
    try:
        from deepface import DeepFace
        
        # DeepFace.extract_faces() returns detected regions
        # We use it to get bounding boxes + confidence
        try:
            faces = DeepFace.extract_faces(
                tile,
                detector_backend="retinaface",
                enforce_detection=False,  # Don't error on no faces
                expand_percentage=0,  # No padding
            )
        except Exception as e:
            logger.warning(f"RetinaFace detection failed on tile: {e}")
            return []
        
        detections = []
        for face in faces:
            # face is a dict with keys: 'facial_area', 'confidence'
            area = face.get("facial_area", {})
            x = area.get("x", 0)
            y = area.get("y", 0)
            w = area.get("w", 0)
            h = area.get("h", 0)
            confidence = face.get("confidence", 0.0)
            
            if w > 0 and h > 0:
                detections.append((float(x), float(y), float(w), float(h), float(confidence)))
        
        return detections
    
    except ImportError:
        logger.error("DeepFace not installed")
        return []
```

- [ ] **Step 3: Add upscaling function**

Append to `detection_core.py`:

```python
def upscale_tile(
    tile: np.ndarray,
    scale_factor: float,
) -> np.ndarray:
    """
    Upscale tile to improve small-face detection.
    
    Args:
        tile: Input tile (H×W×3 or H×W)
        scale_factor: Upscale factor, e.g., 2.0
        
    Returns:
        Upscaled tile
    """
    if scale_factor <= 1.0:
        return tile
    
    h, w = tile.shape[:2]
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    # Use cubic interpolation for quality
    upscaled = cv2.resize(tile, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return upscaled
```

- [ ] **Step 4: Add main tiled detection function**

Append to `detection_core.py`:

```python
def detect_faces_tiled(
    frame: np.ndarray,
    tile_grid: Tuple[int, int] = (3, 2),
    overlap_ratio: float = 0.2,
    upscale_factor: float = 2.0,
    nms_iou_threshold: float = 0.4,
) -> List[Detection]:
    """
    Main function: Run tiled detection on frame.
    
    Args:
        frame: Input frame (H×W×3), uint8
        tile_grid: (cols, rows)
        overlap_ratio: Fractional overlap
        upscale_factor: Upscale before detection
        nms_iou_threshold: NMS deduplication threshold
        
    Returns:
        List of detections in full-frame coordinates: [(x, y, w, h, conf), ...]
    """
    if frame is None or frame.size == 0:
        logger.warning("Invalid frame, skipping tiled detection")
        return []
    
    frame_shape = frame.shape[:2]  # (height, width)
    
    # Step 1: Tile the frame
    tiles = tile_frame(frame, tile_grid, overlap_ratio)
    
    all_detections = []
    
    # Step 2: Detect in each tile
    for tile, tile_idx in tiles:
        try:
            # Upscale tile
            upscaled_tile = upscale_tile(tile, upscale_factor)
            
            # Detect in upscaled tile
            tile_dets = _detect_faces_retinaface(upscaled_tile)
            
            # Remap to full frame
            frame_dets = remap_tile_detections(
                tile_dets,
                tile_idx,
                frame_shape,
                tile_grid,
                overlap_ratio,
                upscale_factor,
            )
            
            all_detections.extend(frame_dets)
        
        except Exception as e:
            logger.warning(f"Error processing tile {tile_idx}: {e}")
            continue
    
    # Step 3: Apply NMS to deduplicate
    merged = nms_merge(all_detections, iou_threshold=nms_iou_threshold)
    
    logger.debug(f"Tiled detection: {len(all_detections)} before NMS, {len(merged)} after")
    
    return merged
```

---

## Task 4: Unit Tests for detection_core.py

**Files:**
- Create: `tests/test_detection_core.py`

**Interfaces:**
- Tests: All functions from Task 2-3

- [ ] **Step 1: Create test file with tile tests**

Create `d:\Git\FaceAndVoiceRecV2\tests\test_detection_core.py`:

```python
"""Unit tests for detection_core.py"""

import numpy as np
import pytest

from detection_core import (
    tile_frame,
    remap_tile_detections,
    nms_merge,
    _calculate_iou,
)


class TestTileFrame:
    def test_tile_frame_creates_correct_grid(self):
        """1920×1080 frame, 3×2 grid should produce 6 tiles."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        tiles = tile_frame(frame, (3, 2), 0.2)
        
        assert len(tiles) == 6
        
        # Each tile should be an array
        for tile, (col, row) in tiles:
            assert isinstance(tile, np.ndarray)
            assert col in range(3)
            assert row in range(2)
    
    def test_tile_dimensions_with_overlap(self):
        """Verify tiles have expected dimensions with 20% overlap."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        tiles = tile_frame(frame, (3, 2), 0.2)
        
        # 3 columns: base_w = 1920 / 3 = 640
        # 2 rows: base_h = 1080 / 2 = 540
        
        for tile, _ in tiles:
            h, w = tile.shape[:2]
            # Tiles at edges might be smaller, but interior should be ~640×540
            assert w <= 650  # Allow small variance
            assert h <= 550


class TestRemapTileDetections:
    def test_remap_single_center_detection(self):
        """Detection in center tile (1, 0) should offset correctly."""
        # Frame is 1920×1080, 3×2 grid
        # Tile (1, 0) starts at approximately (640, 0)
        
        detection = [(10, 20, 30, 40, 0.95)]
        remapped = remap_tile_detections(
            detection,
            (1, 0),  # col=1, row=0
            (1080, 1920),
            (3, 2),
            0.2,
            2.0,  # upscale factor
        )
        
        assert len(remapped) == 1
        x, y, w, h, conf = remapped[0]
        
        # x should be offset by tile position
        # x_unscaled = 10 / 2.0 = 5
        # x_frame = 5 + tile_start
        assert x > 0  # Should not be at origin
        assert y < 30  # y is still near top (row 0)
        assert w == 15  # 30 / 2.0
        assert h == 20  # 40 / 2.0
        assert conf == 0.95
    
    def test_remap_clamps_to_frame_boundary(self):
        """Detection near edge should be clamped to frame bounds."""
        detection = [(1900, 1050, 200, 200, 0.9)]
        remapped = remap_tile_detections(
            detection,
            (2, 1),  # bottom-right tile
            (1080, 1920),
            (3, 2),
            0.2,
            2.0,
        )
        
        assert len(remapped) == 1
        x, y, w, h, conf = remapped[0]
        
        # Coords should be clamped within frame
        assert x >= 0 and x < 1920
        assert y >= 0 and y < 1080
        assert (x + w) <= 1920
        assert (y + h) <= 1080


class TestNMS:
    def test_nms_removes_duplicates(self):
        """Two overlapping boxes: lower-confidence should be removed."""
        dets = [
            (100, 100, 50, 50, 0.95),  # Higher confidence
            (110, 110, 50, 50, 0.85),  # Lower confidence, overlaps
        ]
        
        merged = nms_merge(dets, iou_threshold=0.4)
        
        assert len(merged) == 1
        assert merged[0][4] == 0.95  # Keep higher confidence
    
    def test_nms_keeps_non_overlapping(self):
        """Non-overlapping boxes should both be kept."""
        dets = [
            (100, 100, 50, 50, 0.9),
            (200, 200, 50, 50, 0.85),
        ]
        
        merged = nms_merge(dets, iou_threshold=0.4)
        
        assert len(merged) == 2
    
    def test_nms_empty_input(self):
        """Empty input should return empty output."""
        merged = nms_merge([], iou_threshold=0.4)
        assert merged == []


class TestIOU:
    def test_iou_identical_boxes(self):
        """Identical boxes should have IoU = 1.0."""
        box1 = (100, 100, 50, 50, 0.9)
        box2 = (100, 100, 50, 50, 0.8)
        
        iou = _calculate_iou(box1, box2)
        assert abs(iou - 1.0) < 0.01
    
    def test_iou_non_overlapping(self):
        """Non-overlapping boxes should have IoU = 0.0."""
        box1 = (100, 100, 50, 50, 0.9)
        box2 = (200, 200, 50, 50, 0.8)
        
        iou = _calculate_iou(box1, box2)
        assert iou < 0.01
    
    def test_iou_partial_overlap(self):
        """Partially overlapping boxes should have 0 < IoU < 1."""
        box1 = (100, 100, 100, 100, 0.9)
        box2 = (150, 150, 100, 100, 0.8)
        
        iou = _calculate_iou(box1, box2)
        assert 0 < iou < 1
```

- [ ] **Step 2: Run tests to verify they pass**

```powershell
& "d:\Git\FaceAndVoiceRecV2\3.12venv\Scripts\python.exe" -m pytest tests/test_detection_core.py -v
```

All tests should **PASS**.

- [ ] **Step 3: Commit**

```powershell
git add detection_core.py tests/test_detection_core.py config.py
git commit -m "feat: add tiled face detection module with NMS and coordinate remapping"
```

---

## Task 5: Add Face Upscaling to face_core.py

**Files:**
- Modify: `face_core.py`

**Interfaces:**
- Consumes: `config.FACE_UPSCALE_THRESHOLD`, `config.FACE_UPSCALE_TARGET_SIZE`
- Produces: `preprocess_face_for_recognition()` function (called by video_processor.py)

- [ ] **Step 1: Add upscaling function to face_core.py**

At the end of `face_core.py`, add:

```python
def preprocess_face_for_recognition(
    face_crop: np.ndarray,
    target_size: int = config.FACE_UPSCALE_TARGET_SIZE,
    upscale_threshold: int = config.FACE_UPSCALE_THRESHOLD,
) -> np.ndarray:
    """
    Upscale small face crops before ArcFace embedding.
    
    Small faces detected at distance need upscaling to match ArcFace's
    expected input size (112×112). This improves embedding quality.
    
    Logic:
    - If max(height, width) >= upscale_threshold:
        Return crop as-is
    - Else:
        Calculate scale factor to reach target_size
        Upscale using cubic interpolation
        Return upscaled crop
    
    Args:
        face_crop: Extracted face crop (H×W×3), uint8
        target_size: Target minimum dimension (default 112 for ArcFace)
        upscale_threshold: Min size before upscaling (default 100)
        
    Returns:
        Preprocessed crop ready for ArcFace (H×W×3)
    """
    if face_crop is None or face_crop.size == 0:
        return face_crop
    
    h, w = face_crop.shape[:2]
    max_dim = max(h, w)
    
    if max_dim >= upscale_threshold:
        # Face is large enough, use as-is
        return face_crop
    
    # Calculate upscaling factor
    scale_factor = target_size / max_dim
    
    # Upscale while maintaining aspect ratio
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    import cv2
    upscaled = cv2.resize(
        face_crop,
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC,
    )
    
    return upscaled
```

- [ ] **Step 2: Test manually**

In a Python terminal:

```python
import numpy as np
import cv2
from face_core import preprocess_face_for_recognition

# Create a small test crop (50×60)
small_crop = np.random.randint(0, 256, (50, 60, 3), dtype=np.uint8)

# Should upscale
upscaled = preprocess_face_for_recognition(small_crop)
print(f"Original: {small_crop.shape}, Upscaled: {upscaled.shape}")
# Expected: upscaled should be ~112×? (maintaining aspect ratio)

# Create a large test crop (150×160)
large_crop = np.random.randint(0, 256, (150, 160, 3), dtype=np.uint8)

# Should NOT upscale (already large)
unchanged = preprocess_face_for_recognition(large_crop)
print(f"Original: {large_crop.shape}, Unchanged: {unchanged.shape}")
# Expected: should be same shape
assert np.array_equal(unchanged, large_crop)
```

- [ ] **Step 3: Commit**

```powershell
git add face_core.py
git commit -m "feat: add face upscaling for small/distant faces before ArcFace"
```

---

## Task 6: Modify video_processor.py to Use Tiled Detection

**Files:**
- Modify: `video_processor.py`

**Interfaces:**
- Consumes: `detect_faces_tiled()` from `detection_core.py`, `preprocess_face_for_recognition()` from `face_core.py`
- Produces: Updated detection pipeline (same output format, just using tiled detector instead of MediaPipe)

- [ ] **Step 1: Find current MediaPipe detector usage**

In `video_processor.py`, find the section where faces are detected. Look for something like:

```python
from mediapipe.tasks.python import vision
...
mp_detector = vision.FaceDetector.create_from_options(...)
...
faces = mp_detector.detect(mp_image)
```

Note the exact line numbers. Read the full detect loop if needed to understand how `faces` are used downstream.

- [ ] **Step 2: Replace MediaPipe detector with tiled detector**

At the top of `video_processor.py`, add imports:

```python
from detection_core import detect_faces_tiled
from face_core import preprocess_face_for_recognition
import config
```

Modify the detection loop (pseudocode; adapt to exact code):

**Before:**
```python
for frame_idx, frame in enumerate(frames):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = mp_detector.detect(mp_image)
    faces = [(face.bounding_box.origin_x, ...) for face in detection_result.detections]
```

**After:**
```python
for frame_idx, frame in enumerate(frames):
    # Run tiled detection every N frames
    if frame_idx % config.TILED_DETECTION_INTERVAL == 0:
        faces_raw = detect_faces_tiled(
            frame,
            tile_grid=config.TILE_GRID,
            overlap_ratio=config.TILE_OVERLAP,
            upscale_factor=config.TILE_UPSCALE,
            nms_iou_threshold=config.NMS_IOU_THRESHOLD,
        )
        # Convert to tracker-compatible format: (x, y, w, h)
        current_faces = [(x, y, w, h) for x, y, w, h, _ in faces_raw]
    else:
        # Between detection frames, rely on tracker prediction
        current_faces = [tracker.predict_next_box(tid) for tid in active_track_ids]
    
    # Rest of pipeline: track, recognize, log
    # (unchanged)
```

- [ ] **Step 3: Update recognition pipeline to use upscaling**

In the recognition loop (where `extract_face_crop()` and `arcface.embed()` are called), modify:

**Before:**
```python
face_crop = extract_face_crop(frame, box, padding=10)
embedding = arcface.embed(face_crop)
```

**After:**
```python
face_crop = extract_face_crop(frame, box, padding=10)
face_crop = preprocess_face_for_recognition(face_crop)  # Upscale if needed
embedding = arcface.embed(face_crop)
```

- [ ] **Step 4: Test the modified pipeline**

Run `video_processor.py` on a test video:

```powershell
& "d:\Git\FaceAndVoiceRecV2\3.12venv\Scripts\python.exe" video_processor.py --video tests/sample_video.mp4
```

Should complete without errors. Check logs for "Tiled detection" debug messages and verify face count is reasonable.

- [ ] **Step 5: Commit**

```powershell
git add video_processor.py
git commit -m "feat: integrate tiled detection into video_processor pipeline"
```

---

## Task 7: Integration Test

**Files:**
- Create: `tests/test_detection_integration.py`

**Interfaces:**
- Tests: Full `video_processor.py` pipeline with tiled detection

- [ ] **Step 1: Create integration test file**

Create `d:\Git\FaceAndVoiceRecV2\tests\test_detection_integration.py`:

```python
"""Integration tests for tiled face detection pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from detection_core import detect_faces_tiled


class TestTiledDetectionIntegration:
    def test_tiled_detection_returns_valid_boxes(self):
        """Tiled detection should return boxes in full-frame coordinates."""
        # Create a synthetic frame with visible faces (high contrast)
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 200  # Light background
        
        # Draw some white circles to simulate faces (simple)
        import cv2
        cv2.circle(frame, (200, 200), 50, (255, 255, 255), -1)  # Face at (150, 150)
        cv2.circle(frame, (1500, 500), 30, (255, 255, 255), -1)  # Small face at (1470, 470)
        
        # Run tiled detection
        detections = detect_faces_tiled(
            frame,
            tile_grid=config.TILE_GRID,
            overlap_ratio=config.TILE_OVERLAP,
            upscale_factor=config.TILE_UPSCALE,
            nms_iou_threshold=config.NMS_IOU_THRESHOLD,
        )
        
        # Verify valid output format
        for det in detections:
            x, y, w, h, conf = det
            # Should be in frame bounds
            assert 0 <= x < 1920
            assert 0 <= y < 1080
            assert w > 0 and h > 0
            assert 0 < conf <= 1.0
    
    def test_tiled_detection_handles_empty_frame(self):
        """Detection on frame with no faces should not crash."""
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 50  # Dark, no faces
        
        detections = detect_faces_tiled(frame)
        
        # Should return empty or few detections
        assert isinstance(detections, list)
        # May have false positives, but shouldn't crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: Run integration tests**

```powershell
& "d:\Git\FaceAndVoiceRecV2\3.12venv\Scripts\python.exe" -m pytest tests/test_detection_integration.py -v
```

Tests should **PASS** or indicate expected behavior (e.g., small number of false positives on synthetic frame).

- [ ] **Step 3: Run full pipeline test (manual)**

Create a small test video (or use existing test video if available):

```powershell
& "d:\Git\FaceAndVoiceRecV2\3.12venv\Scripts\python.exe" video_processor.py --video tests/sample_meeting.mp4 --output tracked_output.mp4
```

Verify:
- Script completes without error
- Output video shows detected faces
- Log shows tiled detection running every 5 frames
- Face tracking is maintained across frames

- [ ] **Step 4: Commit**

```powershell
git add tests/test_detection_integration.py
git commit -m "test: add integration tests for tiled detection pipeline"
```

---

## Task 8: Verify All Tests Pass

**Files:**
- All (read-only verification)

**Interfaces:**
- N/A (verification task)

- [ ] **Step 1: Run full test suite**

```powershell
& "d:\Git\FaceAndVoiceRecV2\3.12venv\Scripts\python.exe" -m pytest tests/ -v
```

Expected output:
```
tests/test_config.py ...
tests/test_detection_core.py ........ PASSED
tests/test_detection_integration.py .. PASSED
===== X passed in Y.XXs =====
```

All tests must **PASS**. If any fail, debug and fix before proceeding.

- [ ] **Step 2: Commit final state**

```powershell
git add -A
git commit -m "test: all long-range detection tests passing"
```

---

## Checklist for Completion

- [ ] `detection_core.py` created with tiling, upscaling, RetinaFace, remapping, NMS
- [ ] `config.py` updated with tile parameters
- [ ] `face_core.py` updated with face upscaling function
- [ ] `video_processor.py` integrated with `detect_faces_tiled()`
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Manual testing on sample video confirms detection at range
- [ ] All code committed to git

---

## Performance Notes

- Tiled detection runs every 5 frames (~150ms latency at 30fps)
- Per-frame cost: ~375ms (6 tile passes × 60ms each)
- Memory: ~50MB peak (6 upscaled tiles in memory)
- GPU acceleration via TensorFlow auto-detection

If performance is unacceptable, consider:
- Increasing `TILED_DETECTION_INTERVAL` to 10 (every 10 frames)
- Reducing `TILE_GRID` to (2, 2) (fewer tiles, faster)
- Reducing `TILE_UPSCALE` to 1.5 (less upscaling, faster detection but lower accuracy at range)
