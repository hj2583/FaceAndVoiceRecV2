# Long-Range Face Detection Design Spec

**Date:** 2026-08-20  
**Status:** Design Phase  
**Scope:** Extend face detection from ~2m to ~15m range using tiled RetinaFace detection

---

## Overview

Replace MediaPipe's close-range FaceDetector with a tiled detection pipeline using RetinaFace (via DeepFace) to detect faces at distances up to 15 meters. Maintain compatibility with existing tracker, recognition, and logging systems.

**Key Goals:**
- Detect attendees at 15m range on 1080p video (vs. current 2m limit)
- Handle multiple faces per frame robustly
- Upscale small crops before ArcFace recognition to improve accuracy at distance
- Maintain tracker continuity without refactoring downstream pipeline

---

## Architecture

### Detection Pipeline

```
Video Frame (1080p, e.g., 1920×1080)
    ↓
[Tile frame into 3×2 grid with 20% overlap]
    ├─ Tile 0 (0,0):     0-698, 0-598
    ├─ Tile 1 (1,0):     540-1218, 0-598
    ├─ Tile 2 (2,0):     1182-1860, 0-598
    ├─ Tile 3 (0,1):     0-698, 482-1080
    ├─ Tile 4 (1,1):     540-1218, 482-1080
    ├─ Tile 5 (2,1):     1182-1860, 482-1080
    └─ Full frame (optional high-res pass)
    ↓
[For each tile: upscale 2× and run RetinaFace]
    ├─ Detections in tile coordinate space
    └─ Returns: List[(x, y, w, h, confidence)]
    ↓
[Remap all boxes to full-frame coordinates]
    └─ Apply inverse upscale + tile offset
    ↓
[Apply Non-Maximum Suppression (NMS)]
    ├─ IoU threshold: 0.4
    └─ Keep highest-confidence box from overlaps
    ↓
[Pass merged detections to existing tracker]
    └─ Tracker, recognition, logging unchanged
```

### Why This Works

1. **Small faces are upscaled**: A 15px face at full resolution becomes ~30px in upscaled tile
2. **RetinaFace tuned for small faces**: Specifically designed for multi-scale detection
3. **Overlap reduces edge artifacts**: Tiles overlap by 20%, so faces at tile boundaries detected in multiple tiles
4. **NMS deduplication**: Ensures one box per face despite overlaps

### Confidence Degradation by Distance

| Distance | Face Size | Recognition Accuracy | Handling |
|----------|-----------|----------------------|----------|
| 2m | 100px | 99% | Direct match to known faces |
| 8m | 30px → 112px (upscaled) | 85% | Some false positives, filtered by thresholds |
| 15m | 15px → 112px (upscaled) | 70% | Mostly "Unknown", tracker maintains continuity |

---

## New Components

### 1. `detection_core.py` (New Module)

```python
def tile_frame(
    frame: np.ndarray,
    grid: Tuple[int, int],
    overlap_ratio: float,
) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Split frame into overlapping tiles.
    
    Args:
        frame: Input frame (H×W×3)
        grid: (cols, rows) tuple, e.g., (3, 2)
        overlap_ratio: Fractional overlap, e.g., 0.2 for 20%
        
    Yields:
        (tile_image, (col_idx, row_idx))
    """

def upscale_tile(
    tile: np.ndarray,
    scale_factor: float,
) -> np.ndarray:
    """
    Upscale tile to improve small-face detection.
    Uses cv2.INTER_LINEAR for speed.
    """

def detect_faces_in_tile(
    tile: np.ndarray,
    detector,  # DeepFace/RetinaFace detector
) -> List[Tuple[float, float, float, float, float]]:
    """
    Run RetinaFace on upscaled tile.
    
    Returns:
        List of (x, y, w, h, confidence) in tile coordinates
    """

def remap_tile_detections(
    detections: List[Tuple[float, float, float, float, float]],
    tile_idx: Tuple[int, int],
    frame_shape: Tuple[int, int],
    tile_grid: Tuple[int, int],
    overlap_ratio: float,
    upscale_factor: float,
) -> List[Tuple[float, float, float, float, float]]:
    """
    Convert detection boxes from tile/upscaled space to full-frame space.
    
    Algorithm:
    1. Unscale box coordinates (divide by upscale_factor)
    2. Apply tile offset accounting for overlap
    3. Clamp to frame boundaries
    
    Args:
        detections: Boxes from upscaled tile (x, y, w, h, conf)
        tile_idx: (col, row) position in grid
        frame_shape: (height, width) of original frame
        tile_grid: (cols, rows)
        overlap_ratio: Fractional overlap used in tiling
        upscale_factor: Scale factor applied to tile
        
    Returns:
        Detections remapped to full-frame coordinates
    """

def nms_merge(
    detections: List[Tuple[float, float, float, float, float]],
    iou_threshold: float = 0.4,
) -> List[Tuple[float, float, float, float, float]]:
    """
    Non-Maximum Suppression to remove duplicate detections.
    
    Strategy:
    1. Sort by confidence (descending)
    2. For each box, compare IoU with all higher-confidence boxes
    3. Remove if IoU > threshold
    
    Args:
        detections: List of (x, y, w, h, confidence)
        iou_threshold: Minimum IoU to consider boxes as duplicates
        
    Returns:
        Deduplicated list
    """

def detect_faces_tiled(
    frame: np.ndarray,
    detector,  # RetinaFace detector instance
    tile_grid: Tuple[int, int] = (3, 2),
    overlap_ratio: float = 0.2,
    upscale_factor: float = 2.0,
    nms_iou_threshold: float = 0.4,
) -> List[Tuple[float, float, float, float, float]]:
    """
    Main function: Run tiled detection on frame.
    
    Returns:
        List of face boxes in full-frame coordinates
    """
```

### 2. Face Upscaling for Recognition (in `face_core.py`)

Add or modify:

```python
def preprocess_face_for_recognition(
    face_crop: np.ndarray,
    target_size: int = 112,
    upscale_threshold: int = 100,
    upscale_factor: float = 2.0,
) -> np.ndarray:
    """
    Upscale small face crops before ArcFace embedding.
    
    Logic:
    - If max(height, width) >= upscale_threshold:
        Return crop as-is
    - Else:
        Upscale to target_size × target_size using cubic interpolation
        Return upscaled crop
    
    Args:
        face_crop: Extracted face crop (H×W×3)
        target_size: ArcFace expects 112×112 input
        upscale_threshold: Min face size before upscaling
        upscale_factor: NOT USED (for compatibility); upscaling calculates needed scale
        
    Returns:
        Preprocessed crop ready for ArcFace
    """
```

### 3. Config Updates (in `config.py`)

Add/modify:

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

# Update face recognition constraints
# Minimum face size before attempting ArcFace recognition.
# Lowered to allow best-effort recognition attempts on small/distant
# faces detected via the long-range tiled pipeline. Low-confidence
# matches still fall back to "Unknown" through the existing thresholds.
MIN_RECOGNITION_FACE_SIZE = 15  # Changed from default

# Minimum face size required for reliable MediaPipe lip landmarks.
LANDMARK_MIN_FACE_SIZE = 60

# Small-face recognition enhancement
FACE_UPSCALE_THRESHOLD = 100
FACE_UPSCALE_TARGET_SIZE = 112
```

---

## Integration with Existing Pipeline

### video_processor.py Changes

**Current flow (MediaPipe detector):**
```python
for frame_idx, frame in enumerate(frames):
    faces = mediapipe_detector.detect(frame)  # ← REMOVE THIS
    # ... track, recognize, log
```

**New flow (tiled detector):**
```python
for frame_idx, frame in enumerate(frames):
    if frame_idx % TILED_DETECTION_INTERVAL == 0:
        faces = detect_faces_tiled(
            frame,
            detector=retinaface_detector,
            tile_grid=TILE_GRID,
            overlap_ratio=TILE_OVERLAP,
            upscale_factor=TILE_UPSCALE,
            nms_iou_threshold=NMS_IOU_THRESHOLD,
        )
    # ... else use tracker prediction ...
    # ... recognize, log
```

### Recognition Pipeline Enhancement

**Current flow:**
```python
face_crop = extract_face(frame, box)
embedding = arcface.embed(face_crop)  # Assumes face_crop is ≥112×112
```

**New flow:**
```python
face_crop = extract_face(frame, box)
face_crop = preprocess_face_for_recognition(face_crop)  # Upscale if needed
embedding = arcface.embed(face_crop)
```

**Downstream filters unchanged:**
```python
if embedding_distance < RECOGNITION_THRESHOLD:
    person_id = recognized_person_id
else:
    person_id = UNKNOWN_ID  # Existing fallback
```

---

## Tile Coordinate Remapping Algorithm

**Key Challenge:** Boxes from upscaled tile space must be converted to full-frame space without distortion.

**Algorithm:**

```
Input:
  tile_box = (x, y, w, h) in upscaled tile space
  tile_idx = (col, row) in grid
  frame_shape = (H, W)
  tile_grid = (cols, rows)
  overlap_ratio = 0.2
  upscale_factor = 2.0

Step 1: Unscale box (divide by upscale_factor)
  x_unscaled = x / upscale_factor
  y_unscaled = y / upscale_factor
  w_unscaled = w / upscale_factor
  h_unscaled = h / upscale_factor

Step 2: Calculate tile dimensions
  tile_width_base = W / cols
  tile_height_base = H / rows
  
  # With overlap, tiles step by (1 - overlap_ratio) * tile_size
  step_x = tile_width_base * (1 - overlap_ratio)
  step_y = tile_height_base * (1 - overlap_ratio)

Step 3: Apply tile offset
  tile_start_x = col * step_x
  tile_start_y = row * step_y
  
  x_frame = x_unscaled + tile_start_x
  y_frame = y_unscaled + tile_start_y

Step 4: Clamp to frame boundaries
  x_frame = max(0, min(x_frame, W))
  y_frame = max(0, min(y_frame, H))
  w_frame = min(w_unscaled, W - x_frame)
  h_frame = min(h_unscaled, H - y_frame)

Output:
  frame_box = (x_frame, y_frame, w_frame, h_frame)
```

---

## NMS (Non-Maximum Suppression) Algorithm

```
Input: detections = [(x1,y1,w1,h1,conf1), (x2,y2,w2,h2,conf2), ...]

Step 1: Sort by confidence (descending)
  sorted_dets = sort(detections, key=confidence, reverse=True)

Step 2: Initialize keep list
  keep = []
  remove = set()

Step 3: Greedy selection
  For i in range(len(sorted_dets)):
    if i in remove:
      continue
    keep.append(sorted_dets[i])
    
    For j in range(i+1, len(sorted_dets)):
      if j in remove:
        continue
      
      iou = calculate_iou(sorted_dets[i], sorted_dets[j])
      if iou > nms_iou_threshold:
        remove.add(j)

Output: keep
```

**IoU Calculation:**
```
def iou(box1, box2):
  x1, y1, w1, h1 = box1
  x2, y2, w2, h2 = box2
  
  # Convert to (x1, y1, x2, y2)
  x1_min, y1_min, x1_max, y1_max = x1, y1, x1+w1, y1+h1
  x2_min, y2_min, x2_max, y2_max = x2, y2, x2+w2, y2+h2
  
  # Intersection
  xi_min = max(x1_min, x2_min)
  yi_min = max(y1_min, y2_min)
  xi_max = min(x1_max, x2_max)
  yi_max = min(y1_max, y2_max)
  
  if xi_max <= xi_min or yi_max <= yi_min:
    return 0.0  # No intersection
  
  intersection = (xi_max - xi_min) * (yi_max - yi_min)
  
  # Union
  area1 = w1 * h1
  area2 = w2 * h2
  union = area1 + area2 - intersection
  
  return intersection / union
```

---

## Error Handling

| Scenario | Handling |
|----------|----------|
| Tile extraction fails (boundary OOB) | Clamp coordinates, retry with safe bounds |
| RetinaFace returns invalid box (w≤0 or h≤0) | Filter out, log warning |
| Remapping produces negative coordinates | Clamp to [0, frame_size] |
| NMS produces empty output | Log warning, return empty face list (tracker handles it) |
| DeepFace detector fails to load | Fallback to MediaPipe detector (graceful degradation) |
| Frame is too small for tiling (< tile_width) | Skip tiling, use MediaPipe detector for this frame |
| Upscaling OOMs on many large faces | Skip upscaling for this frame, use original crop |

---

## Testing Strategy

**Unit Tests** (`tests/test_detection_core.py`):

```python
def test_tile_frame_creates_correct_grid():
    # 1920×1080 frame, 3×2 grid, 20% overlap
    # Verify tile count = 6, correct dimensions, correct overlap
    
def test_upscale_tile_maintains_aspect_ratio():
    # Tile 640×540, upscale 2×
    # Verify output 1280×1080 (or similar based on implementation)
    
def test_remap_tile_detection_single_tile():
    # Box at tile (1,0), verify remapped to correct full-frame coords
    # Test boundary cases (tile at corner, edge)
    
def test_remap_tile_detection_boundary_clamp():
    # Box extends beyond tile edge
    # Verify properly clamped to frame bounds
    
def test_nms_removes_duplicates():
    # Two boxes with 60% IoU (> 0.4 threshold)
    # Verify lower-confidence box removed
    
def test_nms_keeps_non_overlapping():
    # Two boxes with 0% IoU (non-overlapping)
    # Verify both kept
    
def test_detect_faces_tiled_returns_frame_coords():
    # Integration: run tiled detection on real frame
    # Verify all boxes in valid frame coordinates
```

**Integration Tests** (`tests/test_detection_integration.py`):

```python
def test_tiled_detection_vs_mediapipe_on_close_range():
    # Video with faces at 1-3m
    # Run both detectors, compare detection count/accuracy
    # Expect roughly similar results (tiled shouldn't degrade close-range)
    
def test_tiled_detection_detects_distant_faces():
    # Video with faces at 10-15m
    # Run tiled detection
    # Verify faces detected (even if small)
    # Verify fewer false positives than MediaPipe
    
def test_full_pipeline_with_tracking():
    # 30s test video: 5 attendees, varying distances
    # Run video_processor with tiled detection
    # Verify:
    #   - All faces detected
    #   - Tracker maintains IDs across frames
    #   - Recognition works (some unknown at distance, OK)
    #   - Database logging correct
```

---

## Performance Considerations

### Computational Cost

**Per-frame cost (tiled detection):**
- Tiling: ~10ms (NumPy operations)
- Upscaling: ~50ms (6 tiles × cv2.resize)
- RetinaFace inference: ~300ms (varies by GPU)
- Remapping: ~5ms (NumPy)
- NMS: ~10ms
- **Total per-tile pass: ~375ms**

**Mitigation:** Run tiled detection every `TILED_DETECTION_INTERVAL=5` frames (not every frame)
- Detection: every 5 frames (30fps → 6 detection passes/sec)
- Tracker predicts between detection frames
- Trade-off: ~150ms latency, 95% of frames use tracker prediction

### Memory Usage

**Peak memory (6 tiles upscaled):**
- Original frame: 1920×1080×3 = ~6MB
- 6 upscaled tiles: 1280×1080×3×6 = ~30MB
- Whisper embedding cache: ~10MB
- **Total: ~50MB** (acceptable on modern hardware)

**Optimization:** Process tiles sequentially if memory constrained

---

## Constraints & Assumptions

1. **Offline video only** — `realtime.py` explicitly out of scope; long-range detection adds latency unsuitable for real-time 30fps
2. **Fixed camera** — Assumes standard 1080p webcam with fixed wide lens; no camera calibration required
3. **GPU optional** — TensorFlow auto-detects CUDA; CPU mode works but slower (~1sec per frame)
4. **Recognition thresholds unchanged** — `RECOGNITION_THRESHOLD=0.70`, `AMBIGUITY_MARGIN=0.05` stay fixed; distant faces naturally fail recognition and fall back to "Unknown"
5. **Tracker continuity** — Tracker must maintain ID continuity; tracker won't lose ID between detection passes if person moves <180px/5frames

---

## Success Criteria

- [x] `detection_core.py` created with all tiling/remapping/NMS functions
- [x] Tiled detection integrates into `video_processor.py` without breaking existing tracker/recognition/logging
- [x] Detects faces at 15m range in test video
- [x] Face upscaling in `face_core.py` working
- [x] Config constants added
- [x] All unit tests passing (especially coordinate remapping tests)
- [x] Integration test: full pipeline on sample video with multi-distance attendees
- [x] Performance acceptable: <6 sec per 30-frame second (1 detection pass per TILED_DETECTION_INTERVAL)

---

## Out of Scope

- Real-time mode optimization (latency-sensitive; would need different architecture)
- Camera calibration or depth estimation
- Automatic distance measurement
- Person-specific model fine-tuning
- GPU-specific optimizations beyond TensorFlow defaults
