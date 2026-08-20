# Long-Range Face Detection & Recognition (up to ~15m) — Design

## Problem

`realtime.py` and `video_processor.py` currently fail to detect faces beyond ~2m. Root causes:

1. **Detector tuned for close range.** `video_processor.py`'s per-frame detection uses MediaPipe's `FaceLandmarker`, whose bundled face detector is optimized for close-range/selfie-style use (matches the observed ~2m cutoff).
2. **Insufficient pixel density at distance.** With a standard 1080p fixed wide-lens webcam, a face at 15m occupies only ~15-30 pixels wide in the raw frame — below reliable detection size and far below what ArcFace needs for a stable embedding.

Camera hardware is fixed (standard webcam, max native 1080p, no PTZ/multi-camera). The meeting room has 15+ attendees, mostly seated in fixed positions. **Offline video processing (`video_processor.py`) is the priority**; `realtime.py` is out of scope for this change.

## Goals / Non-Goals

- **Goal:** Reliably *detect* (and best-effort recognize) attendees up to ~15m in offline video processing.
- **Goal:** Maximize detection recall — count as many attendees as possible.
- **Non-goal:** Guaranteed confident identity match at 15m. A ~20px face is inherently blurry for ArcFace; distant faces will fall back to "unknown" more often. This is an accepted trade-off.
- **Non-goal:** Changes to `realtime.py` or camera hardware.

## Architecture

Per-frame detection in `video_processor.py` changes from a single MediaPipe call to a tiled detection sub-pipeline. Everything downstream (centroid tracker, ArcFace recognition, active-speaker logic, DB logging) is unchanged.

```
Frame (1920x1080, full native resolution)
   ↓
Tile Splitter → overlapping tiles (upscaled) + one full-frame pass
   ↓
RetinaFace detection per tile (via DeepFace backend, GPU-accelerated if available)
   ↓
Coordinate remap to full-frame + NMS merge (dedupe across tiles)
   ↓
Merged face boxes for this frame
   ↓
(existing) Centroid Tracker → ArcFace recognition → Active speaker → Logging
```

## Detection Stage

- **Tiling:** frame split into a `TILE_GRID` (e.g. 3 columns × 2 rows) of overlapping tiles (`TILE_OVERLAP`, e.g. 20%) so faces near tile borders aren't cut off. Each tile upscaled by `TILE_UPSCALE` before detection to increase effective pixel density on distant faces.
- **Full-frame pass:** one additional detection pass on the full (non-tiled) frame to cheaply catch large/near faces without tiling overhead.
- **Detector:** `DeepFace.extract_faces(..., detector_backend="retinaface")` per tile — reuses existing DeepFace dependency, no new library. RetinaFace handles small/far faces far better than MediaPipe's BlazeFace-based detector.
- **Merging:** each tile's boxes remapped to full-frame coordinates (undo tile offset + upscale), then NMS (IoU threshold `NMS_IOU_THRESHOLD`, e.g. 0.4) across all tile + full-frame results to remove duplicate detections of the same face.
- **Performance control:** `TILED_DETECTION_INTERVAL` — run the full tiled pass every K frames, relying on tracker continuity in between (same pattern as existing `RECOGNITION_INTERVAL`). Tunable since offline processing has no hard real-time deadline.

## Recognition Stage Adjustments

- Extend `prepare_recognition_crop` to upscale (Lanczos) small crops up to ArcFace's expected ~112×112 input.
- Lower `MIN_RECOGNITION_FACE_SIZE` so distant small faces still get a recognition attempt instead of being filtered out before trying.
- `RECOGNITION_THRESHOLD` / `AMBIGUITY_MARGIN` unchanged — existing rejection logic naturally falls back to "unknown" for low-confidence distant faces, matching the accepted trade-off.
- `face_quality()` continues to score distant faces lower, so only the best-quality distant captures are retained for unknown-face review.

## Active Speaker / Lip-Landmark Handling

- Faces at or above a landmark-viability size threshold (`LANDMARK_MIN_FACE_SIZE`) keep running MediaPipe FaceLandmarker on the crop for lip-opening ratio, as today.
- Faces below that threshold skip lip-based speaker detection entirely for that frame (excluded from "active speaker" candidacy) rather than producing noisy landmark data.
- Active-speaker selection logic (max lip-open during a VAD speech segment) is otherwise unchanged.

## Config Changes (`config.py`)

- `TILE_GRID = (3, 2)`
- `TILE_OVERLAP = 0.2`
- `TILE_UPSCALE = 2.0`
- `TILED_DETECTION_INTERVAL`
- `NMS_IOU_THRESHOLD = 0.4`
- Lower `MIN_RECOGNITION_FACE_SIZE` (tune empirically, e.g. 30 → 15)
- New `LANDMARK_MIN_FACE_SIZE` (separate gate for lip-landmark viability)

## New Module

`detection_core.py`:
- `tile_frame()` — splits frame into overlapping tiles with coordinate metadata
- `detect_faces_retinaface(tile)` — wraps DeepFace RetinaFace call, catches per-tile detection errors (treated as "no face in tile")
- `merge_detections()` — remaps tile-local boxes to full-frame coords + NMS

`video_processor.py`: replace the direct MediaPipe-detection call with a call into `detection_core`, then feed results into the existing tracker/recognition/logging pipeline unchanged.

`realtime.py`: no changes — out of scope.

## GPU Acceleration

- DeepFace's RetinaFace backend runs on TensorFlow, which uses GPU automatically if a CUDA-compatible GPU + drivers + matching TensorFlow build are present — no code change needed for the detector call itself.
- Add a startup log (`tf.config.list_physical_devices('GPU')`) in `detection_core.py` to confirm at runtime whether GPU is being used.
- No CPU-only fallback toggle — auto-detect (GPU if available, else CPU) is sufficient per user preference. GPU is effectively required for acceptable performance given tiling multiplies detector calls per frame (6-7 tiles + 1 full-frame pass).

## Error Handling

- Per-tile RetinaFace detection wrapped in try/except; failures treated as "no face in this tile", not an aborted frame.
- Log a warning with elapsed time per frame if tiled detection is unexpectedly slow, to help tune `TILE_GRID` / `TILED_DETECTION_INTERVAL`.
- Existing recognition failure handling (temporary miss tolerance, quality thresholds) unchanged.

## Testing / Validation

- Manual validation using a recorded meeting video with attendees at marked distances (2m/5m/10m/15m) — verify detections at each distance in the output log/video.
- Compare detection counts before/after (baseline MediaPipe vs. new tiled RetinaFace) on the same test video.
- Spot-check per-frame processing time to confirm acceptable offline processing duration.
- Unit test for `merge_detections()` NMS logic using synthetic overlapping boxes (pure logic, easy to verify deterministically).
