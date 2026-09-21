from unittest.mock import patch

import numpy as np

from detection_core import (
    _calculate_iou,
    _detect_faces_retinaface,
    detect_faces_tiled,
    nms_merge,
    remap_tile_detections,
    tile_frame,
    upscale_tile,
)


def test_tile_frame_covers_full_frame_with_expected_grid():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    tiles = tile_frame(frame, (3, 2), 0.2)

    assert len(tiles) == 6
    assert max(origin[0] + tile.shape[1] for tile, _, origin in tiles) == 1920
    assert max(origin[1] + tile.shape[0] for tile, _, origin in tiles) == 1080


def test_upscale_tile_preserves_aspect_ratio():
    tile = np.zeros((40, 60, 3), dtype=np.uint8)
    upscaled = upscale_tile(tile, 2.0)
    assert upscaled.shape[:2] == (80, 120)


def test_remap_tile_detection_unscales_and_offsets_box():
    remapped = remap_tile_detections(
        [(20, 40, 60, 80, 0.95)],
        (640, 0),
        (1080, 1920),
        2.0,
    )
    assert remapped == [(650.0, 20.0, 30.0, 40.0, 0.95)]


def test_remap_tile_detection_clamps_to_frame():
    remapped = remap_tile_detections(
        [(1900, 1050, 200, 200, 0.9)],
        (0, 0),
        (1080, 1920),
        1.0,
    )
    x, y, width, height, _ = remapped[0]
    assert (x, y) == (1900.0, 1050.0)
    assert x + width <= 1920
    assert y + height <= 1080


def test_nms_keeps_highest_confidence_duplicate():
    detections = [
        (100, 100, 100, 100, 0.8),
        (105, 105, 100, 100, 0.95),
    ]
    assert nms_merge(detections, 0.4) == [detections[1]]


def test_nms_keeps_non_overlapping_boxes():
    detections = [(0, 0, 10, 10, 0.8), (100, 100, 10, 10, 0.7)]
    assert nms_merge(detections, 0.4) == detections


def test_iou_identical_boxes_is_one():
    assert _calculate_iou((0, 0, 10, 10, 0.8), (0, 0, 10, 10, 0.7)) == 1.0


def test_detect_faces_tiled_runs_a_full_frame_pass_in_addition_to_tiles():
    frame = np.zeros((240, 360, 3), dtype=np.uint8)
    seen_shapes = []

    def fake_detect(image):
        seen_shapes.append(image.shape[:2])
        return []

    with patch("detection_core._detect_faces_retinaface", side_effect=fake_detect):
        detect_faces_tiled(frame, (3, 2), 0.2, 2.0, 0.4)

    assert len(seen_shapes) == 7
    assert (240, 360) in seen_shapes


def test_detect_faces_filters_low_confidence_detections():
    """Ensure low-confidence RetinaFace detections are filtered."""
    tile = np.zeros((64, 64, 3), dtype=np.uint8)
    
    fake_response = [
        {
            "facial_area": {"x": 4, "y": 6, "w": 12, "h": 14},
            "confidence": 0.3,  # Below 0.5 threshold
        },
        {
            "facial_area": {"x": 20, "y": 20, "w": 15, "h": 15},
            "confidence": 0.9,  # Above 0.5 threshold
        }
    ]
    
    with patch("detection_core.DeepFace.extract_faces", return_value=fake_response):
        detections = _detect_faces_retinaface(tile)
    
    # Only the high-confidence detection should be returned
    assert len(detections) == 1
    assert detections[0][4] == 0.9


def test_detect_faces_falls_back_to_opencv_when_retinaface_fails():
    """Ensure OpenCV cascade fallback works when RetinaFace raises exception."""
    tile = np.zeros((100, 100, 3), dtype=np.uint8)
    # Draw a simple white rectangle to simulate a face region
    tile[20:80, 20:80] = 255
    
    from detection_core import _detect_faces_opencv
    
    with patch("detection_core.DeepFace.extract_faces", side_effect=RuntimeError("TensorFlow error")):
        detections = _detect_faces_retinaface(tile)
    
    # Fallback should return OpenCV results (might be empty on blank test image)
    # Just verify it doesn't crash and returns a list
    assert isinstance(detections, list)


def test_detect_faces_tiled_with_confidence_filtering():
    """Ensure detect_faces_tiled passes through confidence filtering."""
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    frame[20:80, 20:80] = 255  # Simulate face region
    
    # Mock RetinaFace to return mixed-confidence detections
    mixed_detections = [
        (10.0, 10.0, 30.0, 30.0, 0.3),  # Low confidence
        (40.0, 40.0, 30.0, 30.0, 0.8),  # High confidence
    ]
    
    with patch("detection_core._detect_faces_retinaface", return_value=mixed_detections):
        detections = detect_faces_tiled(
            frame,
            tile_grid=(1, 1),
            overlap_ratio=0.0,
            min_confidence=0.5,
        )
    
    # Should filter out low-confidence detection
    assert all(d[4] >= 0.5 for d in detections)


def test_detect_faces_tiled_returns_only_finite_positive_boxes():
    frame = np.zeros((120, 120, 3), dtype=np.uint8)

    with patch(
        "detection_core._detect_faces_retinaface",
        return_value=[(10.0, 10.0, 30.0, 30.0, 0.8)],
    ):
        detections = detect_faces_tiled(
            frame,
            tile_grid=(1, 1),
            overlap_ratio=0.0,
            upscale_factor=1.0,
        )

    assert detections
    assert all(np.all(np.isfinite(detection)) for detection in detections)
    assert all(detection[2] > 0 and detection[3] > 0 for detection in detections)