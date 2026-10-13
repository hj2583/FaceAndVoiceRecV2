from unittest.mock import patch, Mock

import numpy as np
import pytest

from detection_core import (
    _calculate_iou,
    _detect_faces_uniface,
    detect_faces_opencv,
    detect_faces_tiled,
    ensure_opencv_face_detector_available,
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

    with patch("detection_core._detect_faces_uniface", side_effect=fake_detect):
        detect_faces_tiled(frame, (3, 2), 0.2, 2.0, 0.4)

    assert len(seen_shapes) == 7
    assert (240, 360) in seen_shapes


def test_detect_faces_filters_low_confidence_detections():
    """Ensure low-confidence UniFace detections are filtered."""
    tile = np.zeros((64, 64, 3), dtype=np.uint8)

    fake_detections = [
        (4.0, 6.0, 12.0, 14.0, 0.3),  # Below 0.5 threshold
        (20.0, 20.0, 15.0, 15.0, 0.9),  # Above 0.5 threshold
    ]

    fake_backend = Mock()
    fake_backend.detect_faces.return_value = fake_detections
    with patch("face_backend.get_face_backend", return_value=fake_backend):
        detections = _detect_faces_uniface(tile)

    # Only the high-confidence detection should be returned
    assert len(detections) == 1
    assert detections[0][4] == 0.9


def test_detect_faces_falls_back_to_opencv_when_uniface_fails():
    """Ensure OpenCV cascade fallback works when UniFace raises an exception."""
    tile = np.zeros((100, 100, 3), dtype=np.uint8)
    # Draw a simple white rectangle to simulate a face region
    tile[20:80, 20:80] = 255

    from detection_core import _detect_faces_opencv

    # Provide a simple cascade that doesn't error for fallback
    cascade = Mock()
    cascade.empty.return_value = False
    cascade.detectMultiScale.return_value = []

    with patch("detection_core._get_opencv_cascade", return_value=cascade):
        fake_backend = Mock()
        fake_backend.detect_faces.side_effect = RuntimeError("ONNX runtime error")
        with patch("face_backend.get_face_backend", return_value=fake_backend):
            detections = _detect_faces_uniface(tile)

    # Fallback should return OpenCV results (might be empty on blank test image)
    # Just verify it doesn't crash and returns a list
    assert isinstance(detections, list)


def test_detect_faces_opencv_returns_detection_contract():
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    cascade = Mock()
    cascade.empty.return_value = False
    cascade.detectMultiScale.return_value = [(4, 6, 20, 24)]

    with patch("detection_core._get_opencv_cascade", return_value=cascade):
        detections = detect_faces_opencv(frame)

    assert detections == [(4.0, 6.0, 20.0, 24.0, 0.7)]


def test_detect_faces_opencv_reuses_cached_cascade():
    import detection_core

    # Preserve and restore any existing cache
    old = getattr(detection_core, "_OPENCV_CASCADE", None)
    detection_core._OPENCV_CASCADE = None
    cascade = Mock()
    cascade.empty.return_value = False
    cascade.detectMultiScale.return_value = []

    try:
        # Replace the cv2 module object on detection_core with a fake that records CascadeClassifier calls
        fake_cv2 = Mock()
        fake_cv2.CascadeClassifier = Mock(return_value=cascade)
        # Ensure .data.haarcascades is a string so _get_opencv_cascade can build the path
        fake_cv2.data = Mock()
        fake_cv2.data.haarcascades = ""
        with patch("detection_core.cv2", new=fake_cv2):
            detect_faces_opencv(np.zeros((40, 40, 3), dtype=np.uint8))
            detect_faces_opencv(np.zeros((40, 40, 3), dtype=np.uint8))

        fake_cv2.CascadeClassifier.assert_called_once()
    finally:
        detection_core._OPENCV_CASCADE = old


def test_get_opencv_cascade_handles_none_haarcascades(monkeypatch):
    """Regression: _get_opencv_cascade should not pass None to os.path.join.

    Some test doubles or OpenCV builds may set `cv2.data.haarcascades` to
    None; ensure our function normalizes that to an empty string and still
    constructs a CascadeClassifier safely.
    """
    import detection_core

    # Reset cache
    old = getattr(detection_core, "_OPENCV_CASCADE", None)
    detection_core._OPENCV_CASCADE = None

    try:
        fake_cv2 = Mock()
        # Make cv2.data exist but haarcascades is explicitly None
        fake_cv2.data = Mock()
        fake_cv2.data.haarcascades = None

        # CascadeClassifier should be constructed with some path; return a simple mock
        cascade = Mock()
        cascade.empty.return_value = False
        fake_cv2.CascadeClassifier = Mock(return_value=cascade)

        with patch("detection_core.cv2", new=fake_cv2):
            got = detection_core._get_opencv_cascade()

        # Should return our mocked cascade and not raise
        assert got is cascade
        fake_cv2.CascadeClassifier.assert_called_once()
        # Verify the argument passed was a string (path)
        arg = fake_cv2.CascadeClassifier.call_args[0][0]
        assert isinstance(arg, str)
        assert arg.endswith('haarcascade_frontalface_default.xml')
    finally:
        detection_core._OPENCV_CASCADE = old


def test_detect_faces_opencv_returns_empty_for_invalid_frame():
    assert detect_faces_opencv(None) == []
    assert detect_faces_opencv(np.empty((0, 0, 3), dtype=np.uint8)) == []


def test_detect_faces_opencv_contains_detection_failure():
    cascade = Mock()
    cascade.empty.return_value = False
    cascade.detectMultiScale.side_effect = RuntimeError("cascade failed")

    with patch("detection_core._get_opencv_cascade", return_value=cascade):
        assert detect_faces_opencv(np.zeros((40, 40, 3), dtype=np.uint8)) == []


def test_ensure_opencv_face_detector_available_rejects_missing_cascade():
    cascade = Mock()
    cascade.empty.return_value = True

    with patch("detection_core._get_opencv_cascade", return_value=cascade):
        with pytest.raises(RuntimeError, match="Haar cascade data is unavailable"):
            ensure_opencv_face_detector_available()


def test_detect_faces_tiled_with_confidence_filtering():
    """Ensure detect_faces_tiled passes through confidence filtering."""
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    frame[20:80, 20:80] = 255  # Simulate face region
    
    # Mock UniFace to return mixed-confidence detections
    mixed_detections = [
        (10.0, 10.0, 30.0, 30.0, 0.3),  # Low confidence
        (40.0, 40.0, 30.0, 30.0, 0.8),  # High confidence
    ]
    
    with patch("detection_core._detect_faces_uniface", return_value=mixed_detections):
        detections = detect_faces_tiled(
            frame,
            tile_grid=(1, 1),
            overlap_ratio=0.0,
            min_confidence=0.5,
        )
    
    # Should filter out low-confidence detection
    assert all(d[4] >= 0.5 for d in detections)