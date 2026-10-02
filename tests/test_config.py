import config
from pathlib import Path


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


def test_detection_face_size_is_lower_than_recognition_gate():
    assert config.MIN_FACE_SIZE < config.MIN_RECOGNITION_FACE_SIZE


def test_face_upscale_target_size_matches_arcface_input():
    assert config.FACE_UPSCALE_TARGET_SIZE == 112


def test_opencv_dependency_excludes_wheels_without_haar_data():
    requirements = (
        Path(__file__).resolve().parent.parent / "requirements.txt"
    ).read_text(encoding="utf-8")

    assert "opencv-contrib-python>=4.10,<4.12" in requirements
    assert "opencv-python>=4.10,<4.12" not in requirements


def test_realtime_detection_defaults_prioritize_responsiveness():
    assert config.REALTIME_DETECTION_INTERVAL == 3
    assert config.REALTIME_HAAR_SCALE_FACTOR > 1.0
    assert config.REALTIME_HAAR_MIN_NEIGHBORS >= 1
    assert config.REALTIME_FPS_WINDOW_SECONDS == 1.0


def test_speaker_attribution_weights_are_configurable_and_positive():
    assert config.SPEAKER_WINDOW_MS > 0
    assert config.SPEAKER_SWITCH_THRESHOLD > 0
    assert 0.0 < config.SPEAKER_MIN_CONFIDENCE < 1.0
    assert config.SPEAKER_GRACE_PERIOD_MS > 0
    assert config.AUDIO_SYNC_TOLERANCE_MS > 0
    assert config.LIP_MOTION_NORM > 0
    assert config.SPEAKER_HISTORY_MAXLEN > 0
    weights = (
        config.VOICE_ACTIVITY_WEIGHT,
        config.LIP_MOTION_WEIGHT,
        config.FACE_CONFIDENCE_WEIGHT,
        config.TEMPORAL_WEIGHT,
    )
    assert all(w >= 0.0 for w in weights)
    assert abs(sum(weights) - 1.0) < 1e-6