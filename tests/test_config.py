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


def test_face_backend_defaults_prefer_uniface_with_deepface_fallback():
    assert config.FACE_BACKEND == "uniface"
    assert config.FACE_BACKEND_FALLBACK == "deepface"
    assert config.FACE_BACKEND_STRICT_COMPATIBILITY is True