import numpy as np

from face_core import face_quality, preprocess_face_for_recognition


def test_face_quality_scores_small_nonempty_crop_instead_of_zero():
    crop = np.zeros((30, 30, 3), dtype=np.uint8)
    crop[::2, :, :] = 255

    quality = face_quality(crop)

    assert quality > 0.0


def test_small_face_crop_is_upscaled_without_distortion():
    crop = np.zeros((40, 60, 3), dtype=np.uint8)
    result = preprocess_face_for_recognition(crop)
    assert result.shape[:2] == (75, 112)


def test_large_face_crop_is_left_unchanged():
    crop = np.zeros((120, 140, 3), dtype=np.uint8)
    result = preprocess_face_for_recognition(crop)
    assert result is crop