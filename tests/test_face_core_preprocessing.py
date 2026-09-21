import numpy as np

import face_core
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


def test_load_embedding_file_handles_legacy_object_array(tmp_path):
    path = tmp_path / "legacy_object_embedding.npy"
    legacy = np.array([np.array([1.0, 2.0, 3.0], dtype=np.float32)], dtype=object)
    np.save(path, legacy)

    loaded = face_core._load_embedding_file(path)

    assert loaded.dtype == np.float32
    assert loaded.shape == (3,)
    np.testing.assert_allclose(loaded, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_normalize_rejects_non_finite_embedding():
    assert face_core._normalize(np.array([np.nan, 1.0])) is None