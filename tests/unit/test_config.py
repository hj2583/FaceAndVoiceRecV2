import importlib
from pathlib import Path

import pytest

from face_voice_rec import config
import face_voice_rec.config.paths as paths


@pytest.fixture(autouse=True)
def _reset_paths_module_state():
    importlib.reload(paths)
    yield
    importlib.reload(paths)


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
    requirements = (Path(__file__).resolve().parents[2] / "requirements.txt").read_text(
        encoding="utf-8"
    )

    assert "opencv-contrib-python>=4.10,<4.12" in requirements
    assert "opencv-python>=4.10,<4.12" not in requirements


def test_realtime_detection_defaults_prioritize_responsiveness():
    assert config.REALTIME_DETECTION_INTERVAL == 3
    assert config.REALTIME_HAAR_SCALE_FACTOR > 1.0
    assert config.REALTIME_HAAR_MIN_NEIGHBORS >= 1
    assert config.REALTIME_FPS_WINDOW_SECONDS == 1.0


def test_speaker_attribution_parameters_are_configurable_and_positive():
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


def test_speaker_attribution_weights_sum_to_one():
    weights = (
        config.VOICE_ACTIVITY_WEIGHT,
        config.LIP_MOTION_WEIGHT,
        config.FACE_CONFIDENCE_WEIGHT,
        config.TEMPORAL_WEIGHT,
    )
    assert abs(sum(weights) - 1.0) < 1e-6


def test_package_paths_use_default_data_layout():
    assert paths.DB_PATH == paths.DATA_ROOT / "db" / "faces.db"
    assert paths.INDEX_PATH == paths.DATA_ROOT / "artifacts" / "face_index.npz"
    assert paths.KNOWN_FACES_DIR == paths.DATA_ROOT / "artifacts" / "known_faces"
    assert paths.KNOWN_VOICES_DIR == paths.DATA_ROOT / "artifacts" / "known_voices"
    assert paths.UNKNOWN_FACES_DIR == paths.DATA_ROOT / "artifacts" / "unknown_faces"
    assert paths.UNKNOWN_VOICES_DIR == paths.DATA_ROOT / "artifacts" / "unknown_voices"
    assert paths.INITIAL_VIDEO_DIR == paths.DATA_ROOT / "artifacts" / "initialVideo"
    assert paths.TRACKED_VIDEO_DIR == paths.DATA_ROOT / "artifacts" / "trackedVideo"
    assert paths.WORKFLOW_TEMP_DIR == paths.DATA_ROOT / "temp" / "workflow"
    assert paths.LIVE_VIDEO_DIR == paths.DATA_ROOT / "artifacts" / "live"
    assert paths.LOG_DIR == paths.DATA_ROOT / "artifacts" / "logs"
    assert paths.TRANSCRIPTS_DIR == paths.DATA_ROOT / "transcripts"


def test_paths_import_is_inert_and_does_not_run_migration_or_create_dirs(monkeypatch):
    import face_voice_rec.config.migration as migration

    calls = []
    original_mkdir = Path.mkdir

    def fake_migrate(project_root, data_root):
        calls.append(("migrate", project_root, data_root))

    def spy_mkdir(self, *args, **kwargs):
        calls.append(("mkdir", self.resolve()))
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(migration, "migrate_legacy_data", fake_migrate)
    monkeypatch.setattr(Path, "mkdir", spy_mkdir)

    importlib.reload(paths)

    assert calls == []


def test_initialize_runtime_paths_honors_injected_roots_without_mutating_exports(
    tmp_path,
    monkeypatch,
):
    baseline_paths = {
        "PROJECT_ROOT": paths.PROJECT_ROOT,
        "DATA_ROOT": paths.DATA_ROOT,
        "DB_PATH": paths.DB_PATH,
        "INDEX_PATH": paths.INDEX_PATH,
        "KNOWN_FACES_DIR": paths.KNOWN_FACES_DIR,
        "KNOWN_VOICES_DIR": paths.KNOWN_VOICES_DIR,
        "UNKNOWN_FACES_DIR": paths.UNKNOWN_FACES_DIR,
        "UNKNOWN_VOICES_DIR": paths.UNKNOWN_VOICES_DIR,
        "INITIAL_VIDEO_DIR": paths.INITIAL_VIDEO_DIR,
        "TRACKED_VIDEO_DIR": paths.TRACKED_VIDEO_DIR,
        "WORKFLOW_TEMP_DIR": paths.WORKFLOW_TEMP_DIR,
        "LIVE_VIDEO_DIR": paths.LIVE_VIDEO_DIR,
        "LOG_DIR": paths.LOG_DIR,
        "VOICE_DIAGNOSTICS_PATH": paths.VOICE_DIAGNOSTICS_PATH,
        "TRANSCRIPTS_DIR": paths.TRANSCRIPTS_DIR,
    }

    events = []

    def fake_migrate(project_root, data_root):
        events.append(("migrate", project_root, data_root))

    def fake_mkdir(self, *args, **kwargs):
        events.append(("mkdir", self.resolve()))
        return None

    monkeypatch.setattr(paths, "migrate_legacy_data", fake_migrate)
    monkeypatch.setattr(Path, "mkdir", fake_mkdir)

    project_root = tmp_path / "project"
    data_root = tmp_path / "runtime_data"

    paths.initialize_runtime_paths(project_root=project_root, data_root=data_root)

    assert events
    assert events[0] == ("migrate", project_root.resolve(), data_root.resolve())

    created_dirs = [entry[1] for entry in events if entry[0] == "mkdir"]
    assert created_dirs
    for created_dir in created_dirs:
        assert created_dir.is_relative_to(data_root.resolve())

    assert paths.PROJECT_ROOT == baseline_paths["PROJECT_ROOT"]
    assert paths.DATA_ROOT == baseline_paths["DATA_ROOT"]
    assert paths.DB_PATH == baseline_paths["DB_PATH"]
    assert paths.INDEX_PATH == baseline_paths["INDEX_PATH"]
    assert paths.KNOWN_FACES_DIR == baseline_paths["KNOWN_FACES_DIR"]
    assert paths.KNOWN_VOICES_DIR == baseline_paths["KNOWN_VOICES_DIR"]
    assert paths.UNKNOWN_FACES_DIR == baseline_paths["UNKNOWN_FACES_DIR"]
    assert paths.UNKNOWN_VOICES_DIR == baseline_paths["UNKNOWN_VOICES_DIR"]
    assert paths.INITIAL_VIDEO_DIR == baseline_paths["INITIAL_VIDEO_DIR"]
    assert paths.TRACKED_VIDEO_DIR == baseline_paths["TRACKED_VIDEO_DIR"]
    assert paths.WORKFLOW_TEMP_DIR == baseline_paths["WORKFLOW_TEMP_DIR"]
    assert paths.LIVE_VIDEO_DIR == baseline_paths["LIVE_VIDEO_DIR"]
    assert paths.LOG_DIR == baseline_paths["LOG_DIR"]
    assert paths.VOICE_DIAGNOSTICS_PATH == baseline_paths["VOICE_DIAGNOSTICS_PATH"]
    assert paths.TRANSCRIPTS_DIR == baseline_paths["TRANSCRIPTS_DIR"]
