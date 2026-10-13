import importlib
from pathlib import Path

import face_voice_rec.config.paths as paths


def test_path_derivation_uses_data_layout_defaults():
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


def test_import_is_inert_and_does_not_run_migration_or_create_dirs(monkeypatch):
    import face_voice_rec.config.migration as migration

    calls = []
    original_mkdir = Path.mkdir

    def fake_migrate(project_root, data_root):
        calls.append(("migrate", project_root, data_root))

    def spy_mkdir(self, *args, **kwargs):
        calls.append(("mkdir", self))
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(migration, "migrate_legacy_data", fake_migrate)
    monkeypatch.setattr(Path, "mkdir", spy_mkdir)

    importlib.reload(paths)

    assert calls == []


def test_initialize_runtime_paths_migrates_before_mkdir_and_honors_injected_roots(
    tmp_path,
    monkeypatch,
):
    importlib.reload(paths)

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

    assert paths.PROJECT_ROOT == project_root.resolve()
    assert paths.DATA_ROOT == data_root.resolve()

    runtime_paths = (
        paths.DB_PATH,
        paths.INDEX_PATH,
        paths.KNOWN_FACES_DIR,
        paths.KNOWN_VOICES_DIR,
        paths.UNKNOWN_FACES_DIR,
        paths.UNKNOWN_VOICES_DIR,
        paths.INITIAL_VIDEO_DIR,
        paths.TRACKED_VIDEO_DIR,
        paths.WORKFLOW_TEMP_DIR,
        paths.LIVE_VIDEO_DIR,
        paths.LOG_DIR,
        paths.VOICE_DIAGNOSTICS_PATH,
        paths.TRANSCRIPTS_DIR,
    )

    for runtime_path in runtime_paths:
        assert runtime_path.is_relative_to(data_root.resolve())

    importlib.reload(paths)
