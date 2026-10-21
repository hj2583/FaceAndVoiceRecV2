# Project Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the application into a `src/face_voice_rec` package, split the Streamlit UI into modules without changing the seven-tab experience, and safely migrate existing runtime data to `data/` on first app/runtime startup.

**Architecture:** Keep `app/main.py` as a tab-composing Streamlit entry point and place existing backend modules in focused face, voice, transcription, pipeline, storage, and config packages. An explicit, idempotent data migration runs before any runtime directory creation or database access; tests use temporary roots and never touch real project data.

**Tech Stack:** Python 3.10+, setuptools `src` package, existing Streamlit/Pytest/runtime requirements, SQLite, `pathlib`, `shutil`.

## Global Constraints

- Preserve the current seven tab labels, order, and workflows.
- Preserve recognition, transcription, attendance, CPU/GPU, and model behavior.
- Preserve all runtime data; never overwrite conflicting destinations.
- Migration tests must use temporary source and destination directories, including synthetic SQLite path references.
- Runtime migration rewrites absolute SQLite path references from legacy roots to canonical `data/` roots before application database initialization.
- Keep `sitecustomize.py` at the repository root.
- Keep runtime dependencies in `requirements.txt` and `requirements-gpu.txt`; do not duplicate them in `pyproject.toml`.
- Do not add speculative services, repositories, or orchestration layers.
- Do not retain root-level compatibility wrappers for former backend modules.
- Do not include or revert the pre-existing README edit that changes launch commands to `streamlit run main.py`; update the README later to the approved `streamlit run app/main.py` command while preserving the rest of that edit.

---

## File Map

| Destination | Responsibility |
|---|---|
| `src/face_voice_rec/config/migration.py` | Conflict-safe, no-clobber file migration and transactional SQLite path-reference repair. |
| `src/face_voice_rec/config/paths.py` | Repository/data path constants and explicit runtime initialization. |
| `src/face_voice_rec/config/settings.py` | Environment-backed runtime settings. |
| `src/face_voice_rec/config/constants.py` | Fixed model, recognition, tracking, and pipeline constants. |
| `src/face_voice_rec/storage/database.py` | Existing SQLite operations. |
| `src/face_voice_rec/domain/face/` | Face backend, detection, embedding/index, and tracking modules. |
| `src/face_voice_rec/domain/voice/` | Audio/VAD, voice embedding, and speaker attribution modules. |
| `src/face_voice_rec/domain/transcription/` | Transcription, refinement, subtitle, and introduction modules. |
| `src/face_voice_rec/pipelines/realtime/` | Realtime process, launcher/runtime, and threaded frame pipeline. |
| `src/face_voice_rec/pipelines/video/` | Video processing/workflow and FFmpeg helpers. |
| `app/main.py` | Initialize runtime, initialize the database, and compose existing tabs. |
| `app/pages/*.py` | One tab renderer per module; include database, audio logs, and attendance pages in addition to the four proposed pages. |
| `app/ui/attendance_view.py` | Existing reusable attendance view/dataframe helpers. |
| `app/ui/renderers.py` | Existing audio-log renderer helpers and genuinely shared UI formatting helpers. |
| `app/ui/status.py` | Runtime status display. |
| `scripts/` | Existing `evaluate_speaker_recognition.py` and `fix_unknown_images.py` CLI utilities. |
| `tests/unit/` | Existing unit tests grouped by subsystem. |
| `tests/integration/` | Package startup and legacy data migration integration tests. |
| `tests/fixtures/runtime_tree.py` | Builder for isolated legacy and destination directory trees used by migration tests. |
| `pyproject.toml` | Setuptools package discovery and pytest test paths, with no duplicated runtime dependencies. |

### Existing module moves

| Current file | Destination |
|---|---|
| `audio_core.py` | `src/face_voice_rec/domain/voice/audio_core.py` |
| `audio_log_utils.py` | `app/ui/renderers.py` |
| `attendance_view.py` | `app/ui/attendance_view.py` |
| `database.py` | `src/face_voice_rec/storage/database.py` |
| `detection_core.py` | `src/face_voice_rec/domain/face/detection_core.py` |
| `evaluate_speaker_recognition.py` | `scripts/evaluate_speaker_recognition.py` |
| `face_backend.py` | `src/face_voice_rec/domain/face/face_backend.py` |
| `face_core.py` | `src/face_voice_rec/domain/face/face_core.py` |
| `ffmpeg_utils.py` | `src/face_voice_rec/pipelines/video/ffmpeg_utils.py` |
| `fix_unknown_images.py` | `scripts/fix_unknown_images.py` |
| `frame_pipeline.py` | `src/face_voice_rec/pipelines/realtime/frame_pipeline.py` |
| `realtime.py` | `src/face_voice_rec/pipelines/realtime/realtime.py` |
| `realtime_launcher.py` | `src/face_voice_rec/pipelines/realtime/realtime_launcher.py` |
| `realtime_runtime.py` | `src/face_voice_rec/pipelines/realtime/realtime_runtime.py` |
| `runtime_status.py` | `app/ui/status.py` |
| `speaker_attribution.py` | `src/face_voice_rec/domain/voice/speaker_attribution.py` |
| `speaker_introductions.py` | `src/face_voice_rec/domain/transcription/speaker_introductions.py` |
| `subtitle_renderer.py` | `src/face_voice_rec/domain/transcription/subtitle_renderer.py` |
| `tracking.py` | `src/face_voice_rec/domain/face/tracking.py` |
| `transcript_refiner.py` | `src/face_voice_rec/domain/transcription/transcript_refiner.py` |
| `transcription_core.py` | `src/face_voice_rec/domain/transcription/transcription_core.py` |
| `video_processor.py` | `src/face_voice_rec/pipelines/video/video_processor.py` |
| `video_workflow.py` | `src/face_voice_rec/pipelines/video/video_workflow.py` |
| `voice_core.py` | `src/face_voice_rec/domain/voice/voice_core.py` |

`sitecustomize.py` stays at the root. The root-level `test_detection_improvements.py` moves to `tests/unit/test_detection_improvements.py`; all existing `tests/test_*.py` files move into `tests/unit/`.

---

### Task 1: Conflict-Safe Legacy Data Migration

**Files:**
- Create: `src/face_voice_rec/__init__.py`
- Create: `src/face_voice_rec/config/__init__.py`
- Create: `src/face_voice_rec/config/migration.py`
- Create: `tests/fixtures/runtime_tree.py`
- Create: `tests/integration/test_data_migration.py`

**Interfaces:**
- Produces `migrate_legacy_data(project_root: Path, data_root: Path) -> None`.
- Produces `DataMigrationConflict(RuntimeError)`, whose message lists every conflicting source and destination.
- File-layout marker: `data/.legacy_migration_v1_complete`.
- Database-reference marker: `data/.legacy_path_references_v1_complete`; this repair runs even when the file-layout marker already exists.
- Migration sources are `faces.db`, `faces.db-wal`, `faces.db-shm`, `face_index.npz`, `known_faces/`, `known_voices/`, `unknown_faces/`, `unknown_voices/`, `initialVideo/`, `trackedVideo/`, `live/`, `logs/`, `transcripts/`, and `.workflow-temp/`.
- Migration destinations are the exact paths in the design spec: database files under `data/db/`, artifacts under `data/artifacts/`, transcripts under `data/transcripts/`, and workflow temp under `data/temp/workflow/`.

- [ ] **Step 1: Write failing migration tests.** Use a fixture that constructs a temporary project root and `data/` root. Cover: moving a real SQLite database plus WAL/SHM files and representative nested face/voice/transcript/workflow artifacts; merging source and destination directories when relative child paths do not overlap; idempotent invocation after the marker exists; conflict detection across the full mapping before any move; a destination created after preflight never being overwritten; a retry after an injected interruption immediately after destination publication; fail-fast on cross-filesystem source/destination before any move; and transactional path-reference repair when the file-layout marker already exists. Include rows from `face_embeddings`, `meetings`, `attendance_events`, `unknown_tracks`, `unknown_samples`, `voice_embeddings`, `unknown_voices`, and `unknown_voice_samples` with legacy absolute paths, plus an external path that must remain unchanged.

```python
def test_conflict_leaves_all_legacy_sources_untouched(tmp_path):
    project_root = tmp_path / "project"
    data_root = project_root / "data"
    source = project_root / "known_faces" / "1" / "sample.jpg"
    destination = data_root / "artifacts" / "known_faces" / "1" / "sample.jpg"
    source.parent.mkdir(parents=True)
    destination.parent.mkdir(parents=True)
    source.write_bytes(b"legacy")
    destination.write_bytes(b"existing")

    with pytest.raises(DataMigrationConflict):
        migrate_legacy_data(project_root, data_root)

    assert source.read_bytes() == b"legacy"
    assert destination.read_bytes() == b"existing"
    assert not (data_root / ".legacy_migration_v1_complete").exists()
```

- [ ] **Step 2: Run the focused tests and verify they fail because the migration interface is absent.**

Run in PowerShell: `$env:PYTHONPATH = "src"` then `python -m pytest tests/integration/test_data_migration.py -q`

Expected: collection fails with the missing `face_voice_rec.config.migration` import.

- [ ] **Step 3: Implement full preflight and fail-closed filesystem rules.** Detect file collisions, path-type mismatches, symlinks, and source/destination filesystem-device mismatches before creating target directories or moving data. Permit disjoint directory merges only. Move files with an atomic no-clobber same-filesystem operation; never call `shutil.move` or fall back to partial cross-device copy. If interruption occurs after destination publication but before source unlink, identify the source/destination as the same file on retry, remove only the legacy source link, and continue. A racing independently created destination must raise `DataMigrationConflict` without altering either file.

- [ ] **Step 4: Add versioned SQLite path-reference repair after file-layout migration.** Even when `.legacy_migration_v1_complete` already exists, inspect only configured path-like text columns in the copied database. In one SQLite transaction, map absolute values under each legacy source root to its corresponding destination root; leave relative paths and paths outside the project root unchanged. Write `.legacy_path_references_v1_complete` only after commit. If interrupted before commit, a later call retries; if interrupted after commit but before marker, a later call performs no further updates and then writes the marker.

- [ ] **Step 5: Run focused migration tests and verify they pass.**

Run in PowerShell: `$env:PYTHONPATH = "src"` then `python -m pytest tests/integration/test_data_migration.py -q`

Expected: all success, merge, idempotence, preflight conflict, SQLite sidecar, and retry cases pass; only temporary paths are changed.

- [ ] **Step 6: Commit the migration boundary.**

```powershell
git add src/face_voice_rec tests/fixtures/runtime_tree.py tests/integration/test_data_migration.py
git commit -m "feat: add safe legacy data migration"
```

### Task 2: Package Metadata and Configuration Split

**Files:**
- Create: `pyproject.toml`
- Create: `src/face_voice_rec/config/paths.py`
- Create: `src/face_voice_rec/config/settings.py`
- Create: `src/face_voice_rec/config/constants.py`
- Modify: `src/face_voice_rec/config/__init__.py`
- Move/delete: root `config.py` into the three configuration modules
- Move: `tests/test_config.py` to `tests/unit/test_config.py`
- Move: root `test_detection_improvements.py` to `tests/unit/test_detection_improvements.py`
- Modify every current Python module and test that imports root `config.py` to use `face_voice_rec.config` before deleting the root module.
- Modify: `app.py` and `realtime.py` entry-point startup ordering
- Test: `tests/unit/test_config.py`

**Interfaces:**
- `config.paths` exports `PROJECT_ROOT`, `DATA_ROOT`, `DB_PATH`, `INDEX_PATH`, `KNOWN_FACES_DIR`, `KNOWN_VOICES_DIR`, `UNKNOWN_FACES_DIR`, `UNKNOWN_VOICES_DIR`, `INITIAL_VIDEO_DIR`, `TRACKED_VIDEO_DIR`, `WORKFLOW_TEMP_DIR`, `LIVE_VIDEO_DIR`, `LOG_DIR`, `VOICE_DIAGNOSTICS_PATH`, and `TRANSCRIPTS_DIR`.
- `initialize_runtime_paths(project_root: Path | None = None, data_root: Path | None = None) -> None` runs migration first, then creates required runtime directories.
- `config.__init__` re-exports the existing configuration names so package modules can use `from face_voice_rec import config` and preserve current monkeypatchable module attributes.
- `pyproject.toml` uses setuptools package discovery from `src`, requires Python `>=3.10`, and configures pytest `testpaths = ["tests"]`; it does not declare runtime dependencies.

- [ ] **Step 1: Move `tests/test_config.py` to `tests/unit/test_config.py`; preserve all 12 original tests from base commit `1ecd20f` and update their imports. Add tests for path derivation and explicit initialization.** Assert default exported paths derive under `DATA_ROOT`; assert importing config does not create directories or run migration; assert `initialize_runtime_paths(project_root, data_root)` migrates and creates directories under the injected roots while leaving the default exported path constants unchanged.
- [ ] **Step 2: In PowerShell set `$env:PYTHONPATH = "src"`, then run `python -m pytest tests/unit/test_config.py -q`; verify collection fails because `face_voice_rec.config.paths` is missing.**
- [ ] **Step 3: Move root `test_detection_improvements.py` to `tests/unit/test_detection_improvements.py` before applying `testpaths = ["tests"]`, so all existing tests remain collected.**
- [ ] **Step 4: Move values from `config.py` without changing their names or defaults.** Put filesystem paths and initialization in `paths.py`, environment reads (`VOICE_DIAGNOSTICS_ENABLED`, `WHISPER_LANGUAGE_MODE`, `WHISPER_LANGUAGE`, and transcript-refinement settings) in `settings.py`, and fixed thresholds/model/pipeline values in `constants.py`. Update every current source and test import of root `config` to package imports (`from face_voice_rec import config` or named re-exports from `face_voice_rec.config`); then remove root `config.py`. Keep package imports inert; only `initialize_runtime_paths()` performs migration or mkdir operations.
- [ ] **Step 5: Call `initialize_runtime_paths()` in `app.py` after `st.set_page_config()` and before `init_db()`. In `realtime.py`, call it inside the `if __name__ == "__main__"` entry-point block immediately before `run()`, not at import time.** Tests must use temporary database paths and must not run default-path migration.
- [ ] **Step 6: Add setuptools metadata and run `python -m pip install -e . --no-deps`; then run `python -m pytest tests/unit/test_config.py -q`.** Both installation and tests must pass without accessing production runtime data.
- [ ] **Step 7: Commit package/config foundation.**

```powershell
git add pyproject.toml config.py app.py realtime.py database.py evaluate_speaker_recognition.py face_backend.py face_core.py fix_unknown_images.py speaker_attribution.py speaker_introductions.py subtitle_renderer.py tracking.py transcript_refiner.py transcription_core.py video_processor.py video_workflow.py voice_core.py tests/conftest.py tests/test_config.py tests/test_database_voice.py tests/test_speaker_attribution.py tests/test_speaker_introductions.py tests/test_subtitle_renderer.py tests/test_transcript_refiner.py tests/test_transcription_core.py tests/test_video_workflow.py tests/test_voice_core.py tests/unit/test_config.py tests/unit/test_detection_improvements.py src/face_voice_rec/config
git commit -m "refactor: package runtime configuration"
```

### Task 3: Storage and Face Domain Migration

**Files:**
- Move: `database.py` to `src/face_voice_rec/storage/database.py`
- Move: `face_core.py`, `face_backend.py`, `detection_core.py`, `tracking.py` to `src/face_voice_rec/domain/face/`
- Move/update: `tests/test_database_attendance.py`, `tests/test_database_audio_logs.py`, `tests/test_database_unknown_cleanup.py`, `tests/test_database_voice.py`, `tests/test_face_backend.py`, `tests/test_face_core_preprocessing.py`, and `tests/test_detection_core.py` into `tests/unit/`
- Delete: root `test_detection_improvements.py` after confirming its already-moved `tests/unit/test_detection_improvements.py` copy contains the same five tests.
- Create/modify: `tests/unit/test_package_imports.py`
- Add package initializers and update every current consumer of database/face modules to package-qualified imports, including `app.py`, `realtime.py`, `video_processor.py`, shared UI modules, and tests that remain in the root `tests/` directory until Task 9.

**Interfaces:**
- Preserve existing database and face public function/class names; callers use `face_voice_rec.storage.database` and `face_voice_rec.domain.face.*`.
- Face modules obtain config from `face_voice_rec.config`; persistence remains SQLite-backed with the same schema and queries.

- [ ] **Step 1: Add package-import smoke tests for database and face modules, asserting representative exports (`init_db`, `FaceIndex`, `get_face_backend`, `detect_faces_tiled`, and `CentroidTracker`).**
- [ ] **Step 2: Run the new import tests and verify they fail before the modules move.**
- [ ] **Step 3: Use `git mv` for the listed source and test files, add package `__init__.py` files, and replace every old absolute database/face import across all source modules and tests with its package path. Preserve function implementations and behavior.
- [ ] **Step 4: Search for any remaining root imports of the moved modules.**

Run: `rg -n "^(from|import) (database|face_core|face_backend|detection_core|tracking)" -g "*.py"`

Expected: no application or test imports remain for moved database/face modules.
- [ ] **Step 5: Run the database and face test slice.**

Run: `python -m pytest tests/unit/test_database_attendance.py tests/unit/test_database_audio_logs.py tests/unit/test_database_unknown_cleanup.py tests/unit/test_database_voice.py tests/unit/test_face_backend.py tests/unit/test_face_core_preprocessing.py tests/unit/test_detection_core.py tests/unit/test_detection_improvements.py -q`

Expected: all existing database/face tests and new import tests pass.

- [ ] **Step 6: Commit the storage and face slice.**

```powershell
git add -A -- src/face_voice_rec/storage src/face_voice_rec/domain/face app.py attendance_view.py realtime.py speaker_introductions.py video_processor.py video_workflow.py voice_core.py tests test_detection_improvements.py
git commit -m "refactor: move storage and face modules into package"
```

### Task 4: Voice and Transcription Domain Migration

**Files:**
- Move `audio_core.py`, `voice_core.py`, and `speaker_attribution.py` to `src/face_voice_rec/domain/voice/`.
- Move `transcription_core.py`, `transcript_refiner.py`, `subtitle_renderer.py`, and `speaker_introductions.py` to `src/face_voice_rec/domain/transcription/`.
- Move/update the existing audio, voice, speaker-attribution, speaker-introduction, subtitle, transcript-refiner, and transcription tests into `tests/unit/`.
- Update every current consumer of moved voice/transcription modules to package-qualified imports, including `app.py`, realtime/video workflows, remaining root tests, and tests moved during this task.
- Modify `evaluate_speaker_recognition.py`, `app.py`, `realtime.py`, `video_processor.py`, and `video_workflow.py` where they import moved voice/transcription modules.
- Update package initializers and imports in touched modules/tests.

**Interfaces:**
- Preserve existing voice/audio/transcription public APIs and dataclasses.
- Voice and transcription modules import storage/face/config through `face_voice_rec` package paths; no UI imports are introduced.

- [ ] **Step 1: Add import smoke tests for `RealtimeVAD`, `diarize_meeting_audio`, `SpeakerAttributor`, `process_meeting_transcription`, `TranscriptRefinementError`, `SubtitleWord`, and `extract_self_introductions` from their new module paths.**
- [ ] **Step 2: Run the new tests and verify they fail because the package modules are not yet present.**
- [ ] **Step 3: Move the listed modules/tests and update every current source/test consumer to package-qualified imports while retaining implementations, behavior, and configuration values unchanged.**
- [ ] **Step 4: Run the voice/transcription tests.**

Run: `python -m pytest tests/unit/test_audio_core.py tests/unit/test_voice_core.py tests/unit/test_speaker_attribution.py tests/unit/test_speaker_introductions.py tests/unit/test_subtitle_renderer.py tests/unit/test_transcript_refiner.py tests/unit/test_transcription_core.py -q`

Expected: all existing behavior tests and new import tests pass.

- [ ] **Step 5: Commit the voice/transcription slice.**

```powershell
git add -A -- src/face_voice_rec/domain/voice src/face_voice_rec/domain/transcription tests/unit tests/test_audio_core.py tests/test_voice_core.py tests/test_speaker_attribution.py tests/test_speaker_introductions.py tests/test_subtitle_renderer.py tests/test_transcript_refiner.py tests/test_transcription_core.py tests/test_video_workflow.py app.py evaluate_speaker_recognition.py realtime.py video_processor.py video_workflow.py
git commit -m "refactor: move voice and transcription modules into package"
```

### Task 5: Realtime and Video Pipeline Migration

**Files:**
- Move realtime modules to `src/face_voice_rec/pipelines/realtime/`: `realtime.py`, `realtime_launcher.py`, `realtime_runtime.py`, and `frame_pipeline.py`.
- Move video modules to `src/face_voice_rec/pipelines/video/`: `video_workflow.py`, `video_processor.py`, and `ffmpeg_utils.py`.
- Move `evaluate_speaker_recognition.py` and `fix_unknown_images.py` to `scripts/`.
- Move `tests/test_frame_pipeline.py`, `tests/test_realtime.py`, `tests/test_realtime_launcher.py`, `tests/test_realtime_runtime.py`, `tests/test_ffmpeg_utils.py`, `tests/test_video_processor.py`, and `tests/test_video_workflow.py` to `tests/unit/`.
- Update current pipeline consumers/importers in `app.py`, `realtime.py`, `video_processor.py`, `video_workflow.py`, the two moved Task 4 transcription modules that import FFmpeg helpers, and moved tests.
- Update package initializers and imports in touched modules/tests.

**Interfaces:**
- Preserve the existing pipeline functions, result types, and CLI utility behavior.
- `launch_realtime(camera: int, cwd: Path, log_dir: Path, startup_timeout: float = 1.0, popen=subprocess.Popen) -> LaunchResult` has no script-path parameter and starts `[sys.executable, "-m", "face_voice_rec.pipelines.realtime.realtime", "--camera", str(camera)]` with the supplied project-root cwd and configured log directory.
- Direct realtime and moved CLI utility entry points call `initialize_runtime_paths()` before opening the database or writing runtime data. `app.py` already initializes paths before `init_db()` from Task 2. Importing modules in tests does not call runtime initialization.

- [ ] **Step 1: Update `tests/test_realtime_launcher.py` to call the new launcher signature and assert the exact `sys.executable -m face_voice_rec.pipelines.realtime.realtime --camera N` argument vector, project-root cwd, and log directory.**
- [ ] **Step 2: Run the focused launcher test and verify it fails against the current file-path subprocess command.**
- [ ] **Step 3: Move all seven pipeline modules, their seven tests, and both CLI utilities to the exact destinations. Add `__init__.py` files for the realtime/video packages.**
- [ ] **Step 4: Update all moved and existing callers to package-qualified imports; switch app caller and launcher tests to the new API/command. Keep `initialize_runtime_paths()` inside each executable entry point only; preserve its existing app and realtime call order.**
- [ ] **Step 5: Run pipeline tests.**

Run: `python -m pytest tests/unit/test_frame_pipeline.py tests/unit/test_realtime.py tests/unit/test_realtime_launcher.py tests/unit/test_realtime_runtime.py tests/unit/test_ffmpeg_utils.py tests/unit/test_video_processor.py tests/unit/test_video_workflow.py tests/unit/test_package_imports.py -q`

Expected: all existing pipeline tests and the module-launch test pass without opening production database/data paths.

- [ ] **Step 6: Commit the pipeline slice.**

```powershell
git add -A -- src/face_voice_rec/pipelines scripts tests/unit app.py src/face_voice_rec/domain/transcription/subtitle_renderer.py src/face_voice_rec/domain/transcription/transcription_core.py
git commit -m "refactor: move realtime and video pipelines into package"
```

### Task 6: Move Shared UI Helpers and Runtime Status

**Files:**
- Create: `app/__init__.py` and `app/ui/__init__.py`.
- Move `attendance_view.py`, `audio_log_utils.py`, and `runtime_status.py` to the shared UI paths.
- Move `tests/test_attendance_view.py`, `tests/test_audio_log_utils.py`, and `tests/test_runtime_status.py` into `tests/unit/`.
- Modify: `app.py` imports of attendance, audio-log, and status helpers.

**Interfaces:**
- `app.ui.attendance_view` preserves all existing attendance frame/filter/export APIs.
- `app.ui.renderers.build_audio_log_frame` preserves the existing audio-log helper signature.
- `app.ui.status.get_runtime_status` preserves its existing return shape.
- Root `app.py` continues to render the same seven tabs in this task; only helper import paths change.

- [ ] **Step 1: Add or update helper tests to import `app.ui.attendance_view`, `app.ui.renderers.build_audio_log_frame`, and `app.ui.status.get_runtime_status`; capture the missing-package import failure before moving the modules.**
- [ ] **Step 2: Move the three helper modules and their existing tests with `git mv`; keep the helper implementations unchanged.**
- [ ] **Step 3: Update only the matching imports in `app.py` and the moved tests.**
- [ ] **Step 4: Run the helper test slice.**

Run: `python -m pytest tests/unit/test_attendance_view.py tests/unit/test_audio_log_utils.py tests/unit/test_runtime_status.py -q`

Expected: helper tests pass at package import locations with their existing behavior unchanged.

- [ ] **Step 5: Commit the shared UI helper moves.**

```powershell
git add -A -- app/__init__.py app/ui app.py attendance_view.py audio_log_utils.py runtime_status.py tests/test_attendance_view.py tests/test_audio_log_utils.py tests/test_runtime_status.py tests/unit/test_attendance_view.py tests/unit/test_audio_log_utils.py tests/unit/test_runtime_status.py
git commit -m "refactor: move shared streamlit UI helpers"
```

### Task 7: Split the Streamlit UI Without Changing Tabs

**Files:**
- Create: `app/main.py`, `app/pages/__init__.py`, and seven tab renderer modules.
- Move: root `app.py` entry point into `app/main.py`, with the seven renderer bodies moved to their named modules.
- Create: `app/pages/realtime.py`, `app/pages/video.py`, `app/pages/face_database.py`, `app/pages/audio_logs.py`, `app/pages/attendance.py`, `app/pages/transcripts.py`, and `app/pages/voice_enrollment.py`.
- Create: `tests/unit/test_app_navigation.py`.
- Delete: root `app.py` after the entry-point smoke test and caller search pass.

**Interfaces:**
- Each page module exports exactly one `render() -> None`: `app.pages.realtime.render`, `video.render`, `face_database.render`, `audio_logs.render`, `attendance.render`, `transcripts.render`, and `voice_enrollment.render`.
- `app.main.main()` creates tabs with the existing labels/order and calls each matching renderer once.
- `app.main` calls `st.set_page_config()` first, then `initialize_runtime_paths()`, then `init_db()`, then runs an explicit single-page `st.navigation` entry whose page renders the seven tabs. This disables Streamlit's implicit `app/pages/` discovery so no extra sidebar links appear. It invokes `main()` only when run as the Streamlit script, so importing it is inert.
- The exact tab order/labels remain: Realtime, Video, Face Database, Audio Logs, Attendance, Transcripts, Voice Enrollment. Keep all existing visible text and Unicode unchanged.
- The NumPy legacy-alias compatibility shim executes in `app/main.py` before importing pandas or Streamlit, as in the former `app.py`; page modules do not duplicate it.
- All navigation and startup tests use monkeypatched Streamlit/runtime/database functions; no test imports or launches the real root app with production initialization.

- [ ] **Step 1: Add `tests/unit/test_app_navigation.py` with fake Streamlit navigation/tabs and renderer spies. Record all calls in one event list. Assert exact order: `set_page_config`, `initialize_runtime_paths`, `init_db`, one-page `st.navigation`, then the seven existing tab labels and one renderer call each. Also assert importing `app.main` alone triggers no startup calls.**
- [ ] **Step 2: Run `python -m pytest tests/unit/test_app_navigation.py -q` and verify the test fails because `app.main` and the page modules do not exist.**
- [ ] **Step 3: Move the NumPy legacy-alias shim to the first executable lines of `app/main.py`, before importing pandas, Streamlit, or any page module. Remove duplicate shims from page modules.**
- [ ] **Step 4: Extract `render_realtime`, `render_video`, `render_database`, and `get_face_index()` into their page modules; then extract `render_audio_logs`, `render_attendance`, `render_transcripts`, and `render_voice_enrollment`. Preserve page state keys, text, labels, exceptions, callbacks, and operation order.**
- [ ] **Step 5: Implement `app.main.main()` with the original seven tabs in the original order; after runtime initialization call `st.navigation([st.Page(render_application, title="AI Face + Active Speaker Recognition", default=True)]).run()`. The sole page function renders the seven tabs. Keep the entry-point guard inert on import.**
- [ ] **Step 6: Preserve Task 5's realtime launcher API in `app/pages/realtime.py`; search source files, scripts, and tests for root `app.py` launch references and update them to `app/main.py`. Do not touch README.md here; Task 9 updates its existing local launch-command edit.**
- [ ] **Step 7: Run Streamlit from a disposable source copy that has no runtime data directories; verify there is no sidebar page list, the seven-tab UI appears, and only the copy receives a `data/` tree. Do not launch against the live repository data.**
- [ ] **Step 8: Run navigation and shared UI helper tests.**

Run: `python -m pytest tests/unit/test_app_navigation.py tests/unit/test_attendance_view.py tests/unit/test_audio_log_utils.py tests/unit/test_runtime_status.py -q`

Expected: all seven tabs are asserted in their prior order and helper behavior remains unchanged.

- [ ] **Step 9: Commit the UI split.**

```powershell
git add -A -- app app.py tests/unit/test_app_navigation.py
git commit -m "refactor: split streamlit tabs into page modules"
```

### Task 8: Test Layout and Import Cleanup

**Files:**
- Modify: `tests/conftest.py` to remove `sys` import and repository-root `sys.path` insertion; keep its temporary database/video/voice isolation fixtures.
- No test modules remain at `tests/` root; all collected test modules are already grouped under `tests/unit/` and `tests/integration/`.

**Interfaces:**
- Test imports resolve through editable installation; no `sys.path.insert` modifies the repository root.
- Tests that use data patch package config or explicit temporary roots; no test runs default-path migration.

- [ ] **Step 1: Search all test modules for imports of moved root modules and update any remaining imports to package-qualified paths.**
- [ ] **Step 2: Remove `import sys` and `sys.path.insert(...)` from `tests/conftest.py`; preserve the shared database/video/voice isolation fixtures.**
- [ ] **Step 3: Run an import scan and verify no imports of moved top-level modules remain in tests.**

Run: `rg -n "^(from|import) (audio_core|audio_log_utils|attendance_view|config|database|detection_core|face_backend|face_core|ffmpeg_utils|frame_pipeline|realtime|realtime_launcher|realtime_runtime|runtime_status|speaker_attribution|speaker_introductions|subtitle_renderer|tracking|transcript_refiner|transcription_core|video_processor|video_workflow|voice_core)" tests -g "*.py"`

Expected: no matches for moved top-level modules.

- [ ] **Step 4: Run the full suite.**

Run: `python -m pytest -q`

Expected: all tests pass except the documented baseline temp-path failure; test collection must include all 346 existing tests plus additions.

- [ ] **Step 5: Commit test import cleanup.**

```powershell
git add tests/conftest.py tests/unit tests/integration
git commit -m "test: organize suite around packaged modules"
```

### Task 9: Data Ignore Rules, README, and Runtime Validation

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`, preserving unrelated content and the pre-existing local edit while replacing the incorrect interim `streamlit run main.py` command with `streamlit run app/main.py`.
- Verify: `pyproject.toml`, package/app entry points, migration integration tests, and all test modules.

**Interfaces:**
- Setup command installs existing requirements and package: `python -m pip install -r requirements.txt` followed by `python -m pip install -e .`.
- App command is `streamlit run app/main.py`.
- Runtime locations match the migration destinations from Task 1.

- [ ] **Step 1: Update `.gitignore` to ignore generated database, index, media, transcript, log, temp, and migration marker files under `data/`, and ignore `*.egg-info/`; retain relevant Python, virtualenv, test-cache, and legacy-root data ignores with `.gitkeep` exceptions to protect not-yet-migrated checkouts.**
- [ ] **Step 2: Update README setup, project tree, important data folders, troubleshooting, and launch commands to match the package and `data/` structure. Preserve the user's existing Ollama configuration and all unrelated README edits. Use `python -m pip` consistently, including the optional GPU reinstall instruction; draw package `__init__.py` files at their correct tree locations.
- [ ] **Step 3: Run the full tests and editable install.**

Run: `python -m pip install -e . --no-deps`

Run: `python -m pytest -q`

Expected: editable install succeeds and the complete suite has no new failures beyond `tests/unit/test_video_workflow.py::test_relative_output_path_uses_temp_outside_output_folder`, the recorded baseline failure.

- [ ] **Step 4: Run a production-safe import/path smoke check without invoking runtime initialization.**

Run: `python -c "from face_voice_rec.config.paths import DB_PATH, TRANSCRIPTS_DIR; print(DB_PATH); print(TRANSCRIPTS_DIR)"`

Expected: both paths resolve under `data/`; importing paths does not create directories or run legacy migration. App entry-point import behavior is covered by the navigation test.

- [ ] **Step 5: Start Streamlit from a disposable copy containing source files but no runtime data directories; verify there is no automatic sidebar page list, startup reaches the seven-tab UI, and any generated `data/` exists only in the copy. Stop the test server after verification.**
- [ ] **Step 6: Test legacy migration only against a disposable copy of the legacy directory tree. Verify destinations, SQLite sidecars, marker creation, conflict behavior, and a second-start no-op. The production migration has already occurred; never run migration against the live repository again.**
- [ ] **Step 7: Commit docs/ignore updates after review.**

```powershell
git add .gitignore README.md
git commit -m "docs: document packaged app and data layout"
```

### Task 10: Final Review Hardening

**Files:**
- Modify: `src/face_voice_rec/config/migration.py` for atomic no-clobber file moves, same-filesystem preflight, retry detection, and transactional SQLite path-reference migration.
- Modify: `tests/integration/test_data_migration.py` for race, retry, cross-filesystem, and v2-reference migration coverage using temporary roots/SQLite databases.
- Modify: `tests/conftest.py` to isolate `WORKFLOW_TEMP_DIR` for every test.
- Modify: `pyproject.toml` to package both `face_voice_rec` from `src/` and `app` from the repository root.
- Modify: `tests/unit/test_app_navigation.py` or add a focused packaging smoke test for `app.main` import from outside the repository after editable install.
- Modify: `README.md` and `.gitignore` only for the deferred Task 9 minor corrections.
- Modify: the design spec to record the SQLite path-reference migration and no-clobber/fail-fast guarantees.

**Interfaces:**
- A second marker `data/.legacy_path_references_v1_complete` records committed DB path rewrites; this migration runs even if `.legacy_migration_v1_complete` exists.
- Path repair maps only absolute DB values under legacy project roots; paths outside the repo remain unchanged.
- File moves never overwrite a destination. If a source and destination are on different filesystems, migration fails in preflight before any move rather than copying partially.
- `app` and `face_voice_rec` imports both work from a clean external working directory after `python -m pip install -e . --no-deps`.
- Workflow tests set `config.WORKFLOW_TEMP_DIR` under `tmp_path`; test runs never create scratch files under live `data/`.

- [ ] **Step 1: Add failing tests for path-reference repair from an existing v1 migration state.** Build a temporary SQLite database with the project's actual path-bearing tables/columns and rows for face embeddings, unknown faces/samples, voice embeddings/unknown voices/samples, meetings, attendance source refs, and one external path. Include both native Windows and POSIX-form absolute paths. Create corresponding canonical target files, write the v1 marker, call migration, and assert only old-root values change, each rewritten path preserves its original separator style, and the v2 marker is created. Assert a second call makes no further changes.
- [ ] **Step 2: Add failing migration tests for a destination appearing after preflight, retry after destination publication but before source unlink, and cross-device preflight failure. Assert no destination is overwritten, both source/destination survive conflicts, retries recover same-file state, and no sources move on cross-device rejection.
- [ ] **Step 3: Implement atomic no-clobber same-filesystem moves.** Reject symlinks and differing filesystem IDs before any changes. Create target directories and move directory contents recursively. Publish files without replacing existing destinations; on retry, recognize a destination hard-linked to the source as a prior successful publish and remove only the source link. Never use `shutil.move`'s copy fallback.
- [ ] **Step 4: Implement transactional path-reference migration with its own marker.** After file-layout migration (or when its marker already exists), open only `data/db/faces.db`, inspect the known path-bearing columns, rewrite absolute legacy-root prefixes to their canonical mapped path while preserving whether the stored value used `/` or `\\`, and commit all updates together. Marker writing occurs after DB commit. A missing database is a no-op; other SQLite errors stop startup without writing the v2 marker.
- [ ] **Step 5: Add `WORKFLOW_TEMP_DIR` to the autouse test isolation fixture, update `.gitignore` legacy root rules and README GPU/tree nits, and configure root `app` package discovery in `pyproject.toml`.
- [ ] **Step 6: Install editable with `--no-deps`, change working directory to a disposable directory, and verify `import app.main` and `import face_voice_rec` succeed without `PYTHONPATH` or runtime initialization.**
- [ ] **Step 7: Run focused migration/workflow/navigation tests, then the full suite.** Do all migration/database tests on temporary trees. Do not run runtime initialization against the live repository. Confirm the previous workflow temp-path failure is now isolated and report the actual suite result.
- [ ] **Step 8: Verify the already-migrated production SQLite database only by read-only inspection; do not execute the new path update on the live database during development.** The next normal app startup will run the versioned reference repair before `init_db()`.
- [ ] **Step 9: Commit the review fixes, tests, packaging, fixture, spec, and minor documentation corrections without staging `data/`, `live/.gitkeep`, the plan, or `.superpowers/sdd/`.**

```powershell
git add .gitignore README.md pyproject.toml src/face_voice_rec/config/migration.py tests/conftest.py tests/integration/test_data_migration.py tests/unit/test_app_navigation.py docs/superpowers/specs/2026-10-13-project-restructure-design.md
git commit -m "fix: harden migration and installed app startup"
```

## Final Acceptance

- The application launches with `streamlit run app/main.py` after editable installation.
- All seven existing tabs and workflows remain available.
- Backend modules import from `face_voice_rec`; no root compatibility wrappers remain.
- Runtime paths resolve under `data/` and do not depend on the current working directory.
- First runtime initialization migrates all listed legacy data once, never overwrites conflicting content, repairs stored SQLite references exactly once, and can retry after interruption.
- Test collection and test execution do not migrate or mutate production runtime data.
- The full test suite, package import check, and disposable-data startup/migration checks pass.