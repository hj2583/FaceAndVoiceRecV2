# Project Restructure Design

Date: 2026-10-13

## Goal

Restructure the existing FaceAndVoiceRecV2 application around a `src/face_voice_rec` Python package, a separated Streamlit UI, and organized runtime data directories. Preserve the current seven-tab experience and all existing runtime data. The restructure will not introduce service or repository abstractions unless an existing responsibility clearly supports them.

## Current State

- `app.py` contains the Streamlit entry point and all seven tab renderers.
- Backend modules, configuration, and helper modules are at the repository root and use root-level absolute imports.
- `config.py` defines project-root data paths and creates several directories during import.
- Runtime state is spread across root-level `faces.db`, `face_index.npz`, face and voice folders, videos, logs, transcripts, and `.workflow-temp/`.
- Tests import root-level modules; `tests/conftest.py` inserts the repository root into `sys.path`.
- The latest restructure commit changed the README's proposed tree; it did not move code.

## Architecture

### Application UI

`app/main.py` remains the single Streamlit entry point and preserves the current seven tabs and their order: Realtime, Video, Face Database, Audio Logs, Attendance, Transcripts, and Voice Enrollment.

The named UI page modules will be `realtime.py`, `video.py`, `transcripts.py`, and `voice_enrollment.py`. Three additional page modules, `face_database.py`, `audio_logs.py`, and `attendance.py`, will preserve the other existing tabs. Each page module owns its tab renderer; `main.py` composes the tabs and calls those renderers.

Shared UI modules will contain the attendance view helpers, audio-log frame/rendering helpers, and runtime status display. UI code may depend on the application/domain/pipeline packages, but those packages will not import Streamlit UI modules.

### Python package

The existing modules will move into `src/face_voice_rec` and use package-qualified imports. The mapping is:

| Package | Existing modules |
|---|---|
| `config` | Split `config.py` into paths/migration, environment-backed settings, and fixed constants. |
| `domain/face` | `face_core.py`, `face_backend.py`, `detection_core.py`, `tracking.py` |
| `domain/voice` | `voice_core.py`, `audio_core.py`, `speaker_attribution.py` |
| `domain/transcription` | `transcription_core.py`, `transcript_refiner.py`, `subtitle_renderer.py`, `speaker_introductions.py` |
| `pipelines/realtime` | `realtime.py`, `realtime_launcher.py`, `realtime_runtime.py`, `frame_pipeline.py` |
| `pipelines/video` | `video_workflow.py`, `video_processor.py`, `ffmpeg_utils.py` |
| `storage` | `database.py` |

`attendance_view.py` and `audio_log_utils.py` move under `app/ui/`; `runtime_status.py` becomes `app/ui/status.py`. The existing `evaluate_speaker_recognition.py` and `fix_unknown_images.py` utilities move under `scripts/`. `sitecustomize.py` remains at the repository root because Python discovers it as a startup hook. No empty attendance service, orchestration layer, repository layer, or speculative service modules will be added.

`pyproject.toml` will configure the `src/face_voice_rec` package and the root `app` package so both resolve after editable installation. Runtime dependencies remain managed by the existing requirements files. Setup documentation will install the project in editable mode with `pip install -e .`, and the app will launch with `streamlit run app/main.py`. No root-level backend compatibility wrappers will be retained; imports of the former top-level modules are replaced by package-qualified imports.

### Tests and documentation

Existing tests will be grouped under `tests/unit/`; focused migration and package-startup integration tests will live under `tests/integration/`, with shared fixtures under `tests/fixtures/` or `tests/conftest.py` as appropriate. Tests will import the installed package rather than inserting the repository root into `sys.path`.

The README will document the new launch/setup commands, source layout, data locations, and one-time legacy migration behavior. `.gitignore` will target generated files under `data/` while retaining appropriate Python/test ignores.

## Runtime Data Layout and Migration

The target data layout is:

- `faces.db` and SQLite `faces.db-wal`/`faces.db-shm` sidecars: `data/db/`
- `face_index.npz`, `known_faces/`, `known_voices/`, `unknown_faces/`, `unknown_voices/`, `initialVideo/`, `trackedVideo/`, `live/`, and `logs/`: `data/artifacts/`
- `transcripts/`: `data/transcripts/`
- `.workflow-temp/`: `data/temp/workflow/`

`known_voices/` is included even though it is absent from the proposed tree because the current application stores enrolled voice data there. Path constants will resolve from the repository root, not the process working directory.

Before creating new data directories or opening SQLite, centralized path initialization runs versioned legacy migrations. The file migration preflights all source/destination paths. Existing destination directories may be merged, but a file, symbolic link, incompatible path, or destination created after preflight is a conflict; migration never overwrites or merges conflicting file contents. File publication must be atomic and no-clobber. If source and destination are on different filesystems, migration fails before moving data rather than using a partial copy fallback. For same-filesystem moves, interruption after destination linking is detected as the same file on retry; a completion marker is written only after all files move successfully.

SQLite contains persisted paths for face/voice embeddings and samples, audio/video sources, and attendance source references. After the file-layout marker is complete, a second versioned, transactional migration rewrites absolute paths rooted under the old project data folders to their corresponding canonical `data/` paths. This reference migration runs even when the file-layout marker already exists, so installations migrated by an earlier version are repaired on their next startup. It updates the currently known path-bearing columns defensively (only when each table/column exists): `face_embeddings(image_path, embedding_path)`, `unknown_tracks(image_path, embedding_path)`, `unknown_samples(image_path, embedding_path)`, `meetings(video_path)`, `attendance_events(source_ref)`, `voice_embeddings(embedding_path, audio_path)`, `unknown_voices(embedding_path)`, and `unknown_voice_samples(embedding_path, audio_path, source_ref)`. A separate reference marker (`data/.legacy_path_references_v1_complete`) is written only after the database transaction commits; retrying after interruption is idempotent. Paths outside the legacy project tree remain unchanged.

Conflicts stop startup with the affected paths and leave source files untouched. Empty legacy directories are removed only after their contents have been moved. Tests use isolated temporary project/database trees; application tests and smoke checks never run these migrations against live user data.

## Behavior and Compatibility

- Preserve the current tab labels, order, and workflows.
- Preserve all runtime data by moving it to the new canonical locations on first startup.
- Keep CPU/GPU requirements and model/runtime behavior unchanged.
- Update in-repository imports and subprocess launch paths to the package layout.
- Replace legacy root-module imports with `face_voice_rec` package imports; no compatibility wrappers are in scope.
- Fail clearly and without overwriting data when destination conflicts exist.

## Validation

Migration tests will use temporary directories and cover successful moves, SQLite sidecar handling, disjoint directory merges, idempotent repeat startup, destination races without overwrite, same-filesystem retry after interruption, cross-filesystem fail-fast, and transactional path-reference repair including an existing file-migration marker. Package tests will verify representative imports and configuration paths without touching project data. A clean editable install must expose both `face_voice_rec` and `app` imports from outside the repository working directory.

Before completion, run the full pytest suite, install/import the package in editable mode, and perform a Streamlit startup smoke test. Confirm the expected data paths, tab composition, and launch command. Test execution must not migrate or modify the user's actual runtime data.

## Out of Scope

- Recognition, transcription, attendance, or UI feature changes.
- Introducing new service, repository, or orchestration layers without an existing implementation boundary.
- Retaining root-level import compatibility wrappers.
- Moving `sitecustomize.py` from the repository root.
- Duplicating the existing CPU/GPU dependency declarations in `pyproject.toml`.