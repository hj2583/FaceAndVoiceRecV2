from __future__ import annotations

from pathlib import Path

from .migration import migrate_legacy_data


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"

# Compatibility alias for legacy callers.
BASE_DIR = PROJECT_ROOT


def _derive_paths(data_root: Path) -> dict[str, Path]:
    return {
        "DB_PATH": data_root / "db" / "faces.db",
        "INDEX_PATH": data_root / "artifacts" / "face_index.npz",
        "KNOWN_FACES_DIR": data_root / "artifacts" / "known_faces",
        "KNOWN_VOICES_DIR": data_root / "artifacts" / "known_voices",
        "UNKNOWN_FACES_DIR": data_root / "artifacts" / "unknown_faces",
        "UNKNOWN_VOICES_DIR": data_root / "artifacts" / "unknown_voices",
        "INITIAL_VIDEO_DIR": data_root / "artifacts" / "initialVideo",
        "TRACKED_VIDEO_DIR": data_root / "artifacts" / "trackedVideo",
        "WORKFLOW_TEMP_DIR": data_root / "temp" / "workflow",
        "LIVE_VIDEO_DIR": data_root / "artifacts" / "live",
        "LOG_DIR": data_root / "artifacts" / "logs",
        "VOICE_DIAGNOSTICS_PATH": data_root / "artifacts" / "logs" / "speaker_diagnostics.json",
        "TRANSCRIPTS_DIR": data_root / "transcripts",
    }


def _apply_runtime_paths(project_root: Path, data_root: Path) -> None:
    global PROJECT_ROOT, DATA_ROOT, BASE_DIR
    PROJECT_ROOT = project_root.resolve()
    DATA_ROOT = data_root.resolve()
    BASE_DIR = PROJECT_ROOT
    globals().update(_derive_paths(DATA_ROOT))


def initialize_runtime_paths(
    project_root: Path | None = None,
    data_root: Path | None = None,
) -> None:
    resolved_project_root = (project_root or PROJECT_ROOT).resolve()
    resolved_data_root = (data_root or DATA_ROOT).resolve()

    _apply_runtime_paths(resolved_project_root, resolved_data_root)

    migrate_legacy_data(PROJECT_ROOT, DATA_ROOT)

    required_directories = (
        DB_PATH.parent,
        INDEX_PATH.parent,
        KNOWN_FACES_DIR,
        KNOWN_VOICES_DIR,
        UNKNOWN_FACES_DIR,
        UNKNOWN_VOICES_DIR,
        INITIAL_VIDEO_DIR,
        TRACKED_VIDEO_DIR,
        WORKFLOW_TEMP_DIR,
        LIVE_VIDEO_DIR,
        LOG_DIR,
        TRANSCRIPTS_DIR,
    )

    for directory in required_directories:
        directory.mkdir(parents=True, exist_ok=True)


_apply_runtime_paths(PROJECT_ROOT, DATA_ROOT)


__all__ = [
    "PROJECT_ROOT",
    "DATA_ROOT",
    "BASE_DIR",
    "DB_PATH",
    "INDEX_PATH",
    "KNOWN_FACES_DIR",
    "KNOWN_VOICES_DIR",
    "UNKNOWN_FACES_DIR",
    "UNKNOWN_VOICES_DIR",
    "INITIAL_VIDEO_DIR",
    "TRACKED_VIDEO_DIR",
    "WORKFLOW_TEMP_DIR",
    "LIVE_VIDEO_DIR",
    "LOG_DIR",
    "VOICE_DIAGNOSTICS_PATH",
    "TRANSCRIPTS_DIR",
    "initialize_runtime_paths",
]