from __future__ import annotations

from pathlib import Path

from .migration import migrate_legacy_data


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"

# Compatibility alias for legacy callers.
BASE_DIR = PROJECT_ROOT


DB_PATH = DATA_ROOT / "db" / "faces.db"
INDEX_PATH = DATA_ROOT / "artifacts" / "face_index.npz"
KNOWN_FACES_DIR = DATA_ROOT / "artifacts" / "known_faces"
KNOWN_VOICES_DIR = DATA_ROOT / "artifacts" / "known_voices"
UNKNOWN_FACES_DIR = DATA_ROOT / "artifacts" / "unknown_faces"
UNKNOWN_VOICES_DIR = DATA_ROOT / "artifacts" / "unknown_voices"
INITIAL_VIDEO_DIR = DATA_ROOT / "artifacts" / "initialVideo"
TRACKED_VIDEO_DIR = DATA_ROOT / "artifacts" / "trackedVideo"
WORKFLOW_TEMP_DIR = DATA_ROOT / "temp" / "workflow"
LIVE_VIDEO_DIR = DATA_ROOT / "artifacts" / "live"
LOG_DIR = DATA_ROOT / "artifacts" / "logs"
VOICE_DIAGNOSTICS_PATH = DATA_ROOT / "artifacts" / "logs" / "speaker_diagnostics.json"
TRANSCRIPTS_DIR = DATA_ROOT / "transcripts"


def initialize_runtime_paths(
    project_root: Path | None = None,
    data_root: Path | None = None,
) -> None:
    resolved_project_root = (project_root or PROJECT_ROOT).resolve()
    resolved_data_root = (data_root or DATA_ROOT).resolve()
    runtime_db_path = resolved_data_root / "db" / "faces.db"
    runtime_index_path = resolved_data_root / "artifacts" / "face_index.npz"
    runtime_known_faces_dir = resolved_data_root / "artifacts" / "known_faces"
    runtime_known_voices_dir = resolved_data_root / "artifacts" / "known_voices"
    runtime_unknown_faces_dir = resolved_data_root / "artifacts" / "unknown_faces"
    runtime_unknown_voices_dir = resolved_data_root / "artifacts" / "unknown_voices"
    runtime_initial_video_dir = resolved_data_root / "artifacts" / "initialVideo"
    runtime_tracked_video_dir = resolved_data_root / "artifacts" / "trackedVideo"
    runtime_workflow_temp_dir = resolved_data_root / "temp" / "workflow"
    runtime_live_video_dir = resolved_data_root / "artifacts" / "live"
    runtime_log_dir = resolved_data_root / "artifacts" / "logs"
    runtime_transcripts_dir = resolved_data_root / "transcripts"

    migrate_legacy_data(resolved_project_root, resolved_data_root)

    required_directories = (
        runtime_db_path.parent,
        runtime_index_path.parent,
        runtime_known_faces_dir,
        runtime_known_voices_dir,
        runtime_unknown_faces_dir,
        runtime_unknown_voices_dir,
        runtime_initial_video_dir,
        runtime_tracked_video_dir,
        runtime_workflow_temp_dir,
        runtime_live_video_dir,
        runtime_log_dir,
        runtime_transcripts_dir,
    )

    for directory in required_directories:
        directory.mkdir(parents=True, exist_ok=True)


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