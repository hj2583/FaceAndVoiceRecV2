from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

MIGRATION_MARKER = ".legacy_migration_v1_complete"

_MIGRATION_MAP: tuple[tuple[Path, Path], ...] = (
    (Path("faces.db"), Path("db/faces.db")),
    (Path("faces.db-wal"), Path("db/faces.db-wal")),
    (Path("faces.db-shm"), Path("db/faces.db-shm")),
    (Path("face_index.npz"), Path("artifacts/face_index.npz")),
    (Path("known_faces"), Path("artifacts/known_faces")),
    (Path("known_voices"), Path("artifacts/known_voices")),
    (Path("unknown_faces"), Path("artifacts/unknown_faces")),
    (Path("unknown_voices"), Path("artifacts/unknown_voices")),
    (Path("initialVideo"), Path("artifacts/initialVideo")),
    (Path("trackedVideo"), Path("artifacts/trackedVideo")),
    (Path("live"), Path("artifacts/live")),
    (Path("logs"), Path("artifacts/logs")),
    (Path("transcripts"), Path("transcripts")),
    (Path(".workflow-temp"), Path("temp/workflow")),
)


@dataclass(frozen=True)
class _Conflict:
    source: Path
    destination: Path
    reason: str


class DataMigrationConflict(RuntimeError):
    def __init__(self, conflicts: list[_Conflict]):
        self.conflicts = tuple(conflicts)
        lines = ["Legacy data migration conflicts detected:"]
        for conflict in conflicts:
            lines.append(
                f"- {conflict.source.as_posix()} -> {conflict.destination.as_posix()} ({conflict.reason})"
            )
        super().__init__("\n".join(lines))


def migrate_legacy_data(project_root: Path, data_root: Path) -> None:
    marker = data_root / MIGRATION_MARKER
    if marker.exists():
        return

    project_root = project_root.resolve()
    data_root = data_root.resolve()

    conflicts = _collect_conflicts(project_root, data_root)
    if conflicts:
        raise DataMigrationConflict(conflicts)

    for source_relative, destination_relative in _MIGRATION_MAP:
        source = project_root / source_relative
        destination = data_root / destination_relative
        _move_source(source, destination, project_root)

    _write_marker_atomically(marker)


def _collect_conflicts(project_root: Path, data_root: Path) -> list[_Conflict]:
    conflicts: list[_Conflict] = []
    for source_relative, destination_relative in _MIGRATION_MAP:
        source = project_root / source_relative
        destination = data_root / destination_relative
        _collect_conflicts_for_path(source, destination, source_relative, destination_relative, conflicts)
    return conflicts


def _collect_conflicts_for_path(
    source: Path,
    destination: Path,
    source_relative: Path,
    destination_relative: Path,
    conflicts: list[_Conflict],
) -> None:
    if not source.exists():
        return

    if _destination_parent_is_file(destination):
        conflicts.append(_Conflict(source_relative, destination_relative, "destination parent is a file"))
        return

    if source.is_file():
        if destination.exists():
            reason = "destination is a directory" if destination.is_dir() else "destination file already exists"
            conflicts.append(_Conflict(source_relative, destination_relative, reason))
        return

    if source.is_dir():
        if destination.exists() and destination.is_file():
            conflicts.append(_Conflict(source_relative, destination_relative, "source is directory but destination is file"))
            return

        if destination.exists() and destination.is_dir():
            for child in sorted(source.iterdir(), key=lambda item: item.name):
                child_source_relative = source_relative / child.name
                child_destination_relative = destination_relative / child.name
                _collect_conflicts_for_path(
                    child,
                    destination / child.name,
                    child_source_relative,
                    child_destination_relative,
                    conflicts,
                )


def _destination_parent_is_file(destination: Path) -> bool:
    parent = destination.parent
    while True:
        if parent.exists() and parent.is_file():
            return True
        next_parent = parent.parent
        if next_parent == parent:
            return False
        parent = next_parent


def _move_source(source: Path, destination: Path, project_root: Path) -> None:
    if not source.exists():
        return

    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        _remove_empty_parents(source.parent, project_root)
        return

    if source.is_dir():
        _move_directory(source, destination, project_root)


def _move_directory(source_dir: Path, destination_dir: Path, project_root: Path) -> None:
    if not source_dir.exists():
        return

    if not destination_dir.exists():
        destination_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_dir), str(destination_dir))
        _remove_empty_parents(source_dir.parent, project_root)
        return

    for child in sorted(source_dir.iterdir(), key=lambda item: item.name):
        _move_source(child, destination_dir / child.name, project_root)

    if source_dir.exists() and source_dir.is_dir() and not any(source_dir.iterdir()):
        source_dir.rmdir()
        _remove_empty_parents(source_dir.parent, project_root)


def _remove_empty_parents(start: Path, stop: Path) -> None:
    current = start
    while current != stop and current.exists() and current.is_dir() and not any(current.iterdir()):
        current.rmdir()
        current = current.parent


def _write_marker_atomically(marker: Path) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    temp_marker = marker.with_name(f"{marker.name}.tmp")
    temp_marker.write_text("complete\n", encoding="utf-8")
    temp_marker.replace(marker)
