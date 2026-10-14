from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sqlite3
from typing import Iterable

from .constants import DB_BUSY_TIMEOUT_SECONDS

MIGRATION_MARKER = ".legacy_migration_v1_complete"
PATH_REFERENCE_MIGRATION_MARKER = ".legacy_path_references_v1_complete"

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
    project_root = project_root.resolve()
    data_root = data_root.resolve()

    marker = data_root / MIGRATION_MARKER
    if not marker.exists():
        conflicts = _collect_conflicts(project_root, data_root)
        if conflicts:
            raise DataMigrationConflict(conflicts)

        for source_relative, destination_relative in _MIGRATION_MAP:
            source = project_root / source_relative
            destination = data_root / destination_relative
            _move_source(source, destination, project_root)

        _write_marker_atomically(marker)

    _migrate_path_references(project_root, data_root)


def _migrate_path_references(project_root: Path, data_root: Path) -> None:
    marker = data_root / PATH_REFERENCE_MIGRATION_MARKER
    if marker.exists():
        return

    db_path = data_root / "db" / "faces.db"
    if not db_path.exists():
        _write_marker_atomically(marker)
        return

    replacements = tuple(_legacy_prefix_replacements(project_root, data_root))
    connection = sqlite3.connect(db_path, timeout=DB_BUSY_TIMEOUT_SECONDS)
    try:
        connection.execute(f"PRAGMA busy_timeout={DB_BUSY_TIMEOUT_SECONDS * 1000}")
        connection.execute("BEGIN IMMEDIATE")
        for table_name, columns in _path_reference_columns():
            if not _table_exists(connection, table_name):
                continue
            existing_columns = set(_column_names(connection, table_name))
            for column_name in columns:
                if column_name not in existing_columns:
                    continue
                _rewrite_table_column(connection, table_name, column_name, replacements)
        connection.commit()
    except sqlite3.Error:
        connection.rollback()
        raise
    finally:
        connection.close()

    _write_marker_atomically(marker)


def _path_reference_columns() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (
        ("face_embeddings", ("image_path", "embedding_path")),
        ("unknown_tracks", ("image_path", "embedding_path")),
        ("unknown_samples", ("image_path", "embedding_path")),
        ("meetings", ("video_path",)),
        ("attendance_events", ("source_ref",)),
        ("voice_embeddings", ("embedding_path", "audio_path")),
        ("unknown_voices", ("embedding_path",)),
        ("unknown_voice_samples", ("embedding_path", "audio_path", "source_ref")),
    )


def _legacy_prefix_replacements(project_root: Path, data_root: Path) -> Iterable[tuple[Path, Path]]:
    return (
        (project_root / "known_faces", data_root / "artifacts" / "known_faces"),
        (project_root / "known_voices", data_root / "artifacts" / "known_voices"),
        (project_root / "unknown_faces", data_root / "artifacts" / "unknown_faces"),
        (project_root / "unknown_voices", data_root / "artifacts" / "unknown_voices"),
        (project_root / "initialVideo", data_root / "artifacts" / "initialVideo"),
        (project_root / "trackedVideo", data_root / "artifacts" / "trackedVideo"),
        (project_root / "live", data_root / "artifacts" / "live"),
        (project_root / "logs", data_root / "artifacts" / "logs"),
        (project_root / "transcripts", data_root / "transcripts"),
        (project_root / ".workflow-temp", data_root / "temp" / "workflow"),
    )


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _column_names(connection: sqlite3.Connection, table_name: str) -> list[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [str(row[1]) for row in rows]


def _rewrite_table_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    replacements: tuple[tuple[Path, Path], ...],
) -> None:
    select_sql = f"SELECT rowid, {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL"
    rows = connection.execute(select_sql).fetchall()
    updates: list[tuple[str, int]] = []
    for rowid, value in rows:
        if not isinstance(value, str):
            continue
        rewritten = _rewrite_legacy_path(value, replacements)
        if rewritten != value:
            updates.append((rewritten, int(rowid)))

    if updates:
        update_sql = f"UPDATE {table_name} SET {column_name} = ? WHERE rowid = ?"
        connection.executemany(update_sql, updates)


def _rewrite_legacy_path(value: str, replacements: tuple[tuple[Path, Path], ...]) -> str:
    candidate = Path(value)
    if not candidate.is_absolute():
        return value

    for source_prefix, destination_prefix in replacements:
        relative_suffix = _relative_if_under(candidate, source_prefix)
        if relative_suffix is not None:
            rewritten_path = destination_prefix / relative_suffix
            return _render_with_original_separators(rewritten_path, value)
    return value


def _render_with_original_separators(path: Path, original_value: str) -> str:
    has_forward_slash = "/" in original_value
    has_backslash = "\\" in original_value

    if has_forward_slash and not has_backslash:
        return path.as_posix()
    if has_backslash and not has_forward_slash:
        return str(path)
    if has_forward_slash and has_backslash:
        first_forward_index = original_value.find("/")
        first_backslash_index = original_value.find("\\")
        if first_forward_index != -1 and (
            first_backslash_index == -1 or first_forward_index < first_backslash_index
        ):
            return path.as_posix()

    return str(path)


def _relative_if_under(candidate: Path, prefix: Path) -> Path | None:
    try:
        return candidate.relative_to(prefix)
    except ValueError:
        if os.name != "nt":
            return None

    try:
        candidate_parts = [part.casefold() for part in candidate.parts]
        prefix_parts = [part.casefold() for part in prefix.parts]
    except Exception:
        return None

    if len(candidate_parts) < len(prefix_parts):
        return None
    if candidate_parts[: len(prefix_parts)] != prefix_parts:
        return None

    return Path(*candidate.parts[len(prefix_parts) :])

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

    if source.is_symlink():
        conflicts.append(_Conflict(source_relative, destination_relative, "symbolic links are not supported"))
        return

    if _destination_parent_is_file(destination):
        conflicts.append(_Conflict(source_relative, destination_relative, "destination parent is a file"))
        return

    if not _is_same_filesystem(source, destination):
        conflicts.append(_Conflict(source_relative, destination_relative, "source and destination are on different filesystems"))
        return

    if source.is_file():
        if destination.exists():
            if destination.is_symlink():
                conflicts.append(_Conflict(source_relative, destination_relative, "symbolic links are not supported"))
                return

            if destination.is_dir():
                conflicts.append(_Conflict(source_relative, destination_relative, "destination is a directory"))
                return

            if not _is_same_file(source, destination):
                conflicts.append(_Conflict(source_relative, destination_relative, "destination file already exists"))
        return

    if source.is_dir():
        if destination.exists() and destination.is_symlink():
            conflicts.append(_Conflict(source_relative, destination_relative, "symbolic links are not supported"))
            return

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

    if source.is_symlink():
        raise DataMigrationConflict([_Conflict(source.relative_to(project_root), destination, "symbolic links are not supported")])

    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _publish_file_no_clobber(source, destination)
        _remove_empty_parents(source.parent, project_root)
        return

    if source.is_dir():
        _move_directory(source, destination, project_root)


def _move_directory(source_dir: Path, destination_dir: Path, project_root: Path) -> None:
    if not source_dir.exists():
        return

    if source_dir.is_symlink():
        raise DataMigrationConflict([_Conflict(source_dir.relative_to(project_root), destination_dir, "symbolic links are not supported")])

    destination_dir.mkdir(parents=True, exist_ok=True)

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


def _publish_file_no_clobber(source: Path, destination: Path) -> None:
    if destination.exists():
        if _is_same_file(source, destination):
            source.unlink()
            return
        raise DataMigrationConflict([_Conflict(source, destination, "destination file already exists")])

    os.link(source, destination)
    source.unlink()


def _is_same_file(source: Path, destination: Path) -> bool:
    try:
        return source.exists() and destination.exists() and source.samefile(destination)
    except OSError:
        return False


def _is_same_filesystem(source: Path, destination: Path) -> bool:
    source_stat = os.stat(source)
    destination_probe = destination if destination.exists() else _first_existing_parent(destination)
    destination_stat = os.stat(destination_probe)
    return source_stat.st_dev == destination_stat.st_dev


def _first_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            break
        current = parent
    return current
