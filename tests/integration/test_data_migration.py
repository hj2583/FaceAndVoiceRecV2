from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import sys

import pytest

from face_voice_rec.config.migration import DataMigrationConflict, migrate_legacy_data
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures.runtime_tree import RuntimeTree


PATH_REFERENCE_MIGRATION_MARKER = ".legacy_path_references_v1_complete"


def _create_path_reference_test_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            CREATE TABLE face_embeddings (
                id INTEGER PRIMARY KEY,
                image_path TEXT,
                embedding_path TEXT
            );
            CREATE TABLE unknown_tracks (
                id INTEGER PRIMARY KEY,
                image_path TEXT,
                embedding_path TEXT
            );
            CREATE TABLE unknown_samples (
                id INTEGER PRIMARY KEY,
                image_path TEXT,
                embedding_path TEXT
            );
            CREATE TABLE meetings (
                id INTEGER PRIMARY KEY,
                video_path TEXT
            );
            CREATE TABLE attendance_events (
                id INTEGER PRIMARY KEY,
                source_ref TEXT
            );
            CREATE TABLE voice_embeddings (
                id INTEGER PRIMARY KEY,
                embedding_path TEXT,
                audio_path TEXT
            );
            CREATE TABLE unknown_voices (
                id INTEGER PRIMARY KEY,
                embedding_path TEXT
            );
            CREATE TABLE unknown_voice_samples (
                id INTEGER PRIMARY KEY,
                embedding_path TEXT,
                audio_path TEXT,
                source_ref TEXT
            );
            """
        )
        connection.commit()
    finally:
        connection.close()


def _create_empty_sqlite_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.close()


def _read_table(connection: sqlite3.Connection, table: str, columns: str) -> list[tuple[str, ...]]:
    cursor = connection.execute(f"SELECT {columns} FROM {table} ORDER BY id")
    return [tuple(row) for row in cursor.fetchall()]


@pytest.fixture
def runtime_tree(tmp_path: Path) -> RuntimeTree:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    return RuntimeTree(project_root=project_root)


def test_moves_legacy_sources_to_data_layout(runtime_tree: RuntimeTree) -> None:
    _create_empty_sqlite_db(runtime_tree.legacy_path("faces.db"))
    runtime_tree.write_legacy("faces.db-wal", b"wal")
    runtime_tree.write_legacy("faces.db-shm", b"shm")
    runtime_tree.write_legacy("face_index.npz", b"index")
    runtime_tree.write_legacy("known_faces/1/sample.jpg", b"face")
    runtime_tree.write_legacy("known_voices/63/voice.npy", b"voice")
    runtime_tree.write_legacy("unknown_faces/track_1.npy", b"unknown-face")
    runtime_tree.write_legacy("unknown_voices/track_1.npy", b"unknown-voice")
    runtime_tree.write_legacy("initialVideo/input.mp4", b"input")
    runtime_tree.write_legacy("trackedVideo/output.mp4", b"output")
    runtime_tree.write_legacy("live/frame_1.jpg", b"live")
    runtime_tree.write_legacy("logs/session.json", b"log")
    runtime_tree.write_legacy("transcripts/1/transcript.json", b"transcript")
    runtime_tree.write_legacy(".workflow-temp/work/item.txt", b"workflow")

    migrate_legacy_data(runtime_tree.project_root, runtime_tree.data_root)

    assert runtime_tree.data_path("db/faces.db").exists()
    assert runtime_tree.data_path("artifacts/face_index.npz").read_bytes() == b"index"
    assert runtime_tree.data_path("artifacts/known_faces/1/sample.jpg").read_bytes() == b"face"
    assert runtime_tree.data_path("artifacts/known_voices/63/voice.npy").read_bytes() == b"voice"
    assert runtime_tree.data_path("artifacts/unknown_faces/track_1.npy").read_bytes() == b"unknown-face"
    assert runtime_tree.data_path("artifacts/unknown_voices/track_1.npy").read_bytes() == b"unknown-voice"
    assert runtime_tree.data_path("artifacts/initialVideo/input.mp4").read_bytes() == b"input"
    assert runtime_tree.data_path("artifacts/trackedVideo/output.mp4").read_bytes() == b"output"
    assert runtime_tree.data_path("artifacts/live/frame_1.jpg").read_bytes() == b"live"
    assert runtime_tree.data_path("artifacts/logs/session.json").read_bytes() == b"log"
    assert runtime_tree.data_path("transcripts/1/transcript.json").read_bytes() == b"transcript"
    assert runtime_tree.data_path("temp/workflow/work/item.txt").read_bytes() == b"workflow"
    assert runtime_tree.data_path(".legacy_migration_v1_complete").exists()


def test_directory_merge_moves_only_missing_children(runtime_tree: RuntimeTree) -> None:
    runtime_tree.write_legacy("known_faces/1/new.jpg", b"new")
    runtime_tree.write_data("artifacts/known_faces/2/existing.jpg", b"existing")

    migrate_legacy_data(runtime_tree.project_root, runtime_tree.data_root)

    assert runtime_tree.data_path("artifacts/known_faces/1/new.jpg").read_bytes() == b"new"
    assert runtime_tree.data_path("artifacts/known_faces/2/existing.jpg").read_bytes() == b"existing"


def test_idempotent_when_marker_exists(runtime_tree: RuntimeTree) -> None:
    runtime_tree.write_data(".legacy_migration_v1_complete", b"done")
    runtime_tree.write_legacy("known_faces/1/sample.jpg", b"legacy")

    migrate_legacy_data(runtime_tree.project_root, runtime_tree.data_root)

    assert runtime_tree.legacy_path("known_faces/1/sample.jpg").read_bytes() == b"legacy"
    assert runtime_tree.data_path(".legacy_migration_v1_complete").read_bytes() == b"done"


def test_conflict_leaves_all_legacy_sources_untouched(runtime_tree: RuntimeTree) -> None:
    source = runtime_tree.write_legacy("known_faces/1/sample.jpg", b"legacy")
    destination = runtime_tree.write_data("artifacts/known_faces/1/sample.jpg", b"existing")
    untouched = runtime_tree.write_legacy("known_voices/63/voice.npy", b"voice")

    with pytest.raises(DataMigrationConflict):
        migrate_legacy_data(runtime_tree.project_root, runtime_tree.data_root)

    assert source.read_bytes() == b"legacy"
    assert destination.read_bytes() == b"existing"
    assert untouched.read_bytes() == b"voice"
    assert not runtime_tree.data_path(".legacy_migration_v1_complete").exists()


def test_conflict_detected_for_all_sources_before_any_move(runtime_tree: RuntimeTree) -> None:
    known_faces_src = runtime_tree.write_legacy("known_faces/1/sample.jpg", b"legacy-face")
    known_faces_dst = runtime_tree.write_data("artifacts/known_faces/1/sample.jpg", b"existing-face")
    known_voices_src = runtime_tree.write_legacy("known_voices/63/voice.npy", b"legacy-voice")
    known_voices_dst = runtime_tree.write_data("artifacts/known_voices/63/voice.npy", b"existing-voice")

    with pytest.raises(DataMigrationConflict) as exc_info:
        migrate_legacy_data(runtime_tree.project_root, runtime_tree.data_root)

    message = str(exc_info.value)
    assert "known_faces/1/sample.jpg" in message
    assert "known_voices/63/voice.npy" in message
    assert known_faces_src.read_bytes() == b"legacy-face"
    assert known_faces_dst.read_bytes() == b"existing-face"
    assert known_voices_src.read_bytes() == b"legacy-voice"
    assert known_voices_dst.read_bytes() == b"existing-voice"


def test_retry_after_move_failure_skips_already_moved_sources(runtime_tree: RuntimeTree, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_tree.write_legacy("face_index.npz", b"index")
    runtime_tree.write_legacy("known_faces/1/sample.jpg", b"face")

    import face_voice_rec.config.migration as migration

    original_publish = migration._publish_file_no_clobber
    calls: list[tuple[Path, Path]] = []

    def flaky_publish(source: Path, destination: Path):
        calls.append((source, destination))
        if len(calls) == 2:
            raise OSError("injected publish failure")
        return original_publish(source, destination)

    monkeypatch.setattr(migration, "_publish_file_no_clobber", flaky_publish)

    with pytest.raises(OSError, match="injected publish failure"):
        migrate_legacy_data(runtime_tree.project_root, runtime_tree.data_root)

    assert not runtime_tree.data_path(".legacy_migration_v1_complete").exists()

    monkeypatch.setattr(migration, "_publish_file_no_clobber", original_publish)

    migrate_legacy_data(runtime_tree.project_root, runtime_tree.data_root)

    assert runtime_tree.data_path("artifacts/face_index.npz").read_bytes() == b"index"
    assert runtime_tree.data_path("artifacts/known_faces/1/sample.jpg").read_bytes() == b"face"
    assert runtime_tree.data_path(".legacy_migration_v1_complete").exists()


def test_conflict_when_destination_file_appears_after_preflight(runtime_tree: RuntimeTree, monkeypatch: pytest.MonkeyPatch) -> None:
    source = runtime_tree.write_legacy("face_index.npz", b"source-index")
    destination = runtime_tree.data_path("artifacts/face_index.npz")

    import face_voice_rec.config.migration as migration

    original_collect_conflicts = migration._collect_conflicts

    def race_collect_conflicts(project_root, data_root):
        conflicts = original_collect_conflicts(project_root, data_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"existing-index")
        return conflicts

    monkeypatch.setattr(migration, "_collect_conflicts", race_collect_conflicts)

    with pytest.raises(Exception):
        migrate_legacy_data(runtime_tree.project_root, runtime_tree.data_root)

    assert source.read_bytes() == b"source-index"
    assert destination.read_bytes() == b"existing-index"
    assert not runtime_tree.data_path(".legacy_migration_v1_complete").exists()


def test_retry_handles_same_file_after_destination_publish(runtime_tree: RuntimeTree) -> None:
    source = runtime_tree.write_legacy("face_index.npz", b"index")
    destination = runtime_tree.data_path("artifacts/face_index.npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)

    migrate_legacy_data(runtime_tree.project_root, runtime_tree.data_root)

    assert not source.exists()
    assert destination.read_bytes() == b"index"
    assert runtime_tree.data_path(".legacy_migration_v1_complete").exists()


def test_cross_filesystem_rejected_before_any_move(runtime_tree: RuntimeTree, monkeypatch: pytest.MonkeyPatch) -> None:
    source = runtime_tree.write_legacy("face_index.npz", b"index")

    import face_voice_rec.config.migration as migration

    publishes: list[tuple[Path, Path]] = []
    original_publish = migration._publish_file_no_clobber

    def spying_publish(source_path: Path, destination_path: Path):
        publishes.append((source_path, destination_path))
        return original_publish(source_path, destination_path)

    monkeypatch.setattr(migration, "_publish_file_no_clobber", spying_publish)
    monkeypatch.setattr(migration, "_is_same_filesystem", lambda *_args, **_kwargs: False, raising=False)

    with pytest.raises(DataMigrationConflict):
        migrate_legacy_data(runtime_tree.project_root, runtime_tree.data_root)

    assert source.read_bytes() == b"index"
    assert publishes == []
    assert not runtime_tree.data_path("artifacts/face_index.npz").exists()
    assert not runtime_tree.data_path(".legacy_migration_v1_complete").exists()


def test_path_reference_migration_runs_after_v1_marker_and_is_idempotent(runtime_tree: RuntimeTree) -> None:
    runtime_tree.write_data(".legacy_migration_v1_complete", b"done\n")
    db_path = runtime_tree.data_path("db/faces.db")
    _create_path_reference_test_db(db_path)

    legacy_known_face_image = str(runtime_tree.project_root / "known_faces" / "1" / "sample.jpg")
    legacy_known_face_embedding = str(runtime_tree.project_root / "known_faces" / "1" / "sample.npy")
    legacy_unknown_track_image = str(runtime_tree.project_root / "unknown_faces" / "track_1.jpg")
    legacy_unknown_track_embedding = str(runtime_tree.project_root / "unknown_faces" / "track_1.npy")
    legacy_unknown_sample_image = str(runtime_tree.project_root / "unknown_faces" / "sample_1.jpg")
    legacy_unknown_sample_embedding = str(runtime_tree.project_root / "unknown_faces" / "sample_1.npy")
    legacy_meeting_video = str(runtime_tree.project_root / "trackedVideo" / "meeting.mp4")
    legacy_attendance_ref = str(runtime_tree.project_root / "initialVideo" / "source.mp4")
    legacy_voice_embedding = str(runtime_tree.project_root / "known_voices" / "63" / "voice.npy")
    legacy_voice_audio = str(runtime_tree.project_root / "known_voices" / "63" / "voice.wav")
    legacy_unknown_voice_embedding = str(runtime_tree.project_root / "unknown_voices" / "voice_track.npy")
    legacy_unknown_voice_sample_embedding = str(runtime_tree.project_root / "unknown_voices" / "sample_track.npy")
    legacy_unknown_voice_sample_audio = str(runtime_tree.project_root / "unknown_voices" / "sample_track.wav")
    legacy_unknown_voice_source_ref = str(runtime_tree.project_root / "logs" / "voice_source.json")

    external_path = r"C:\external\archive\audio.wav"
    relative_path = "unknown_faces/relative.jpg"

    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "INSERT INTO face_embeddings(id, image_path, embedding_path) VALUES (1, ?, ?)",
            (legacy_known_face_image, legacy_known_face_embedding),
        )
        connection.execute(
            "INSERT INTO unknown_tracks(id, image_path, embedding_path) VALUES (1, ?, ?)",
            (legacy_unknown_track_image, legacy_unknown_track_embedding),
        )
        connection.execute(
            "INSERT INTO unknown_samples(id, image_path, embedding_path) VALUES (1, ?, ?)",
            (legacy_unknown_sample_image, legacy_unknown_sample_embedding),
        )
        connection.execute("INSERT INTO meetings(id, video_path) VALUES (1, ?)", (legacy_meeting_video,))
        connection.execute(
            "INSERT INTO attendance_events(id, source_ref) VALUES (1, ?)",
            (legacy_attendance_ref,),
        )
        connection.execute(
            "INSERT INTO voice_embeddings(id, embedding_path, audio_path) VALUES (1, ?, ?)",
            (legacy_voice_embedding, legacy_voice_audio),
        )
        connection.execute(
            "INSERT INTO unknown_voices(id, embedding_path) VALUES (1, ?)",
            (legacy_unknown_voice_embedding,),
        )
        connection.execute(
            "INSERT INTO unknown_voice_samples(id, embedding_path, audio_path, source_ref) VALUES (1, ?, ?, ?)",
            (legacy_unknown_voice_sample_embedding, legacy_unknown_voice_sample_audio, legacy_unknown_voice_source_ref),
        )

        connection.execute(
            "INSERT INTO face_embeddings(id, image_path, embedding_path) VALUES (2, ?, ?)",
            (external_path, relative_path),
        )
        connection.commit()
    finally:
        connection.close()

    migrate_legacy_data(runtime_tree.project_root, runtime_tree.data_root)

    connection = sqlite3.connect(db_path)
    try:
        assert _read_table(connection, "face_embeddings", "image_path, embedding_path") == [
            (
                str(runtime_tree.data_path("artifacts/known_faces/1/sample.jpg")),
                str(runtime_tree.data_path("artifacts/known_faces/1/sample.npy")),
            ),
            (external_path, relative_path),
        ]
        assert _read_table(connection, "unknown_tracks", "image_path, embedding_path") == [
            (
                str(runtime_tree.data_path("artifacts/unknown_faces/track_1.jpg")),
                str(runtime_tree.data_path("artifacts/unknown_faces/track_1.npy")),
            )
        ]
        assert _read_table(connection, "unknown_samples", "image_path, embedding_path") == [
            (
                str(runtime_tree.data_path("artifacts/unknown_faces/sample_1.jpg")),
                str(runtime_tree.data_path("artifacts/unknown_faces/sample_1.npy")),
            )
        ]
        assert _read_table(connection, "meetings", "video_path") == [
            (str(runtime_tree.data_path("artifacts/trackedVideo/meeting.mp4")),)
        ]
        assert _read_table(connection, "attendance_events", "source_ref") == [
            (str(runtime_tree.data_path("artifacts/initialVideo/source.mp4")),)
        ]
        assert _read_table(connection, "voice_embeddings", "embedding_path, audio_path") == [
            (
                str(runtime_tree.data_path("artifacts/known_voices/63/voice.npy")),
                str(runtime_tree.data_path("artifacts/known_voices/63/voice.wav")),
            )
        ]
        assert _read_table(connection, "unknown_voices", "embedding_path") == [
            (str(runtime_tree.data_path("artifacts/unknown_voices/voice_track.npy")),)
        ]
        assert _read_table(connection, "unknown_voice_samples", "embedding_path, audio_path, source_ref") == [
            (
                str(runtime_tree.data_path("artifacts/unknown_voices/sample_track.npy")),
                str(runtime_tree.data_path("artifacts/unknown_voices/sample_track.wav")),
                str(runtime_tree.data_path("artifacts/logs/voice_source.json")),
            )
        ]
    finally:
        connection.close()

    assert runtime_tree.data_path(PATH_REFERENCE_MIGRATION_MARKER).exists()

    pre_second_run_bytes = db_path.read_bytes()
    migrate_legacy_data(runtime_tree.project_root, runtime_tree.data_root)
    post_second_run_bytes = db_path.read_bytes()
    assert pre_second_run_bytes == post_second_run_bytes


def test_path_reference_migration_missing_db_is_noop(runtime_tree: RuntimeTree) -> None:
    runtime_tree.write_data(".legacy_migration_v1_complete", b"done\n")

    migrate_legacy_data(runtime_tree.project_root, runtime_tree.data_root)

    assert runtime_tree.data_path(".legacy_migration_v1_complete").exists()
    assert runtime_tree.data_path(PATH_REFERENCE_MIGRATION_MARKER).exists()


def test_path_reference_migration_sqlite_error_does_not_write_marker(runtime_tree: RuntimeTree) -> None:
    runtime_tree.write_data(".legacy_migration_v1_complete", b"done\n")
    db_path = runtime_tree.data_path("db/faces.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"not-a-valid-sqlite-db")

    with pytest.raises(sqlite3.Error):
        migrate_legacy_data(runtime_tree.project_root, runtime_tree.data_root)

    assert not runtime_tree.data_path(PATH_REFERENCE_MIGRATION_MARKER).exists()


def test_path_reference_migration_preserves_posix_separator_style_for_attendance_source_ref(
    runtime_tree: RuntimeTree,
) -> None:
    runtime_tree.write_data(".legacy_migration_v1_complete", b"done\n")
    db_path = runtime_tree.data_path("db/faces.db")
    _create_path_reference_test_db(db_path)

    legacy_posix_attendance_ref = (
        f"{runtime_tree.project_root.as_posix()}/initialVideo/source.mp4"
    )

    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "INSERT INTO attendance_events(id, source_ref) VALUES (1, ?)",
            (legacy_posix_attendance_ref,),
        )
        connection.commit()
    finally:
        connection.close()

    migrate_legacy_data(runtime_tree.project_root, runtime_tree.data_root)

    connection = sqlite3.connect(db_path)
    try:
        assert _read_table(connection, "attendance_events", "source_ref") == [
            (runtime_tree.data_path("artifacts/initialVideo/source.mp4").as_posix(),)
        ]
    finally:
        connection.close()
