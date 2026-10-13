from __future__ import annotations

from pathlib import Path
import sys

import pytest

from face_voice_rec.config.migration import DataMigrationConflict, migrate_legacy_data
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures.runtime_tree import RuntimeTree


@pytest.fixture
def runtime_tree(tmp_path: Path) -> RuntimeTree:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    return RuntimeTree(project_root=project_root)


def test_moves_legacy_sources_to_data_layout(runtime_tree: RuntimeTree) -> None:
    runtime_tree.write_legacy("faces.db", b"db")
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

    assert runtime_tree.data_path("db/faces.db").read_bytes() == b"db"
    assert runtime_tree.data_path("db/faces.db-wal").read_bytes() == b"wal"
    assert runtime_tree.data_path("db/faces.db-shm").read_bytes() == b"shm"
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

    original_move = migration.shutil.move
    calls: list[tuple[Path, Path]] = []

    def flaky_move(source: Path, destination: Path):
        calls.append((Path(source), Path(destination)))
        if len(calls) == 2:
            raise OSError("injected move failure")
        return original_move(source, destination)

    monkeypatch.setattr(migration.shutil, "move", flaky_move)

    with pytest.raises(OSError, match="injected move failure"):
        migrate_legacy_data(runtime_tree.project_root, runtime_tree.data_root)

    assert not runtime_tree.data_path(".legacy_migration_v1_complete").exists()

    monkeypatch.setattr(migration.shutil, "move", original_move)

    migrate_legacy_data(runtime_tree.project_root, runtime_tree.data_root)

    assert runtime_tree.data_path("artifacts/face_index.npz").read_bytes() == b"index"
    assert runtime_tree.data_path("artifacts/known_faces/1/sample.jpg").read_bytes() == b"face"
    assert runtime_tree.data_path(".legacy_migration_v1_complete").exists()
