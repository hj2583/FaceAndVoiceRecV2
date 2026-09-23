from pathlib import Path

import database


def _setup_database(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "faces.db")
    database.init_db()


def _create_unknown(tmp_path, label, quality, resolved=False):
    image_path = tmp_path / f"{label}.jpg"
    embedding_path = tmp_path / f"{label}.npy"
    image_path.write_bytes(b"image")
    embedding_path.write_bytes(b"embedding")
    unknown_id = database.create_unknown(label, image_path, embedding_path)
    database.add_unknown_sample(
        unknown_id,
        image_path,
        embedding_path,
        quality=quality,
    )
    if resolved:
        person_id = database.create_person(f"Person {label}")
        database.resolve_unknown(unknown_id, person_id)
    return unknown_id, image_path, embedding_path


def test_delete_unknown_removes_unresolved_record_samples_and_files(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    unknown_id, image_path, embedding_path = _create_unknown(tmp_path, "unknown", 0.2)

    assert database.delete_unknown(unknown_id) is True

    assert not image_path.exists()
    assert not embedding_path.exists()
    assert database.list_unknown_samples(unknown_id) == []
    assert database.list_unknowns(include_resolved=True) == []


def test_delete_low_quality_unknowns_only_removes_unresolved_records(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    low_id, low_image, low_embedding = _create_unknown(tmp_path, "low", 0.39)
    good_id, good_image, good_embedding = _create_unknown(tmp_path, "good", 0.4)
    labeled_id, labeled_image, labeled_embedding = _create_unknown(
        tmp_path,
        "labeled",
        0.1,
        resolved=True,
    )

    deleted = database.delete_low_quality_unknowns(0.4)

    assert deleted == [low_id]
    assert not low_image.exists()
    assert not low_embedding.exists()
    assert good_image.exists()
    assert good_embedding.exists()
    assert labeled_image.exists()
    assert labeled_embedding.exists()
    assert {row[0] for row in database.list_unknowns(include_resolved=True)} == {
        good_id,
        labeled_id,
    }
