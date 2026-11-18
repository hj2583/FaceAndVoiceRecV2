from sqlalchemy import create_engine, select

from face_voice_rec.storage import sqlite_postgres_migration as migration
from face_voice_rec.storage.relational_schema import metadata


def _create_schema(path):
    engine = create_engine(f"sqlite:///{path.as_posix()}")
    metadata.create_all(engine)
    return engine


def test_copy_preserves_ids_and_foreign_keys(tmp_path, monkeypatch):
    source_path = tmp_path / "faces.db"
    target_path = tmp_path / "target.db"
    source_engine = _create_schema(source_path)
    target_engine = _create_schema(target_path)
    with source_engine.begin() as connection:
        connection.execute(
            metadata.tables["persons"].insert(),
            {
                "person_id": 12,
                "name": "Ada",
                "created_at": "2026-10-01T00:00:00+00:00",
                "updated_at": "2026-10-01T00:00:00+00:00",
            },
        )
        connection.execute(
            metadata.tables["recognition_logs"].insert(),
            {
                "log_id": 54,
                "timestamp_sec": 1.5,
                "person_id": 12,
                "person_name": "Ada",
                "confidence": 0.9,
                "source": "test",
            },
        )
    original_create_engine = migration.create_engine
    monkeypatch.setattr(
        migration,
        "create_engine",
        lambda url: source_engine if url.startswith("sqlite:") else target_engine,
    )

    copied = migration.copy_sqlite_database(
        source_path,
        "postgresql+psycopg://unused/unused",
    )

    assert copied["persons"] == 1
    assert copied["recognition_logs"] == 1
    with target_engine.connect() as connection:
        assert connection.execute(select(metadata.tables["persons"])).mappings().all()[0]["person_id"] == 12
        assert connection.execute(
            select(metadata.tables["recognition_logs"])
        ).mappings().all()[0]["person_id"] == 12
    monkeypatch.setattr(migration, "create_engine", original_create_engine)
    source_engine.dispose()
    target_engine.dispose()


def test_copy_supports_legacy_table_missing_nullable_column(tmp_path, monkeypatch):
    source_path = tmp_path / "legacy_faces.db"
    target_path = tmp_path / "target.db"
    source_engine = _create_schema(source_path)
    target_engine = _create_schema(target_path)
    with source_engine.begin() as connection:
        connection.exec_driver_sql("ALTER TABLE audio_logs DROP COLUMN source_ref")
        connection.exec_driver_sql(
            "INSERT INTO audio_logs "
            "(log_id, start_time, end_time, source, transcript) "
            "VALUES (1, 0.0, 1.0, 'test', 'hello')"
        )
    monkeypatch.setattr(
        migration,
        "create_engine",
        lambda url: source_engine if url.startswith("sqlite:") else target_engine,
    )

    copied = migration.copy_sqlite_database(
        source_path,
        "postgresql+psycopg://unused/unused",
    )

    assert copied["audio_logs"] == 1
    with target_engine.connect() as connection:
        row = connection.execute(select(metadata.tables["audio_logs"])).mappings().one()
    assert row["transcript"] == "hello"
    assert row["source_ref"] is None
    source_engine.dispose()
    target_engine.dispose()


def test_copy_refuses_nonempty_target_without_overwriting(tmp_path, monkeypatch):
    source_path = tmp_path / "faces.db"
    target_path = tmp_path / "target.db"
    source_engine = _create_schema(source_path)
    target_engine = _create_schema(target_path)
    with source_engine.begin() as connection:
        connection.execute(
            metadata.tables["persons"].insert(),
            {
                "person_id": 1,
                "name": "Source",
                "created_at": "now",
                "updated_at": "now",
            },
        )
    with target_engine.begin() as connection:
        connection.execute(
            metadata.tables["persons"].insert(),
            {
                "person_id": 2,
                "name": "Existing",
                "created_at": "now",
                "updated_at": "now",
            },
        )
    monkeypatch.setattr(
        migration,
        "create_engine",
        lambda url: source_engine if url.startswith("sqlite:") else target_engine,
    )

    try:
        migration.copy_sqlite_database(
            source_path,
            "postgresql+psycopg://unused/unused",
        )
    except ValueError as error:
        assert "non-empty PostgreSQL tables" in str(error)
    else:
        raise AssertionError("non-empty target should be rejected")

    with target_engine.connect() as connection:
        people = connection.execute(select(metadata.tables["persons"])).mappings().all()
    assert [(person["person_id"], person["name"]) for person in people] == [
        (2, "Existing")
    ]
    source_engine.dispose()
    target_engine.dispose()


def test_inspect_reports_sqlite_table_counts(tmp_path):
    source_path = tmp_path / "faces.db"
    source_engine = _create_schema(source_path)
    with source_engine.begin() as connection:
        connection.execute(
            metadata.tables["persons"].insert(),
            {
                "person_id": 4,
                "name": "Grace",
                "created_at": "now",
                "updated_at": "now",
            },
        )

    counts = migration.inspect_sqlite_database(source_path)

    assert counts["persons"] == 1
    assert counts["meetings"] == 0
    source_engine.dispose()