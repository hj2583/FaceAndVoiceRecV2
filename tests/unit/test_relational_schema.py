from sqlalchemy import create_engine, inspect
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from face_voice_rec.storage.relational_schema import metadata


def test_relational_schema_creates_all_current_tables_in_sqlite():
    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)

    assert set(inspect(engine).get_table_names()) == set(metadata.tables)

    with engine.connect() as connection:
        connection.exec_driver_sql(
            "INSERT INTO persons(name, created_at, updated_at) "
            "VALUES ('Ari', 'now', 'now')"
        )
        connection.exec_driver_sql(
            "INSERT INTO meetings(video_path, created_at) VALUES ('meeting.mp4', 'now')"
        )
        connection.exec_driver_sql(
            "INSERT INTO attendance_events(" 
            "person_id, person_name, source, source_ref, track_id, confidence, "
            "observed_at_utc, media_offset_ms, created_at) "
            "VALUES (1, 'Ari', 'realtime', 'camera-1', 1, 0.9, 'now', NULL, 'now')"
        )
        connection.commit()

    engine.dispose()


def test_relational_schema_compiles_for_postgresql():
    statements = {
        table.name: str(CreateTable(table).compile(dialect=postgresql.dialect()))
        for table in metadata.sorted_tables
    }

    assert "SERIAL" in statements["persons"]
    assert "FOREIGN KEY(person_id) REFERENCES persons (person_id)" in statements[
        "face_embeddings"
    ]
    assert "CHECK" in statements["attendance_events"]