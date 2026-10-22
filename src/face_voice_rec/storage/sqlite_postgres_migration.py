"""Copy the existing relational SQLite database into an empty PostgreSQL schema."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, func, inspect, select, text

from face_voice_rec.storage.relational_schema import metadata


_BATCH_SIZE = 500


def inspect_sqlite_database(source_path: Path) -> dict[str, int]:
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"SQLite source database does not exist: {source_path}")

    engine = create_engine(f"sqlite:///{source_path.resolve().as_posix()}")
    try:
        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
        missing_tables = sorted(set(metadata.tables) - existing_tables)
        if missing_tables:
            raise ValueError(
                "SQLite source is missing schema tables: " + ", ".join(missing_tables)
            )
        counts = {}
        with engine.connect() as connection:
            for table in metadata.sorted_tables:
                counts[table.name] = int(
                    connection.execute(
                        select(func.count()).select_from(table)
                    ).scalar_one()
                )
        return counts
    finally:
        engine.dispose()


def copy_sqlite_database(source_path: Path, target_url: str) -> dict[str, int]:
    """Copy all schema-managed rows into a verified-empty SQLAlchemy target."""
    if not target_url.startswith(("postgresql://", "postgresql+psycopg://")):
        raise ValueError("Migration target must be PostgreSQL")
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"SQLite source database does not exist: {source_path}")

    source_engine = create_engine(f"sqlite:///{source_path.resolve().as_posix()}")
    target_engine = create_engine(target_url)
    copied = {}
    try:
        source_inspector = inspect(source_engine)
        target_inspector = inspect(target_engine)
        source_tables = set(source_inspector.get_table_names())
        target_tables = set(target_inspector.get_table_names())
        missing_source = sorted(set(metadata.tables) - source_tables)
        missing_target = sorted(set(metadata.tables) - target_tables)
        if missing_source:
            raise ValueError(
                "SQLite source is missing schema tables: " + ", ".join(missing_source)
            )
        if missing_target:
            raise ValueError(
                "PostgreSQL schema is missing tables; run Alembic first: "
                + ", ".join(missing_target)
            )

        with source_engine.connect() as source, target_engine.begin() as target:
            populated_tables = []
            for table in metadata.sorted_tables:
                count = int(
                    target.execute(select(func.count()).select_from(table)).scalar_one()
                )
                if count:
                    populated_tables.append(table.name)
            if populated_tables:
                raise ValueError(
                    "Refusing to import into non-empty PostgreSQL tables: "
                    + ", ".join(populated_tables)
                )

            for table in metadata.sorted_tables:
                result = source.execute(select(table)).mappings()
                table_count = 0
                while rows := result.fetchmany(_BATCH_SIZE):
                    target.execute(table.insert(), [dict(row) for row in rows])
                    table_count += len(rows)
                copied[table.name] = table_count

            if target.dialect.name == "postgresql":
                _reset_postgres_sequences(target)
    finally:
        source_engine.dispose()
        target_engine.dispose()
    return copied


def _reset_postgres_sequences(connection) -> None:
    for table in metadata.sorted_tables:
        primary_key = list(table.primary_key.columns)
        if len(primary_key) != 1:
            continue
        column = primary_key[0]
        max_id = connection.execute(select(func.max(column))).scalar_one()
        if max_id is None:
            continue
        connection.execute(
            text(
                "SELECT setval(pg_get_serial_sequence(:table_name, :column_name), "
                ":max_id, true)"
            ),
            {
                "table_name": table.name,
                "column_name": column.name,
                "max_id": int(max_id),
            },
        )