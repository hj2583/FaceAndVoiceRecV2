from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from face_voice_rec import config as app_config
from face_voice_rec.storage.relational_schema import metadata


def test_initial_migration_upgrades_and_downgrades_empty_database(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "migration.db"
    alembic_config = Config(
        str(Path(__file__).resolve().parents[2] / "alembic.ini")
    )
    alembic_config.set_main_option(
        "sqlalchemy.url",
        f"sqlite:///{database_path.as_posix()}",
    )
    monkeypatch.setattr(app_config, "DATABASE_URL", "")

    command.upgrade(alembic_config, "head")
    command.check(alembic_config)

    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    assert set(inspect(engine).get_table_names()) == set(metadata.tables) | {
        "alembic_version"
    }
    inspector = inspect(engine)
    for table_name, table in metadata.tables.items():
        assert {
            column["name"] for column in inspector.get_columns(table_name)
        } == set(table.columns.keys())
        expected_foreign_keys = {
            (
                tuple(element.parent.name for element in constraint.elements),
                tuple(element.column.table.name for element in constraint.elements),
                tuple(element.column.name for element in constraint.elements),
            )
            for constraint in table.foreign_key_constraints
        }
        actual_foreign_keys = {
            (
                tuple(constraint["constrained_columns"]),
                (constraint["referred_table"],),
                tuple(constraint["referred_columns"]),
            )
            for constraint in inspector.get_foreign_keys(table_name)
        }
        assert actual_foreign_keys == expected_foreign_keys
    engine.dispose()

    command.downgrade(alembic_config, "base")

    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    assert inspect(engine).get_table_names() == ["alembic_version"]
    engine.dispose()