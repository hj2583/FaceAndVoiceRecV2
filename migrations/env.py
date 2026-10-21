from alembic import context
from sqlalchemy import create_engine, pool

from face_voice_rec import config as app_config
from face_voice_rec.storage.relational_schema import metadata


alembic_config = context.config
target_metadata = metadata


def _database_url() -> str:
    url = app_config.DATABASE_URL or alembic_config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError(
            "Set DATABASE_URL before running database migrations"
        )
    return url


def run_migrations_offline() -> None:
    url = _database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(_database_url(), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()