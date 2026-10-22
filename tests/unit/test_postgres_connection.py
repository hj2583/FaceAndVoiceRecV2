from face_voice_rec.storage.database import (
    _PostgresConnection,
    _hybrid_row,
    _hybrid_row_factory,
    _insert_id,
    _postgres_sql,
    init_db,
)


class _FakeCursor:
    def __init__(self, row=None):
        self.row = row

    def fetchone(self):
        return self.row


class _FakePsycopgConnection:
    def __init__(self, row=None):
        self.row = row
        self.calls = []

    def execute(self, statement, parameters=None):
        self.calls.append((statement, parameters))
        return _FakeCursor(self.row)

    def executemany(self, statement, parameters):
        self.calls.append((statement, list(parameters)))
        return _FakeCursor()


def test_postgres_sql_translates_qmark_placeholders_outside_quoted_strings():
    statement = "SELECT '?' AS marker FROM persons WHERE name=? AND active=?"

    assert _postgres_sql(statement) == (
        "SELECT '?' AS marker FROM persons WHERE name=%s AND active=%s"
    )


def test_postgres_sql_translates_sqlite_ignore_insert():
    statement = "INSERT OR IGNORE INTO persons(name) VALUES (?)"

    assert _postgres_sql(statement) == (
        "INSERT INTO persons(name) VALUES (%s) ON CONFLICT DO NOTHING"
    )


def test_postgres_connection_executes_translated_sql():
    connection = _FakePsycopgConnection()
    adapted = _PostgresConnection(connection)

    adapted.execute("SELECT person_id FROM persons WHERE name=?", ("Ari",))

    assert connection.calls == [
        ("SELECT person_id FROM persons WHERE name=%s", ("Ari",))
    ]


def test_postgres_insert_id_uses_returning_primary_key():
    connection = _FakePsycopgConnection(row=(73,))
    adapted = _PostgresConnection(connection)

    inserted_id = _insert_id(
        adapted,
        "INSERT INTO persons(name) VALUES (?)",
        ("Ari",),
        "person_id",
    )

    assert inserted_id == 73
    assert connection.calls == [
        ("INSERT INTO persons(name) VALUES (%s) RETURNING person_id", ("Ari",))
    ]


def test_init_db_uses_alembic_when_postgres_is_configured(monkeypatch):
    from alembic import command
    from face_voice_rec import config

    calls = []
    monkeypatch.setattr(config, "DATABASE_URL", "postgresql+psycopg://db/app")
    monkeypatch.setattr(command, "upgrade", lambda cfg, revision: calls.append(revision))

    init_db()

    assert calls == ["head"]


def test_postgres_row_supports_numeric_and_named_column_access():
    class Column:
        def __init__(self, name):
            self.name = name

    class Cursor:
        description = (Column("segment_id"), Column("text"))

    row = _hybrid_row(Cursor(), (12, "hello"))

    assert row[0] == row["segment_id"] == 12
    assert row[1] == row["text"] == "hello"


def test_postgres_hybrid_row_factory_follows_psycopg_factory_protocol():
    class Column:
        def __init__(self, name):
            self.name = name

    class Cursor:
        description = (Column("segment_id"), Column("text"))

    make_row = _hybrid_row_factory(Cursor())
    row = make_row((12, "hello"))

    assert row[0] == row["segment_id"] == 12
    assert row[1] == row["text"] == "hello"

def test_get_conn_passes_hybrid_row_factory_to_psycopg(monkeypatch):
    import sqlite3

    import psycopg

    from face_voice_rec import config
    from face_voice_rec.storage import database

    class Connection:
        def __init__(self):
            self.events = []

        def commit(self):
            self.events.append("commit")

        def rollback(self):
            self.events.append("rollback")

        def close(self):
            self.events.append("close")

    connection = Connection()
    captured = {}

    def fake_connect(dsn, **kwargs):
        captured["dsn"] = dsn
        captured["kwargs"] = kwargs
        return connection

    monkeypatch.setattr(config, "DATABASE_URL", "postgresql+psycopg://db/app")
    monkeypatch.setattr(psycopg, "connect", fake_connect)

    with database.get_conn(row_factory=sqlite3.Row) as conn:
        assert isinstance(conn, _PostgresConnection)

    assert captured["kwargs"]["row_factory"] is _hybrid_row_factory
    assert connection.events == ["commit", "close"]
