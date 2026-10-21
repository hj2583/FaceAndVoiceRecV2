from face_voice_rec.storage.database import (
    _PostgresConnection,
    _insert_id,
    _postgres_sql,
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