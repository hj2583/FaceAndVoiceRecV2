from sqlalchemy.engine import make_url

from face_voice_rec.docker_entrypoint import rewrite_database_url


def test_rewrite_database_url_changes_only_host():
    original = (
        "postgresql+psycopg://sample_user:p%40ss%3Aword@localhost:5432/"
        "face_voice_rec?sslmode=prefer"
    )

    rewritten = rewrite_database_url(original, "postgres")
    parsed = make_url(rewritten)

    assert parsed.host == "postgres"
    assert parsed.port == 5432
    assert parsed.database == "face_voice_rec"
    assert parsed.username == "sample_user"
    assert parsed.password == "p@ss:word"
    assert parsed.query == {"sslmode": "prefer"}
