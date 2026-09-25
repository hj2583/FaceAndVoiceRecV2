import logging
import sqlite3
import threading
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import config
from config import DB_PATH


logger = logging.getLogger(__name__)

_DB_LOCK = threading.RLock()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                person_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                embedding_path TEXT NOT NULL,
                image_path TEXT,
                quality REAL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(person_id) REFERENCES persons(person_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS unknown_tracks (
                unknown_id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                image_path TEXT,
                embedding_path TEXT,
                created_at TEXT NOT NULL,
                resolved_person_id INTEGER,
                FOREIGN KEY(resolved_person_id) REFERENCES persons(person_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS unknown_samples (
                sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                unknown_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                embedding_path TEXT NOT NULL,
                quality REAL DEFAULT 0,
                angle_score REAL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(unknown_id)
                    REFERENCES unknown_tracks(unknown_id)
                    ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS recognition_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_sec REAL NOT NULL,
                person_id INTEGER,
                person_name TEXT,
                confidence REAL,
                source TEXT NOT NULL,
                FOREIGN KEY(person_id) REFERENCES persons(person_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS audio_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                person_id INTEGER,
                person_name TEXT,
                confidence REAL,
                source TEXT NOT NULL,
                transcript TEXT,
                FOREIGN KEY(person_id) REFERENCES persons(person_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS meetings (
                meeting_id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_path TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                processed_at TEXT,
                transcription_status TEXT NOT NULL DEFAULT 'pending',
                error_log TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS transcription_segments (
                segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER NOT NULL,
                speaker_label TEXT NOT NULL,
                person_id INTEGER,
                start_ms INTEGER NOT NULL,
                end_ms INTEGER NOT NULL,
                text TEXT NOT NULL,
                confidence REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(meeting_id) REFERENCES meetings(meeting_id),
                FOREIGN KEY(person_id) REFERENCES persons(person_id)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_transcription_meeting
            ON transcription_segments(meeting_id)
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS voice_embeddings (
                embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                embedding_path TEXT NOT NULL,
                quality REAL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(person_id) REFERENCES persons(person_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS unknown_voices (
                unknown_voice_id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                embedding_path TEXT,
                created_at TEXT NOT NULL,
                resolved_person_id INTEGER,
                FOREIGN KEY(resolved_person_id) REFERENCES persons(person_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS unknown_voice_samples (
                sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                unknown_voice_id INTEGER NOT NULL,
                embedding_path TEXT NOT NULL,
                quality REAL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(unknown_voice_id)
                    REFERENCES unknown_voices(unknown_voice_id)
                    ON DELETE CASCADE
            )
        """)

        conn.commit()


@contextmanager
def get_conn():
    with _DB_LOCK:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def create_person(name):
    name = name.strip()
    if not name:
        raise ValueError("Name cannot be empty")

    now = utc_now()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO persons(name, created_at, updated_at) VALUES (?, ?, ?)",
            (name, now, now),
        )
        row = conn.execute(
            "SELECT person_id FROM persons WHERE name=?",
            (name,),
        ).fetchone()
        return int(row[0])


def rename_person(person_id, name):
    name = name.strip()
    if not name:
        raise ValueError("Name cannot be empty")
    with get_conn() as conn:
        conn.execute(
            "UPDATE persons SET name=?, updated_at=? WHERE person_id=?",
            (name, utc_now(), person_id),
        )
        meeting_ids = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT meeting_id FROM transcription_segments WHERE person_id=?",
                (person_id,),
            ).fetchall()
        ]
        conn.execute(
            "UPDATE transcription_segments SET speaker_label=? WHERE person_id=?",
            (name, person_id),
        )
        conn.execute(
            "UPDATE audio_logs SET person_name=? WHERE person_id=?",
            (name, person_id),
        )
    _regenerate_meeting_transcripts(meeting_ids)


def _regenerate_meeting_transcripts(meeting_ids):
    """Rewrite each meeting's on-disk .txt transcripts from its stored segments."""
    if not meeting_ids:
        return
    # Lazy import: transcription_core -> voice_core -> database.
    import transcription_core

    for meeting_id in meeting_ids:
        try:
            with get_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT speaker_label, start_ms, end_ms, text, confidence
                    FROM transcription_segments
                    WHERE meeting_id=?
                    ORDER BY start_ms
                    """,
                    (meeting_id,),
                ).fetchall()
            meeting_dir = Path(config.TRANSCRIPTS_DIR) / str(meeting_id)
            # Old labels' files would otherwise linger next to the regenerated ones.
            for stale_path in meeting_dir.glob("*.txt"):
                stale_path.unlink(missing_ok=True)
            segments = [transcription_core.TranscriptionSegment(*row) for row in rows]
            # DB rows are already up to date, so only the files are rewritten.
            transcription_core.save_transcripts(segments, meeting_dir)
        except Exception:
            logger.exception("Failed regenerating transcripts for meeting %s", meeting_id)


def add_embedding(person_id, embedding_path, image_path=None, quality=0.0):
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO face_embeddings(
                person_id, embedding_path, image_path, quality, created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                person_id,
                str(embedding_path),
                str(image_path) if image_path else None,
                float(quality),
                utc_now(),
            ),
        )
        return int(cur.lastrowid)

def create_unknown(label, image_path, embedding_path):
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO unknown_tracks(
                label, image_path, embedding_path, created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (label, str(image_path), str(embedding_path), utc_now()),
        )
        return int(cur.lastrowid)

def add_unknown_sample(
    unknown_id,
    image_path,
    embedding_path,
    quality=0.0,
    angle_score=0.0,
):
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO unknown_samples(
                unknown_id,
                image_path,
                embedding_path,
                quality,
                angle_score,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                int(unknown_id),
                str(image_path),
                str(embedding_path),
                float(quality),
                float(angle_score),
                utc_now(),
            ),
        )

        return int(cur.lastrowid)

def update_unknown_image(unknown_id, image_path):
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE unknown_tracks
            SET image_path=?
            WHERE unknown_id=?
            """,
            (
                str(image_path),
                int(unknown_id),
            ),
        )

def list_unknown_samples(unknown_id):
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT
                sample_id,
                image_path,
                embedding_path,
                quality,
                angle_score,
                created_at
            FROM unknown_samples
            WHERE unknown_id=?
            ORDER BY quality DESC
            """,
            (int(unknown_id),),
        ).fetchall()

def resolve_unknown(unknown_id, person_id):
    with get_conn() as conn:
        conn.execute(
            "UPDATE unknown_tracks SET resolved_person_id=? WHERE unknown_id=?",
            (person_id, unknown_id),
        )


_FACE_REFERENCE_SPECS = (
    ("unknown_tracks", ("image_path", "embedding_path")),
    ("unknown_samples", ("image_path", "embedding_path")),
)

_VOICE_REFERENCE_SPECS = (
    ("voice_embeddings", ("embedding_path",)),
    ("unknown_voices", ("embedding_path",)),
    ("unknown_voice_samples", ("embedding_path",)),
)


def _unlink_unreferenced(paths, conn, reference_specs=_FACE_REFERENCE_SPECS):
    candidates = {str(Path(path)) for path in paths if path}
    if not candidates:
        return

    referenced = set()
    for table, columns in reference_specs:
        for column in columns:
            rows = conn.execute(
                f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL"
            ).fetchall()
            referenced.update(str(Path(row[0])) for row in rows if row[0])

    for path in candidates - referenced:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


def _delete_unknown_locked(conn, unknown_id):
    parent = conn.execute(
        "SELECT image_path, embedding_path, resolved_person_id "
        "FROM unknown_tracks WHERE unknown_id=?",
        (int(unknown_id),),
    ).fetchone()
    if parent is None or parent[2] is not None:
        return False

    sample_rows = conn.execute(
        "SELECT image_path, embedding_path FROM unknown_samples WHERE unknown_id=?",
        (int(unknown_id),),
    ).fetchall()
    paths = [parent[0], parent[1]]
    paths.extend(path for row in sample_rows for path in row)

    conn.execute("DELETE FROM unknown_samples WHERE unknown_id=?", (int(unknown_id),))
    conn.execute("DELETE FROM unknown_tracks WHERE unknown_id=?", (int(unknown_id),))
    _unlink_unreferenced(paths, conn)
    return True


def delete_unknown(unknown_id):
    with get_conn() as conn:
        return _delete_unknown_locked(conn, unknown_id)


def delete_low_quality_unknowns(threshold=0.4):
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT ut.unknown_id
            FROM unknown_tracks ut
            LEFT JOIN unknown_samples us ON us.unknown_id = ut.unknown_id
            WHERE ut.resolved_person_id IS NULL
            GROUP BY ut.unknown_id
            HAVING COALESCE(MAX(us.quality), 0.0) < ?
            ORDER BY ut.unknown_id
            """,
            (float(threshold),),
        ).fetchall()
        deleted = [
            int(row[0])
            for row in rows
            if _delete_unknown_locked(conn, row[0])
        ]
        return deleted


def add_voice_embedding(person_id, embedding_path, quality=0.0):
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO voice_embeddings(person_id, embedding_path, quality, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (int(person_id), str(embedding_path), float(quality), utc_now()),
        )
        return int(cur.lastrowid)


def list_voice_embeddings():
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT ve.embedding_id, ve.person_id, p.name, ve.embedding_path, ve.quality
            FROM voice_embeddings ve
            JOIN persons p ON p.person_id = ve.person_id
            ORDER BY ve.embedding_id
            """
        ).fetchall()


def create_unknown_voice(label, embedding_path):
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO unknown_voices(label, embedding_path, created_at)
            VALUES (?, ?, ?)
            """,
            (label, str(embedding_path) if embedding_path else None, utc_now()),
        )
        return int(cur.lastrowid)


def update_unknown_voice(unknown_voice_id, label, embedding_path):
    with get_conn() as conn:
        conn.execute(
            "UPDATE unknown_voices SET label=?, embedding_path=? WHERE unknown_voice_id=?",
            (label, str(embedding_path), int(unknown_voice_id)),
        )


def add_unknown_voice_sample(unknown_voice_id, embedding_path, quality=0.0):
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO unknown_voice_samples(unknown_voice_id, embedding_path, quality, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (int(unknown_voice_id), str(embedding_path), float(quality), utc_now()),
        )
        return int(cur.lastrowid)


def list_unknown_voice_samples(unknown_voice_id):
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT sample_id, embedding_path, quality, created_at
            FROM unknown_voice_samples
            WHERE unknown_voice_id=?
            ORDER BY quality DESC
            """,
            (int(unknown_voice_id),),
        ).fetchall()


def list_unknown_voice_samples_with_embeddings():
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT uvs.sample_id, uvs.unknown_voice_id, uvs.embedding_path, uvs.quality
            FROM unknown_voice_samples uvs
            JOIN unknown_voices uv ON uv.unknown_voice_id = uvs.unknown_voice_id
            WHERE uv.resolved_person_id IS NULL
            ORDER BY uvs.quality DESC
            """
        ).fetchall()


def list_unknown_voices(include_resolved=False):
    query = """
        SELECT unknown_voice_id, label, embedding_path, created_at, resolved_person_id
        FROM unknown_voices
    """
    if not include_resolved:
        query += " WHERE resolved_person_id IS NULL"
    query += " ORDER BY unknown_voice_id DESC"
    with get_conn() as conn:
        return conn.execute(query).fetchall()


def resolve_unknown_voice(unknown_voice_id, person_id):
    meeting_ids = []
    with get_conn() as conn:
        label_row = conn.execute(
            "SELECT label FROM unknown_voices WHERE unknown_voice_id=?",
            (int(unknown_voice_id),),
        ).fetchone()
        conn.execute(
            "UPDATE unknown_voices SET resolved_person_id=? WHERE unknown_voice_id=?",
            (person_id, unknown_voice_id),
        )
        if label_row is None:
            return
        label = label_row[0]
        person_name = conn.execute(
            "SELECT name FROM persons WHERE person_id=?",
            (person_id,),
        ).fetchone()[0]
        meeting_ids = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT meeting_id FROM transcription_segments WHERE speaker_label=?",
                (label,),
            ).fetchall()
        ]
        conn.execute(
            "UPDATE transcription_segments SET speaker_label=?, person_id=? WHERE speaker_label=?",
            (person_name, person_id, label),
        )
        conn.execute(
            "UPDATE audio_logs SET person_name=?, person_id=? WHERE person_name=?",
            (person_name, person_id, label),
        )
    _regenerate_meeting_transcripts(meeting_ids)


def _delete_unknown_voice_locked(conn, unknown_voice_id):
    parent = conn.execute(
        "SELECT embedding_path, resolved_person_id FROM unknown_voices WHERE unknown_voice_id=?",
        (int(unknown_voice_id),),
    ).fetchone()
    if parent is None or parent[1] is not None:
        return False

    sample_rows = conn.execute(
        "SELECT embedding_path FROM unknown_voice_samples WHERE unknown_voice_id=?",
        (int(unknown_voice_id),),
    ).fetchall()
    paths = [parent[0]] + [row[0] for row in sample_rows]

    conn.execute(
        "DELETE FROM unknown_voice_samples WHERE unknown_voice_id=?",
        (int(unknown_voice_id),),
    )
    conn.execute(
        "DELETE FROM unknown_voices WHERE unknown_voice_id=?",
        (int(unknown_voice_id),),
    )
    _unlink_unreferenced(paths, conn, reference_specs=_VOICE_REFERENCE_SPECS)
    return True


def delete_unknown_voice(unknown_voice_id):
    with get_conn() as conn:
        return _delete_unknown_voice_locked(conn, unknown_voice_id)


def list_persons():
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT person_id, name, created_at, updated_at
            FROM persons
            ORDER BY name
            """
        ).fetchall()


def list_embeddings():
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT
                fe.embedding_id,
                fe.person_id,
                p.name,
                fe.embedding_path,
                fe.image_path,
                fe.quality
            FROM face_embeddings fe
            JOIN persons p ON p.person_id = fe.person_id
            ORDER BY fe.embedding_id
            """
        ).fetchall()


def list_unknowns(include_resolved=False):
    with get_conn() as conn:
        if include_resolved:
            return conn.execute(
                """
                SELECT
                    unknown_id, label, image_path, embedding_path,
                    created_at, resolved_person_id
                FROM unknown_tracks
                ORDER BY unknown_id DESC
                """
            ).fetchall()

        return conn.execute(
            """
            SELECT
                unknown_id, label, image_path, embedding_path,
                created_at, resolved_person_id
            FROM unknown_tracks
            WHERE resolved_person_id IS NULL
            ORDER BY unknown_id DESC
            """
        ).fetchall()

def list_unknown_samples_with_embeddings():
    """
    Return all unresolved unknown samples together with
    their parent unknown_id and embedding path.
    """

    with get_conn() as conn:
        return conn.execute(
            """
            SELECT
                us.sample_id,
                us.unknown_id,
                us.embedding_path,
                us.quality
            FROM unknown_samples us
            JOIN unknown_tracks ut
                ON ut.unknown_id = us.unknown_id
            WHERE ut.resolved_person_id IS NULL
            ORDER BY us.quality DESC
            """
        ).fetchall()

def log_recognition(timestamp_sec, person_id, person_name, confidence, source):
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO recognition_logs(
                timestamp_sec, person_id, person_name, confidence, source
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                float(timestamp_sec),
                person_id,
                person_name,
                float(confidence),
                source,
            ),
        )


def log_audio(start_time, end_time, person_id, person_name, confidence, source, transcript=None):
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO audio_logs(
                start_time, end_time, person_id, person_name,
                confidence, source, transcript
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                float(start_time),
                float(end_time),
                person_id,
                person_name,
                float(confidence),
                source,
                transcript,
            ),
        )


def fetch_audio_logs(source=None):
    with get_conn() as conn:
        if source:
            return conn.execute(
                """
                SELECT log_id, start_time, end_time, person_name,
                       confidence, source, transcript
                FROM audio_logs
                WHERE source=?
                ORDER BY start_time DESC
                """,
                (source,),
            ).fetchall()

        return conn.execute(
            """
            SELECT log_id, start_time, end_time, person_name,
                   confidence, source, transcript
            FROM audio_logs
            ORDER BY start_time DESC
            """
        ).fetchall()
