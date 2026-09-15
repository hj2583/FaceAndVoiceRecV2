import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from config import DB_PATH


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
