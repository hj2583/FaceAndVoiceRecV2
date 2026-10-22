import logging
import json
import math
import re
import sqlite3
import threading
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
from numbers import Integral, Real
from typing import Mapping, Sequence

import psycopg

from face_voice_rec import config
from face_voice_rec.config import DB_PATH


logger = logging.getLogger(__name__)

_DB_LOCK = threading.RLock()
DATABASE_ERRORS = (sqlite3.Error, psycopg.Error)
INTEGRITY_ERRORS = (sqlite3.IntegrityError, psycopg.IntegrityError)


def _postgres_sql(statement):
    """Translate this repository's qmark parameters and SQLite ignore inserts."""
    ignore_conflicts = re.match(
        r"^(\s*)INSERT\s+OR\s+IGNORE\s+INTO\b",
        statement,
        flags=re.IGNORECASE,
    )
    if ignore_conflicts:
        statement = re.sub(
            r"^(\s*)INSERT\s+OR\s+IGNORE\s+INTO\b",
            r"\1INSERT INTO",
            statement,
            count=1,
            flags=re.IGNORECASE,
        )

    translated = []
    in_single_quote = False
    in_double_quote = False
    index = 0
    while index < len(statement):
        character = statement[index]
        if character == "'" and not in_double_quote:
            translated.append(character)
            if in_single_quote and index + 1 < len(statement) and statement[index + 1] == "'":
                translated.append("'")
                index += 2
                continue
            in_single_quote = not in_single_quote
        elif character == '"' and not in_single_quote:
            translated.append(character)
            if in_double_quote and index + 1 < len(statement) and statement[index + 1] == '"':
                translated.append('"')
                index += 2
                continue
            in_double_quote = not in_double_quote
        elif character == "?" and not in_single_quote and not in_double_quote:
            translated.append("%s")
        else:
            translated.append(character)
        index += 1

    statement = "".join(translated)
    if ignore_conflicts and "ON CONFLICT" not in statement.upper():
        statement = statement.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"
    return statement


class _PostgresConnection:
    def __init__(self, connection):
        self._connection = connection

    def execute(self, statement, parameters=None):
        return self._connection.execute(_postgres_sql(statement), parameters)

    def executemany(self, statement, parameters):
        return self._connection.executemany(_postgres_sql(statement), parameters)

    def commit(self):
        self._connection.commit()

    def rollback(self):
        self._connection.rollback()

    def close(self):
        self._connection.close()

    @property
    def row_factory(self):
        return self._connection.row_factory

    @row_factory.setter
    def row_factory(self, factory):
        self._connection.row_factory = (
            _hybrid_row_factory if factory is sqlite3.Row else factory
        )


class _HybridRow:
    """Support both sqlite3.Row-style numeric and named column access."""

    def __init__(self, columns, values):
        self._columns = tuple(columns)
        self._values = tuple(values)
        self._positions = {name: index for index, name in enumerate(self._columns)}

    def __getitem__(self, key):
        if isinstance(key, str):
            key = self._positions[key]
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


def _hybrid_row(cursor, values):
    return _HybridRow((column.name for column in cursor.description), values)


def _hybrid_row_factory(cursor):
    columns = tuple(column.name for column in cursor.description)
    return lambda values: _HybridRow(columns, values)


def _insert_id(connection, statement, parameters, column_name):
    if isinstance(connection, _PostgresConnection):
        statement = statement.rstrip().rstrip(";") + f" RETURNING {column_name}"
        row = connection.execute(statement, parameters).fetchone()
        return int(row[0])
    cursor = connection.execute(statement, parameters)
    return int(cursor.lastrowid)


def _postgres_dsn():
    return config.DATABASE_URL.replace(
        "postgresql+psycopg://",
        "postgresql://",
        1,
    )


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def init_db():
    if config.DATABASE_URL:
        if not config.DATABASE_URL.startswith(
            ("postgresql://", "postgresql+psycopg://")
        ):
            raise ValueError("DATABASE_URL must be a PostgreSQL URL")
        from alembic import command
        from alembic.config import Config

        project_root = Path(__file__).resolve().parents[3]
        alembic_config = Config(str(project_root / "alembic.ini"))
        command.upgrade(alembic_config, "head")
        return

    with _DB_LOCK:
        conn = sqlite3.connect(DB_PATH, timeout=config.DB_BUSY_TIMEOUT_SECONDS)
        try:
            conn.execute(
                f"PRAGMA busy_timeout={config.DB_BUSY_TIMEOUT_SECONDS * 1000}"
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("BEGIN IMMEDIATE")

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
                source_ref TEXT,
                FOREIGN KEY(person_id) REFERENCES persons(person_id)
            )
        """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS attendance_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                person_name TEXT NOT NULL,
                source TEXT NOT NULL CHECK(source IN ('realtime', 'video')),
                source_ref TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                confidence REAL NOT NULL,
                observed_at_utc TEXT,
                media_offset_ms INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY(person_id) REFERENCES persons(person_id),
                UNIQUE(source, source_ref, track_id, person_id),
                CHECK (
                    (source = 'realtime' AND observed_at_utc IS NOT NULL AND media_offset_ms IS NULL)
                    OR
                    (source = 'video' AND observed_at_utc IS NULL AND media_offset_ms IS NOT NULL)
                )
            )
        """)

            audio_log_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(audio_logs)")
        }
            if "track_id" not in audio_log_columns:
                conn.execute("ALTER TABLE audio_logs ADD COLUMN track_id INTEGER")
            if "source_ref" not in audio_log_columns:
                conn.execute("ALTER TABLE audio_logs ADD COLUMN source_ref TEXT")

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
            CREATE TABLE IF NOT EXISTS meeting_introductions (
                introduction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                job_title TEXT,
                start_ms INTEGER NOT NULL,
                end_ms INTEGER NOT NULL,
                speaker_label TEXT NOT NULL,
                source_segment_ids TEXT NOT NULL,
                voice_unknown_id INTEGER,
                face_track_id INTEGER,
                person_id INTEGER,
                parser_confidence REAL NOT NULL,
                match_confidence REAL,
                status TEXT NOT NULL,
                reason TEXT,
                face_unknown_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(meeting_id) REFERENCES meetings(meeting_id) ON DELETE CASCADE,
                FOREIGN KEY(voice_unknown_id)
                    REFERENCES unknown_voices(unknown_voice_id) ON DELETE SET NULL,
                FOREIGN KEY(person_id) REFERENCES persons(person_id),
                FOREIGN KEY(face_unknown_id)
                    REFERENCES unknown_tracks(unknown_id) ON DELETE SET NULL
            )
        """)
            introduction_columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(meeting_introductions)"
            ).fetchall()
        }
            if "face_unknown_id" not in introduction_columns:
                conn.execute(
                "ALTER TABLE meeting_introductions "
                "ADD COLUMN face_unknown_id INTEGER "
                "REFERENCES unknown_tracks(unknown_id) ON DELETE SET NULL"
            )
            conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_meeting_introduction_source
            ON meeting_introductions(
                meeting_id, name, start_ms, end_ms
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
                original_text TEXT,
                confidence REAL,
                sentence_type TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(meeting_id) REFERENCES meetings(meeting_id),
                FOREIGN KEY(person_id) REFERENCES persons(person_id)
            )
        """)

            segment_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(transcription_segments)")
        }
            if "sentence_type" not in segment_columns:
                conn.execute("ALTER TABLE transcription_segments ADD COLUMN sentence_type TEXT")
            if "original_text" not in segment_columns:
                conn.execute(
                "ALTER TABLE transcription_segments ADD COLUMN original_text TEXT"
            )
            conn.execute(
            "UPDATE transcription_segments SET original_text=text "
            "WHERE original_text IS NULL"
        )
            if "word_timestamps" not in segment_columns:
                conn.execute(
                "ALTER TABLE transcription_segments ADD COLUMN word_timestamps TEXT"
            )

            conn.execute("""
            CREATE TABLE IF NOT EXISTS refined_transcription_segments (
                refined_segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER NOT NULL,
                speaker_label TEXT NOT NULL,
                person_id INTEGER,
                start_ms INTEGER NOT NULL,
                end_ms INTEGER NOT NULL,
                text TEXT NOT NULL,
                source_segment_ids TEXT NOT NULL,
                sentence_type TEXT,
                refinement_method TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(meeting_id) REFERENCES meetings(meeting_id) ON DELETE CASCADE,
                FOREIGN KEY(person_id) REFERENCES persons(person_id)
            )
        """)
            conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_refined_transcription_meeting
            ON refined_transcription_segments(meeting_id)
        """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS transcript_refinement_runs (
                meeting_id INTEGER PRIMARY KEY,
                status TEXT NOT NULL,
                method TEXT NOT NULL,
                speaker_change_recommendations TEXT NOT NULL DEFAULT '[]',
                error TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(meeting_id) REFERENCES meetings(meeting_id) ON DELETE CASCADE
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
                audio_path TEXT,
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
                audio_path TEXT,
                source_ref TEXT,
                start_ms INTEGER,
                end_ms INTEGER,
                quality REAL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(unknown_voice_id)
                    REFERENCES unknown_voices(unknown_voice_id)
                    ON DELETE CASCADE
            )
        """)

            voice_embedding_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(voice_embeddings)")
        }
            if "audio_path" not in voice_embedding_columns:
                conn.execute("ALTER TABLE voice_embeddings ADD COLUMN audio_path TEXT")

            unknown_voice_sample_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(unknown_voice_samples)")
        }
            for column, definition in (
            ("audio_path", "TEXT"),
            ("source_ref", "TEXT"),
            ("start_ms", "INTEGER"),
            ("end_ms", "INTEGER"),
        ):
                if column not in unknown_voice_sample_columns:
                    conn.execute(
                    f"ALTER TABLE unknown_voice_samples ADD COLUMN {column} {definition}"
                )
            conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_unknown_voice_sample_source_window
            ON unknown_voice_samples(unknown_voice_id, source_ref, start_ms, end_ms)
            WHERE source_ref IS NOT NULL AND start_ms IS NOT NULL AND end_ms IS NOT NULL
        """)

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


@contextmanager
def get_conn(*, row_factory=None):
    if config.DATABASE_URL:
        postgres_row_factory = _hybrid_row_factory if row_factory is sqlite3.Row else None
        connection = psycopg.connect(
            _postgres_dsn(),
            row_factory=postgres_row_factory,
        )
        adapted_connection = _PostgresConnection(connection)
        try:
            yield adapted_connection
            adapted_connection.commit()
        except Exception:
            adapted_connection.rollback()
            raise
        finally:
            adapted_connection.close()
        return

    with _DB_LOCK:
        conn = sqlite3.connect(
            DB_PATH,
            timeout=config.DB_BUSY_TIMEOUT_SECONDS,
        )
        if row_factory is not None:
            conn.row_factory = row_factory
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
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
            "UPDATE refined_transcription_segments SET speaker_label=? WHERE person_id=?",
            (name, person_id),
        )
        conn.execute(
            "UPDATE audio_logs SET person_name=? WHERE person_id=?",
            (name, person_id),
        )
    _regenerate_meeting_transcripts(meeting_ids)


def update_transcription_segments(meeting_id, updates):
    """Apply manual corrections: dicts of segment_id, text, sentence_type, confidence."""
    with get_conn() as conn:
        conn.executemany(
            """
            UPDATE transcription_segments
            SET text=?, sentence_type=?, confidence=?, word_timestamps=NULL
            WHERE segment_id=? AND meeting_id=?
            """,
            [
                (
                    update["text"],
                    update["sentence_type"],
                    update["confidence"],
                    update["segment_id"],
                    meeting_id,
                )
                for update in updates
            ],
        )
    _regenerate_meeting_transcripts([meeting_id])
    from face_voice_rec.domain.transcription import transcription_core

    transcription_core.refresh_readable_transcript(meeting_id, use_llm=False)


def _regenerate_meeting_transcripts(meeting_ids):
    """Rewrite each meeting's on-disk .txt transcripts from its stored segments."""
    if not meeting_ids:
        return
    # Lazy import: transcription_core -> voice_core -> database.
    from face_voice_rec.domain.transcription import transcription_core

    for meeting_id in meeting_ids:
        try:
            with get_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT speaker_label, start_ms, end_ms, text, confidence, sentence_type
                    FROM transcription_segments
                    WHERE meeting_id=?
                    ORDER BY start_ms
                    """,
                    (meeting_id,),
                ).fetchall()
            meeting_dir = Path(config.TRANSCRIPTS_DIR) / str(meeting_id)
            # Old labels' files would otherwise linger next to the regenerated ones.
            for stale_path in meeting_dir.glob("*.txt"):
                try:
                    stale_path.unlink(missing_ok=True)
                except OSError:
                    # Windows refuses to delete files another process has open.
                    logger.warning("Could not remove stale transcript %s", stale_path)
            segments = [transcription_core.TranscriptionSegment(*row) for row in rows]
            # DB rows are already up to date, so only the files are rewritten.
            transcription_core.save_transcripts(segments, meeting_dir)
        except Exception:
            logger.exception("Failed regenerating transcripts for meeting %s", meeting_id)


def add_embedding(person_id, embedding_path, image_path=None, quality=0.0):
    with get_conn() as conn:
        return _insert_id(
            conn,
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
            "embedding_id",
        )

def create_unknown(label, image_path, embedding_path):
    with get_conn() as conn:
        return _insert_id(
            conn,
            """
            INSERT INTO unknown_tracks(
                label, image_path, embedding_path, created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (label, str(image_path), str(embedding_path), utc_now()),
            "unknown_id",
        )

def add_unknown_sample(
    unknown_id,
    image_path,
    embedding_path,
    quality=0.0,
    angle_score=0.0,
):
    with get_conn() as conn:
        return _insert_id(
            conn,
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
            "sample_id",
        )

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

def get_unknown_face_track_by_label(label):
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT unknown_id, resolved_person_id
            FROM unknown_tracks
            WHERE label=?
            ORDER BY unknown_id DESC
            LIMIT 2
            """,
            (str(label),),
        ).fetchall()
    if len(rows) != 1:
        return None
    unknown_id, resolved_person_id = rows[0]
    return {
        "unknown_id": int(unknown_id),
        "resolved_person_id": (
            int(resolved_person_id)
            if resolved_person_id is not None
            else None
        ),
    }


_FACE_REFERENCE_SPECS = (
    ("unknown_tracks", ("image_path", "embedding_path")),
    ("unknown_samples", ("image_path", "embedding_path")),
)

_VOICE_REFERENCE_SPECS = (
    ("voice_embeddings", ("embedding_path", "audio_path")),
    ("unknown_voices", ("embedding_path",)),
    ("unknown_voice_samples", ("embedding_path", "audio_path")),
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


def add_voice_embedding(person_id, embedding_path, quality=0.0, audio_path=None):
    with get_conn() as conn:
        return _insert_id(
            conn,
            """
            INSERT INTO voice_embeddings(
                person_id, embedding_path, audio_path, quality, created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                int(person_id),
                str(embedding_path),
                str(audio_path) if audio_path else None,
                float(quality),
                utc_now(),
            ),
            "embedding_id",
        )


def reassign_voice_embedding(embedding_id, person_id):
    """Reassign an enrollment and any source-linked unknown-speaker cluster."""
    embedding_id = int(embedding_id)
    person_id = int(person_id)
    affected_meetings = set()
    with get_conn() as conn:
        enrollment = conn.execute(
            """
            SELECT ve.embedding_id, ve.person_id, ve.embedding_path
            FROM voice_embeddings ve
            WHERE ve.embedding_id=?
            """,
            (embedding_id,),
        ).fetchone()
        if enrollment is None:
            return False
        person = conn.execute(
            "SELECT name FROM persons WHERE person_id=?",
            (person_id,),
        ).fetchone()
        if person is None:
            raise ValueError(f"Person {person_id} does not exist")
        new_person_name = person[0]
        linked_unknown_ids = {
            int(row[0])
            for row in conn.execute(
                """
                SELECT DISTINCT unknown_voice_id
                FROM unknown_voice_samples
                WHERE embedding_path=?
                """,
                (enrollment[2],),
            ).fetchall()
        }
        linked_unknown_ids.update(
            int(row[0])
            for row in conn.execute(
                """
                SELECT unknown_voice_id
                FROM unknown_voices
                WHERE embedding_path=?
                """,
                (enrollment[2],),
            ).fetchall()
        )
        if len(linked_unknown_ids) > 1:
            raise ValueError(
                "The voice enrollment is linked to multiple unknown-speaker records"
            )

        conn.execute(
            "UPDATE voice_embeddings SET person_id=? WHERE embedding_id=?",
            (person_id, embedding_id),
        )
        for unknown_voice_id in linked_unknown_ids:
            unknown = conn.execute(
                """
                SELECT resolved_person_id, embedding_path
                FROM unknown_voices
                WHERE unknown_voice_id=?
                """,
                (unknown_voice_id,),
            ).fetchone()
            if unknown is None or unknown[0] is None:
                continue
            old_person_id = int(unknown[0])
            sample_rows = conn.execute(
                """
                SELECT embedding_path, source_ref, start_ms, end_ms
                FROM unknown_voice_samples
                WHERE unknown_voice_id=?
                """,
                (unknown_voice_id,),
            ).fetchall()
            paths = {
                str(row[0])
                for row in sample_rows
                if row[0]
            }
            if unknown[1]:
                paths.add(str(unknown[1]))
            if paths:
                placeholders = ",".join("?" for _ in paths)
                conn.execute(
                    f"""
                    UPDATE voice_embeddings SET person_id=?
                    WHERE embedding_path IN ({placeholders})
                    """,
                    (person_id, *sorted(paths)),
                )
            conn.execute(
                "UPDATE unknown_voices SET resolved_person_id=? WHERE unknown_voice_id=?",
                (person_id, unknown_voice_id),
            )
            for _, source_ref, start_ms, end_ms in sample_rows:
                meeting_id = _meeting_id_for_transcript_audio(source_ref, conn)
                if (
                    meeting_id is None
                    or start_ms is None
                    or end_ms is None
                    or int(end_ms) <= int(start_ms)
                ):
                    continue
                _reassign_transcript_interval(
                    conn,
                    meeting_id,
                    int(start_ms),
                    int(end_ms),
                    old_person_id,
                    person_id,
                    new_person_name,
                )
                affected_meetings.add(meeting_id)

    if affected_meetings:
        _regenerate_meeting_transcripts(sorted(affected_meetings))
        from face_voice_rec.domain.transcription import transcription_core

        for meeting_id in sorted(affected_meetings):
            transcription_core.refresh_readable_transcript(
                meeting_id,
                use_llm=False,
            )
    return True


def _meeting_id_for_transcript_audio(source_ref, conn):
    """Return the meeting id only for this application's stored audio path."""
    if not source_ref:
        return None
    source_path = Path(source_ref).resolve()
    meeting_dir = source_path.parent
    if source_path.name.lower() != "audio.wav" or not meeting_dir.name.isdigit():
        return None
    meeting_id = int(meeting_dir.name)
    expected_path = (
        Path(config.TRANSCRIPTS_DIR).resolve()
        / str(meeting_id)
        / "audio.wav"
    )
    if source_path != expected_path:
        return None
    exists = conn.execute(
        "SELECT 1 FROM meetings WHERE meeting_id=?",
        (meeting_id,),
    ).fetchone()
    return meeting_id if exists else None


def _reassign_transcript_interval(
        conn,
        meeting_id,
        start_ms,
        end_ms,
        old_person_id,
        new_person_id,
        new_person_name,
):
    """Relabel only old-person transcript rows substantially covered by a sample."""
    sample_duration = end_ms - start_ms
    for table, id_column in (
        ("transcription_segments", "segment_id"),
        ("refined_transcription_segments", "refined_segment_id"),
    ):
        rows = conn.execute(
            f"""
            SELECT {id_column}, start_ms, end_ms
            FROM {table}
            WHERE meeting_id=? AND person_id=?
            """,
            (meeting_id, old_person_id),
        ).fetchall()
        for segment_id, segment_start, segment_end in rows:
            segment_duration = int(segment_end) - int(segment_start)
            if segment_duration <= 0:
                continue
            overlap = max(
                0,
                min(int(segment_end), end_ms) - max(int(segment_start), start_ms),
            )
            if overlap / min(segment_duration, sample_duration) < 0.8:
                continue
            conn.execute(
                f"""
                UPDATE {table}
                SET person_id=?, speaker_label=?
                WHERE {id_column}=?
                """,
                (new_person_id, new_person_name, segment_id),
            )


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


def list_voice_enrollments():
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT ve.embedding_id, ve.person_id, p.name, ve.embedding_path,
                   ve.audio_path, ve.quality, ve.created_at
            FROM voice_embeddings ve
            JOIN persons p ON p.person_id = ve.person_id
            ORDER BY ve.embedding_id DESC
            """
        ).fetchall()


def create_unknown_voice(label, embedding_path):
    with get_conn() as conn:
        return _insert_id(
            conn,
            """
            INSERT INTO unknown_voices(label, embedding_path, created_at)
            VALUES (?, ?, ?)
            """,
            (label, str(embedding_path) if embedding_path else None, utc_now()),
            "unknown_voice_id",
        )


def update_unknown_voice(unknown_voice_id, label, embedding_path):
    with get_conn() as conn:
        conn.execute(
            "UPDATE unknown_voices SET label=?, embedding_path=? WHERE unknown_voice_id=?",
            (label, str(embedding_path), int(unknown_voice_id)),
        )


def add_unknown_voice_sample(
    unknown_voice_id,
    embedding_path,
    quality=0.0,
    audio_path=None,
    source_ref=None,
    start_ms=None,
    end_ms=None,
):
    with get_conn() as conn:
        normalized_source_ref = str(source_ref) if source_ref else None
        normalized_start_ms = int(start_ms) if start_ms is not None else None
        normalized_end_ms = int(end_ms) if end_ms is not None else None
        if (
            normalized_source_ref is not None
            and normalized_start_ms is not None
            and normalized_end_ms is not None
        ):
            existing = conn.execute(
                """
                SELECT sample_id FROM unknown_voice_samples
                WHERE unknown_voice_id=? AND source_ref=? AND start_ms=? AND end_ms=?
                """,
                (
                    int(unknown_voice_id),
                    normalized_source_ref,
                    normalized_start_ms,
                    normalized_end_ms,
                ),
            ).fetchone()
            if existing is not None:
                return int(existing[0])

            unattached = conn.execute(
                """
                SELECT sample_id FROM unknown_voice_samples
                WHERE unknown_voice_id=? AND audio_path IS NULL AND source_ref IS NULL
                ORDER BY quality DESC, sample_id LIMIT 1
                """,
                (int(unknown_voice_id),),
            ).fetchone()
            if unattached is not None:
                conn.execute(
                    """
                    UPDATE unknown_voice_samples
                    SET embedding_path=?, audio_path=?, source_ref=?, start_ms=?,
                        end_ms=?, quality=?, created_at=?
                    WHERE sample_id=?
                    """,
                    (
                        str(embedding_path),
                        str(audio_path) if audio_path else None,
                        normalized_source_ref,
                        normalized_start_ms,
                        normalized_end_ms,
                        float(quality),
                        utc_now(),
                        int(unattached[0]),
                    ),
                )
                return int(unattached[0])

        return _insert_id(
            conn,
            """
            INSERT INTO unknown_voice_samples(
                unknown_voice_id, embedding_path, audio_path, source_ref,
                start_ms, end_ms, quality, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(unknown_voice_id),
                str(embedding_path),
                str(audio_path) if audio_path else None,
                normalized_source_ref,
                normalized_start_ms,
                normalized_end_ms,
                float(quality),
                utc_now(),
            ),
            "sample_id",
        )


def list_unknown_voice_samples(unknown_voice_id):
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT sample_id, embedding_path, quality, created_at, audio_path,
                   source_ref, start_ms, end_ms
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


def list_resolved_unknown_voice_samples_for_source(source_ref):
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT uvs.sample_id, uvs.embedding_path, uvs.start_ms, uvs.end_ms,
                   uv.resolved_person_id, ve.person_id
            FROM unknown_voice_samples uvs
            JOIN unknown_voices uv ON uv.unknown_voice_id = uvs.unknown_voice_id
            LEFT JOIN voice_embeddings ve
                ON ve.embedding_path = uvs.embedding_path
            WHERE uv.resolved_person_id IS NOT NULL
              AND uvs.source_ref=?
              AND uvs.start_ms IS NOT NULL
              AND uvs.end_ms IS NOT NULL
            ORDER BY uvs.sample_id, ve.embedding_id
            """,
            (str(source_ref),),
        ).fetchall()
        samples = {}
        for sample_id, embedding_path, start_ms, end_ms, resolved_person_id, profile_person_id in rows:
            sample = samples.setdefault(
                sample_id,
                {
                    "embedding_path": embedding_path,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "resolved_person_id": int(resolved_person_id),
                    "profile_person_ids": set(),
                },
            )
            if profile_person_id is not None:
                sample["profile_person_ids"].add(int(profile_person_id))

        effective_samples = []
        for sample_id, sample in samples.items():
            profile_person_ids = sample["profile_person_ids"]
            if len(profile_person_ids) > 1:
                logger.warning(
                    "Skipping resolved source sample %s with conflicting enrollment person IDs",
                    sample_id,
                )
                continue
            effective_person_id = (
                next(iter(profile_person_ids))
                if profile_person_ids
                else sample["resolved_person_id"]
            )
            effective_samples.append(
                (
                    effective_person_id,
                    sample["start_ms"],
                    sample["end_ms"],
                )
            )

        person_ids = {sample[0] for sample in effective_samples}
        if not person_ids:
            return []
        placeholders = ",".join("?" for _ in person_ids)
        names_by_person_id = {
            int(person_id): name
            for person_id, name in conn.execute(
                f"SELECT person_id, name FROM persons WHERE person_id IN ({placeholders})",
                tuple(sorted(person_ids)),
            ).fetchall()
        }
        return [
            (
                person_id,
                names_by_person_id[person_id],
                start_ms,
                end_ms,
            )
            for person_id, start_ms, end_ms in effective_samples
            if person_id in names_by_person_id
        ]


def get_unknown_voice_by_label(label):
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT unknown_voice_id, resolved_person_id
            FROM unknown_voices
            WHERE label=?
            """,
            (str(label),),
        ).fetchone()


def save_meeting_introduction(
    meeting_id,
    name,
    job_title,
    start_ms,
    end_ms,
    speaker_label,
    source_segment_ids,
    parser_confidence,
    status,
    voice_unknown_id=None,
    face_track_id=None,
    person_id=None,
    match_confidence=None,
    reason=None,
    face_unknown_id=None,
):
    start_ms = int(start_ms)
    end_ms = int(end_ms)
    if end_ms <= start_ms:
        raise ValueError("Introduction end time must be after its start time")
    source_segment_ids = [int(value) for value in source_segment_ids]
    if not source_segment_ids:
        raise ValueError("Introduction must reference at least one source segment")
    now = utc_now()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO meeting_introductions(
                meeting_id, name, job_title, start_ms, end_ms, speaker_label,
                source_segment_ids, voice_unknown_id, face_track_id, person_id,
                parser_confidence, match_confidence, status, reason,
                created_at, updated_at, face_unknown_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(meeting_id, name, start_ms, end_ms)
            DO UPDATE SET
                job_title=excluded.job_title,
                source_segment_ids=excluded.source_segment_ids,
                voice_unknown_id=COALESCE(
                    excluded.voice_unknown_id,
                    meeting_introductions.voice_unknown_id
                ),
                face_track_id=excluded.face_track_id,
                person_id=CASE
                    WHEN meeting_introductions.status='manually_confirmed'
                    THEN meeting_introductions.person_id
                    ELSE excluded.person_id
                END,
                parser_confidence=excluded.parser_confidence,
                match_confidence=excluded.match_confidence,
                status=CASE
                    WHEN meeting_introductions.status='manually_confirmed'
                    THEN meeting_introductions.status
                    ELSE excluded.status
                END,
                reason=CASE
                    WHEN meeting_introductions.status='manually_confirmed'
                    THEN meeting_introductions.reason
                    ELSE excluded.reason
                END,
                face_unknown_id=COALESCE(
                    excluded.face_unknown_id,
                    meeting_introductions.face_unknown_id
                ),
                updated_at=excluded.updated_at
            """,
            (
                int(meeting_id),
                str(name),
                str(job_title).strip() if job_title else None,
                int(start_ms),
                int(end_ms),
                str(speaker_label),
                json.dumps([int(value) for value in source_segment_ids]),
                int(voice_unknown_id) if voice_unknown_id is not None else None,
                int(face_track_id) if face_track_id is not None else None,
                int(person_id) if person_id is not None else None,
                float(parser_confidence),
                float(match_confidence) if match_confidence is not None else None,
                str(status),
                str(reason) if reason else None,
                now,
                now,
                int(face_unknown_id) if face_unknown_id is not None else None,
            ),
        )
        if person_id is not None and str(status).startswith(
            ("matched_", "voice_enrolled_")
        ):
            segment_ids = [int(value) for value in source_segment_ids]
            if segment_ids:
                placeholders = ",".join("?" for _ in segment_ids)
                person_name = conn.execute(
                    "SELECT name FROM persons WHERE person_id=?",
                    (int(person_id),),
                ).fetchone()
                if person_name is None:
                    raise ValueError(f"Person {person_id} does not exist")
                conn.execute(
                    f"""
                    UPDATE transcription_segments
                    SET speaker_label=?, person_id=?
                    WHERE meeting_id=? AND segment_id IN ({placeholders})
                    """,
                    (
                        person_name[0],
                        int(person_id),
                        int(meeting_id),
                        *segment_ids,
                    ),
                )
        return int(
            conn.execute(
                """
                SELECT introduction_id FROM meeting_introductions
                WHERE meeting_id=? AND name=? AND start_ms=? AND end_ms=?
                """,
                (
                    int(meeting_id),
                    str(name),
                    int(start_ms),
                    int(end_ms),
                ),
            ).fetchone()[0]
        )


def list_meeting_introductions(meeting_id):
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT introduction_id, name, job_title, start_ms, end_ms,
                   speaker_label, source_segment_ids, voice_unknown_id,
                   face_track_id, person_id, parser_confidence,
                   match_confidence, status, reason, face_unknown_id
            FROM meeting_introductions
            WHERE meeting_id=?
            ORDER BY start_ms, introduction_id
            """,
            (int(meeting_id),),
        ).fetchall()


def confirm_meeting_introduction(introduction_id, person_id):
    """Manually confirm a detected introduction for its source transcript rows."""
    introduction_id = int(introduction_id)
    person_id = int(person_id)
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT meeting_id, name, speaker_label, source_segment_ids,
                   voice_unknown_id, status, face_unknown_id
            FROM meeting_introductions WHERE introduction_id=?
            """,
            (introduction_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Introduction {introduction_id} does not exist")
        person = conn.execute(
            "SELECT name FROM persons WHERE person_id=?",
            (person_id,),
        ).fetchone()
        if person is None:
            raise ValueError(f"Person {person_id} does not exist")
        meeting_id = int(row[0])
        segment_ids = json.loads(row[3])
        if (
            not isinstance(segment_ids, list)
            or not segment_ids
            or any(type(value) is not int or value <= 0 for value in segment_ids)
        ):
            raise ValueError("Introduction has no valid source transcript segments")
        voice_unknown_id = int(row[4]) if row[4] is not None else None
        status = str(row[5])
        face_unknown_id = int(row[6]) if row[6] is not None else None
        person_name = person[0]
    if status == "pending_multiple_introductions":
        raise ValueError(
            "This transcript segment contains multiple introductions and must be "
            "split into speaker-specific segments before confirmation"
        )
    if voice_unknown_id is not None:
        with get_conn() as conn:
            unknown = conn.execute(
                """
                SELECT resolved_person_id FROM unknown_voices
                WHERE unknown_voice_id=?
                """,
                (voice_unknown_id,),
            ).fetchone()
            profile = conn.execute(
                """
                SELECT ve.embedding_id
                FROM voice_embeddings ve
                JOIN unknown_voice_samples uvs
                    ON uvs.embedding_path=ve.embedding_path
                WHERE uvs.unknown_voice_id=?
                ORDER BY uvs.quality DESC, ve.embedding_id
                LIMIT 1
                """,
                (voice_unknown_id,),
            ).fetchone()
        if unknown is None:
            raise ValueError(f"Unknown voice {voice_unknown_id} no longer exists")
        if unknown[0] is None:
            resolve_unknown_voice(voice_unknown_id, person_id=person_id)
        elif int(unknown[0]) != person_id:
            if profile is None:
                raise ValueError(
                    "The linked unknown voice has no enrolled sample to reassign"
                )
            reassign_voice_embedding(int(profile[0]), person_id)

            from face_voice_rec.domain.voice.voice_core import index_voice_enrollments_for_person

            index_voice_enrollments_for_person(person_id)

    if face_unknown_id is not None:
        from face_voice_rec.domain.face.face_core import enroll_unknown_face_track

        enroll_unknown_face_track(face_unknown_id, person_id)

    with get_conn() as conn:
        placeholders = ",".join("?" for _ in segment_ids)
        conn.execute(
            f"""
            UPDATE transcription_segments
            SET speaker_label=?, person_id=?
            WHERE meeting_id=? AND segment_id IN ({placeholders})
            """,
            (person_name, person_id, meeting_id, *segment_ids),
        )
        conn.execute(
            """
            UPDATE meeting_introductions
            SET person_id=?, status='manually_confirmed', reason=NULL,
                updated_at=?
            WHERE introduction_id=?
            """,
            (person_id, utc_now(), introduction_id),
        )
    _regenerate_meeting_transcripts([meeting_id])
    from face_voice_rec.domain.transcription import transcription_core

    transcription_core.refresh_readable_transcript(meeting_id, use_llm=False)
    return person_id


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


def resolve_unknown_voice(unknown_voice_id, person_id=None, new_person_name=None):
    meeting_ids = []
    with get_conn() as conn:
        if (person_id is None) == (new_person_name is None):
            raise ValueError("Provide either an existing person_id or a new person name")
        if new_person_name is not None:
            new_person_name = str(new_person_name).strip()
            if not new_person_name:
                raise ValueError("Person name cannot be empty")
            now = utc_now()
            conn.execute(
                """
                INSERT OR IGNORE INTO persons(name, created_at, updated_at)
                VALUES (?, ?, ?)
                """,
                (new_person_name, now, now),
            )
            person_id = conn.execute(
                "SELECT person_id FROM persons WHERE name=?",
                (new_person_name,),
            ).fetchone()[0]
        person_id = int(person_id)
        voice = conn.execute(
            """
            SELECT label, embedding_path, resolved_person_id
            FROM unknown_voices WHERE unknown_voice_id=?
            """,
            (int(unknown_voice_id),),
        ).fetchone()
        if voice is None:
            raise ValueError(f"Unknown voice {unknown_voice_id} no longer exists")
        if voice[2] is not None:
            raise ValueError(f"Unknown voice {unknown_voice_id} is already resolved")

        person = conn.execute(
            "SELECT name FROM persons WHERE person_id=?",
            (int(person_id),),
        ).fetchone()
        if person is None:
            raise ValueError(f"Person {person_id} does not exist")
        person_name = person[0]

        samples = conn.execute(
            """
            SELECT embedding_path, audio_path, quality
            FROM unknown_voice_samples
            WHERE unknown_voice_id=?
            ORDER BY quality DESC, sample_id
            """,
            (int(unknown_voice_id),),
        ).fetchall()
        profile_samples = {}
        for embedding_path, audio_path, quality in samples:
            if embedding_path:
                profile_samples.setdefault(
                    str(embedding_path),
                    (str(audio_path) if audio_path else None, float(quality or 0.0)),
                )
        if voice[1]:
            profile_samples.setdefault(str(voice[1]), (None, 0.7))
        if not profile_samples:
            raise ValueError(
                f"Unknown voice {unknown_voice_id} has no saved embedding to enroll"
            )
        for embedding_path, (audio_path, quality) in profile_samples.items():
            if not Path(embedding_path).is_file():
                raise FileNotFoundError(
                    f"Voice embedding for unknown voice {unknown_voice_id} is missing: "
                    f"{embedding_path}"
                )
            conn.execute(
                """
                INSERT INTO voice_embeddings(
                    person_id, embedding_path, audio_path, quality, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    int(person_id),
                    embedding_path,
                    audio_path,
                    quality,
                    utc_now(),
                ),
            )

        conn.execute(
            "UPDATE unknown_voices SET resolved_person_id=? WHERE unknown_voice_id=?",
            (int(person_id), int(unknown_voice_id)),
        )
        label = voice[0]
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
    return person_id


def _delete_unknown_voice_locked(conn, unknown_voice_id):
    parent = conn.execute(
        "SELECT embedding_path, resolved_person_id FROM unknown_voices WHERE unknown_voice_id=?",
        (int(unknown_voice_id),),
    ).fetchone()
    if parent is None or parent[1] is not None:
        return False

    sample_rows = conn.execute(
        "SELECT embedding_path, audio_path FROM unknown_voice_samples WHERE unknown_voice_id=?",
        (int(unknown_voice_id),),
    ).fetchall()
    paths = [parent[0]] + [path for row in sample_rows for path in row]

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


def log_audio(
    start_time,
    end_time,
    person_id,
    person_name,
    confidence,
    source,
    transcript=None,
    track_id=None,
    source_ref=None,
):
    normalized_source_ref = (
        _normalize_source_ref(source, source_ref)
        if source_ref is not None
        else None
    )
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO audio_logs(
                start_time, end_time, person_id, person_name,
                confidence, source, transcript, track_id, source_ref
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                float(start_time),
                float(end_time),
                person_id,
                person_name,
                float(confidence),
                source,
                transcript,
                track_id,
                normalized_source_ref,
            ),
        )


def fetch_audio_logs(
    source=None,
    person_id=None,
    from_time=None,
    to_time=None,
    source_ref=None,
):
    with get_conn() as conn:
        query = """
            SELECT log_id, start_time, end_time, person_name,
                   confidence, source, transcript, track_id
            FROM audio_logs
        """
        where_clauses = []
        params = []
        if source is not None:
            where_clauses.append("source=?")
            params.append(source)
        if source_ref is not None:
            where_clauses.append("source_ref=?")
            params.append(_normalize_source_ref(source, source_ref))
        if person_id is not None:
            if person_id == "UNKNOWN" or person_id == -1:
                where_clauses.append(
                    "(person_id IS NULL OR person_name='UNKNOWN' OR person_name IS NULL)"
                )
            else:
                where_clauses.append("person_id=?")
                params.append(int(person_id))
        if from_time is not None:
            where_clauses.append("start_time>=?")
            params.append(float(from_time))
        if to_time is not None:
            where_clauses.append("start_time<=?")
            params.append(float(to_time))
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        query += " ORDER BY start_time DESC"

        return conn.execute(query, params).fetchall()


def list_audio_log_video_sources():
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT source_ref
            FROM audio_logs
            WHERE source='video' AND source_ref IS NOT NULL
            ORDER BY source_ref
            """
        ).fetchall()
    return [row[0] for row in rows]


def _normalize_source_ref(source, source_ref):
    normalized = str(source_ref).strip()
    if not normalized:
        raise ValueError("source_ref cannot be empty")
    if source == "video":
        return Path(normalized).expanduser().resolve(strict=False).as_posix()
    return normalized


def _validate_attendance_times(source, observed_at_utc, media_offset_ms):
    if source not in {"realtime", "video"}:
        raise ValueError("source must be one of {'realtime', 'video'}")

    if source == "realtime":
        if not isinstance(observed_at_utc, str) or not observed_at_utc.strip():
            raise ValueError("realtime attendance requires observed_at_utc")
        if media_offset_ms is not None:
            raise ValueError("realtime attendance does not allow media_offset_ms")
        normalized = observed_at_utc.strip()
        if normalized.endswith("Z"):
            normalized = f"{normalized[:-1]}+00:00"

        try:
            observed_dt = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError("observed_at_utc must be a valid ISO-8601 datetime") from exc

        if observed_dt.tzinfo is None or observed_dt.utcoffset() is None:
            raise ValueError("observed_at_utc must be timezone-aware")

        return observed_dt.astimezone(timezone.utc).isoformat(), None

    if observed_at_utc is not None:
        raise ValueError("video attendance does not allow observed_at_utc")
    if media_offset_ms is None:
        raise ValueError("video attendance requires media_offset_ms")

    return None, _require_nonnegative_int(media_offset_ms, "media_offset_ms")


def _require_nonnegative_int(value, field_name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field_name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return normalized


def _validate_confidence(confidence):
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        raise ValueError("confidence must be a finite number in [0, 1]")
    normalized = float(confidence)
    if not math.isfinite(normalized):
        raise ValueError("confidence must be a finite number in [0, 1]")
    if normalized < 0.0 or normalized > 1.0:
        raise ValueError("confidence must be a finite number in [0, 1]")
    return normalized


def _validated_attendance_payload(
    *,
    person_id,
    person_name,
    source,
    source_ref,
    track_id,
    confidence,
    observed_at_utc,
    media_offset_ms,
):
    normalized_source_ref = _normalize_source_ref(source, source_ref)
    normalized_observed_at_utc, normalized_media_offset_ms = _validate_attendance_times(
        source,
        observed_at_utc,
        media_offset_ms,
    )

    if isinstance(person_id, bool) or not isinstance(person_id, Integral):
        raise ValueError("person_id must be an integer")

    normalized_person_name = str(person_name).strip()
    if not normalized_person_name:
        raise ValueError("person_name cannot be empty")

    return {
        "person_id": int(person_id),
        "person_name": normalized_person_name,
        "source": source,
        "source_ref": normalized_source_ref,
        "track_id": _require_nonnegative_int(track_id, "track_id"),
        "confidence": _validate_confidence(confidence),
        "observed_at_utc": normalized_observed_at_utc,
        "media_offset_ms": normalized_media_offset_ms,
    }


def _insert_attendance_event(
    conn,
    *,
    person_id,
    person_name,
    source,
    source_ref,
    track_id,
    confidence,
    observed_at_utc,
    media_offset_ms,
):
    payload = _validated_attendance_payload(
        person_id=person_id,
        person_name=person_name,
        source=source,
        source_ref=source_ref,
        track_id=track_id,
        confidence=confidence,
        observed_at_utc=observed_at_utc,
        media_offset_ms=media_offset_ms,
    )

    cur = conn.execute(
        """
        INSERT INTO attendance_events(
            person_id,
            person_name,
            source,
            source_ref,
            track_id,
            confidence,
            observed_at_utc,
            media_offset_ms,
            created_at
        )
        VALUES (:person_id, :person_name, :source, :source_ref, :track_id, :confidence, :observed_at_utc, :media_offset_ms, :created_at)
        ON CONFLICT(source, source_ref, track_id, person_id) DO NOTHING
        """,
        {
            **payload,
            "created_at": utc_now(),
        },
    )
    return cur.rowcount == 1


def record_attendance_event(
    person_id,
    person_name,
    source,
    source_ref,
    track_id,
    confidence,
    observed_at_utc=None,
    media_offset_ms=None,
):
    with get_conn() as conn:
        return _insert_attendance_event(
            conn,
            person_id=person_id,
            person_name=person_name,
            source=source,
            source_ref=source_ref,
            track_id=track_id,
            confidence=confidence,
            observed_at_utc=observed_at_utc,
            media_offset_ms=media_offset_ms,
        )


def replace_video_attendance(source_ref: str, events: Sequence[Mapping[str, object]]) -> int:
    normalized_source_ref = _normalize_source_ref("video", source_ref)
    normalized_events = [
        _validated_attendance_payload(
            person_id=event["person_id"],
            person_name=event["person_name"],
            source="video",
            source_ref=normalized_source_ref,
            track_id=event["track_id"],
            confidence=event["confidence"],
            observed_at_utc=event.get("observed_at_utc"),
            media_offset_ms=event.get("media_offset_ms"),
        )
        for event in events
    ]
    inserted = 0
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM attendance_events WHERE source='video' AND source_ref=?",
            (normalized_source_ref,),
        )
        for event in normalized_events:
            cur = conn.execute(
                """
                INSERT INTO attendance_events(
                    person_id,
                    person_name,
                    source,
                    source_ref,
                    track_id,
                    confidence,
                    observed_at_utc,
                    media_offset_ms,
                    created_at
                )
                VALUES (:person_id, :person_name, :source, :source_ref, :track_id, :confidence, :observed_at_utc, :media_offset_ms, :created_at)
                ON CONFLICT(source, source_ref, track_id, person_id) DO NOTHING
                """,
                {
                    **event,
                    "created_at": utc_now(),
                },
            )
            inserted += int(cur.rowcount == 1)
    return inserted


def _normalize_utc_iso_for_query(value, field_name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty ISO-8601 datetime")

    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"

    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid ISO-8601 datetime") from exc

    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")

    return dt.astimezone(timezone.utc)


def fetch_attendance(
    source=None,
    person_id=None,
    source_ref=None,
    observed_from_utc=None,
    observed_to_utc=None,
):
    query = """
        SELECT
            event_id,
            person_id,
            person_name,
            source,
            source_ref,
            track_id,
            confidence,
            observed_at_utc,
            media_offset_ms,
            created_at
        FROM attendance_events
    """
    where_clauses = []
    params = []

    if source is not None:
        where_clauses.append("source=?")
        params.append(source)
    if person_id is not None:
        where_clauses.append("person_id=?")
        params.append(int(person_id))
    if source_ref is not None:
        normalized_source_ref = _normalize_source_ref(source or "realtime", source_ref)
        where_clauses.append("source_ref=?")
        params.append(normalized_source_ref)

    if observed_from_utc is not None or observed_to_utc is not None:
        if source != "realtime":
            raise ValueError("start/end attendance period filters require source='realtime'")

        start_dt = None
        end_dt = None
        if observed_from_utc is not None:
            start_dt = _normalize_utc_iso_for_query(observed_from_utc, "observed_from_utc")
            where_clauses.append("observed_at_utc>=?")
            params.append(start_dt.isoformat())
        if observed_to_utc is not None:
            end_dt = _normalize_utc_iso_for_query(observed_to_utc, "observed_to_utc")
            where_clauses.append("observed_at_utc<=?")
            params.append(end_dt.isoformat())
        if start_dt is not None and end_dt is not None and start_dt > end_dt:
            raise ValueError("start datetime must be earlier than or equal to end datetime")

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    query += """
        ORDER BY
            CASE
                WHEN source='realtime' THEN observed_at_utc
                ELSE created_at
            END DESC,
            created_at DESC,
            event_id DESC
    """

    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(query, tuple(params)).fetchall()
