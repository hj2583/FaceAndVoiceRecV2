from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.queues import Queue
from pathlib import Path
from threading import Barrier

from face_voice_rec import config
from face_voice_rec.storage import database


def _assert_compatibility_columns_exist_once(database_path: Path) -> None:
    with sqlite3.connect(database_path) as conn:
        audio_columns = [
            row[1] for row in conn.execute("PRAGMA table_info(audio_logs)").fetchall()
        ]
        segment_columns = [
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(transcription_segments)"
            ).fetchall()
        ]
        unknown_voice_columns = [
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(unknown_voice_samples)"
            ).fetchall()
        ]

    assert audio_columns.count("track_id") == 1
    assert segment_columns.count("original_text") == 1
    assert unknown_voice_columns.count("source_ref") == 1


def _create_legacy_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(
            """
            CREATE TABLE audio_logs (
                log_id INTEGER PRIMARY KEY,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                person_id INTEGER,
                person_name TEXT,
                confidence REAL,
                source TEXT NOT NULL,
                transcript TEXT
            );
            CREATE TABLE transcription_segments (
                segment_id INTEGER PRIMARY KEY,
                meeting_id INTEGER NOT NULL,
                speaker_label TEXT NOT NULL,
                person_id INTEGER,
                start_ms INTEGER NOT NULL,
                end_ms INTEGER NOT NULL,
                text TEXT NOT NULL,
                confidence REAL,
                sentence_type TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE unknown_voices (
                unknown_voice_id INTEGER PRIMARY KEY,
                label TEXT NOT NULL,
                embedding_path TEXT,
                created_at TEXT NOT NULL,
                resolved_person_id INTEGER
            );
            CREATE TABLE unknown_voice_samples (
                sample_id INTEGER PRIMARY KEY,
                unknown_voice_id INTEGER NOT NULL,
                embedding_path TEXT NOT NULL,
                quality REAL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            INSERT INTO transcription_segments (
                segment_id,
                meeting_id,
                speaker_label,
                start_ms,
                end_ms,
                text,
                created_at
            ) VALUES (1, 1, 'speaker_1', 0, 1000, 'legacy transcript text', 'now');
            """
        )


def _assert_legacy_row_backfilled(database_path: Path) -> None:
    with sqlite3.connect(database_path) as conn:
        row = conn.execute(
            "SELECT text, original_text FROM transcription_segments WHERE segment_id = 1"
        ).fetchone()
    assert row == ("legacy transcript text", "legacy transcript text")


def _spawn_init_worker(database_path: str, barrier, result_queue: Queue) -> None:
    database.DB_PATH = Path(database_path)
    try:
        barrier.wait(timeout=10)
        database.init_db()
        result_queue.put(None)
    except Exception as error:  # pragma: no cover - exercised by failure mode
        result_queue.put(repr(error))
        raise


def test_init_db_serializes_concurrent_thread_initializers(tmp_path, monkeypatch):
    database_path = tmp_path / "init_race" / "faces.db"
    _create_legacy_database(database_path)
    monkeypatch.setattr(database, "DB_PATH", database_path)
    monkeypatch.setattr(config, "DB_PATH", database_path)
    errors = []

    for _ in range(10):
        barrier = Barrier(4)

        def initialize() -> None:
            barrier.wait(timeout=10)
            try:
                database.init_db()
            except Exception as error:
                errors.append(error)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(initialize) for _ in range(4)]
            for future in futures:
                future.result(timeout=20)

    assert errors == []
    _assert_compatibility_columns_exist_once(database_path)
    _assert_legacy_row_backfilled(database_path)


def test_init_db_waits_for_shared_db_lock(tmp_path, monkeypatch):
    database_path = tmp_path / "init_lock" / "faces.db"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(database, "DB_PATH", database_path)
    monkeypatch.setattr(config, "DB_PATH", database_path)

    connect_called = threading.Event()
    worker_started = threading.Event()
    worker_errors = []
    raw_connect = database.sqlite3.connect

    def _spy_connect(*args, **kwargs):
        connect_called.set()
        return raw_connect(*args, **kwargs)

    monkeypatch.setattr(database.sqlite3, "connect", _spy_connect)

    def _worker() -> None:
        worker_started.set()
        try:
            database.init_db()
        except Exception as error:
            worker_errors.append(error)

    worker = threading.Thread(target=_worker)
    database._DB_LOCK.acquire()
    try:
        worker.start()
        assert worker_started.wait(timeout=2)
        assert not connect_called.wait(timeout=0.25)
    finally:
        database._DB_LOCK.release()

    worker.join(timeout=10)
    assert not worker.is_alive()
    assert connect_called.wait(timeout=5)
    assert worker_errors == []
    _assert_compatibility_columns_exist_once(database_path)


def test_init_db_serializes_spawn_process_initializers(tmp_path, monkeypatch):
    database_path = tmp_path / "process_race" / "faces.db"
    _create_legacy_database(database_path)
    monkeypatch.setattr(database, "DB_PATH", database_path)
    monkeypatch.setattr(config, "DB_PATH", database_path)

    spawn_ctx = __import__("multiprocessing").get_context("spawn")
    barrier = spawn_ctx.Barrier(4)
    result_queue = spawn_ctx.Queue()

    processes = [
        spawn_ctx.Process(
            target=_spawn_init_worker,
            args=(str(database_path), barrier, result_queue),
        )
        for _ in range(4)
    ]

    try:
        for process in processes:
            process.start()

        for process in processes:
            process.join(timeout=30)

        for process in processes:
            assert process.exitcode == 0

        results = [result_queue.get(timeout=10) for _ in range(4)]
        assert results == [None, None, None, None]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        if hasattr(result_queue, "close"):
            result_queue.close()
        if hasattr(result_queue, "join_thread"):
            result_queue.join_thread()

    _assert_compatibility_columns_exist_once(database_path)
    _assert_legacy_row_backfilled(database_path)
