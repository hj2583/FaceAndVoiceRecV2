from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.queues import Queue
from pathlib import Path
from threading import Barrier

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
    database_path = tmp_path / "db" / "faces.db"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(database, "DB_PATH", database_path)
    raw_connect = database.sqlite3.connect

    def _zero_timeout_connect(*args, **kwargs):
        kwargs.setdefault("timeout", 0.0)
        return raw_connect(*args, **kwargs)

    monkeypatch.setattr(database.sqlite3, "connect", _zero_timeout_connect)
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


def test_init_db_serializes_spawn_process_initializers(tmp_path, monkeypatch):
    database_path = tmp_path / "db" / "faces.db"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(database, "DB_PATH", database_path)

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

    for process in processes:
        process.start()

    for process in processes:
        process.join(timeout=30)

    for process in processes:
        assert process.exitcode == 0

    results = [result_queue.get(timeout=10) for _ in range(4)]
    assert results == [None, None, None, None]

    _assert_compatibility_columns_exist_once(database_path)
