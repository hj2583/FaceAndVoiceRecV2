# SQLite Initialization Locking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent concurrent app/realtime startup lock failures and preserve user data/workflow state during migrations and lazy tab reruns.

**Architecture:** Serialize schema initialization with the existing process-local RLock and SQLite's bounded writer timeout/transaction. Serialize cross-process file/path migration with a persistent SQLite lock database under `data/`, complete recursive preflight before moves, and keep file/marker publication no-clobber. Preserve Streamlit widget state by rendering active and previously visited lazy pages; only never-visited pages remain unloaded.

**Tech Stack:** Python 3.10+, sqlite3, threading.RLock, Streamlit `st.cache_resource`, pathlib, os.link, pytest, multiprocessing.

## Global Constraints

- Preserve all seven tab labels, order, and workflows.
- Preserve recognition, transcription, attendance, CPU/GPU, and model behavior.
- Preserve all user data; never overwrite conflicting destinations.
- All DB/migration tests use temporary roots/databases; never initialize/migrate or modify live `data/` during implementation validation.
- Keep `sitecustomize.py` at the repository root.
- Keep Python `>=3.10` and Streamlit `>=1.40`; do not add dependencies or raise floors.
- Preserve database schema and voice/face recognition policy.
- Preserve existing uncommitted deletions (`app.py`, `prompt.txt`, `realtime.py`, `video_processor.py`, `video_workflow.py`), other untracked root modules, both plan files, and all runtime data.
- Run tests using `d:/Git/FaceAndVoiceRecV2/directmlvenv/Scripts/python.exe -m pytest tests/unit tests/integration`; bare root pytest has duplicate legacy collection failures outside this plan.

---

## Task 1: Test Concurrent Initialization Against a Legacy Schema

**Files:**
- Modify: `tests/unit/test_database_initialization.py`

**Interfaces:**
- `init_db() -> None` stays public.
- Tests use DB paths under `tmp_path / "init_race"` or `tmp_path / "process_race"`, not the autouse fixture path `tmp_path / "db" / "faces.db`.
- `_create_legacy_database(path: Path) -> None` creates the three tables with the exact old column omissions and one backfill row shown below.

- [ ] **Step 1: Update the test helper to seed a real legacy schema on a distinct temporary path before concurrency starts.** The old schema omits `audio_logs.track_id`, `transcription_segments.original_text`/`word_timestamps`, and `unknown_voice_samples.audio_path`/`source_ref`/`start_ms`/`end_ms`. Store one transcription row with `text="legacy transcript text"` so the existing `original_text` backfill is asserted after concurrent init.

```python
def _create_legacy_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(
            """
            CREATE TABLE audio_logs (
                log_id INTEGER PRIMARY KEY, start_time REAL NOT NULL,
                end_time REAL NOT NULL, person_id INTEGER, person_name TEXT,
                confidence REAL, source TEXT NOT NULL, transcript TEXT
            );
            CREATE TABLE transcription_segments (
                segment_id INTEGER PRIMARY KEY, meeting_id INTEGER NOT NULL,
                speaker_label TEXT NOT NULL, person_id INTEGER,
                start_ms INTEGER NOT NULL, end_ms INTEGER NOT NULL,
                text TEXT NOT NULL, confidence REAL, sentence_type TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE unknown_voices (
                unknown_voice_id INTEGER PRIMARY KEY, label TEXT NOT NULL,
                embedding_path TEXT, created_at TEXT NOT NULL,
                resolved_person_id INTEGER
            );
            CREATE TABLE unknown_voice_samples (
                sample_id INTEGER PRIMARY KEY, unknown_voice_id INTEGER NOT NULL,
                embedding_path TEXT NOT NULL, quality REAL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            INSERT INTO transcription_segments (
                segment_id, meeting_id, speaker_label, start_ms, end_ms,
                text, created_at
            ) VALUES (1, 1, 'speaker_1', 0, 1000, 'legacy transcript text', 'now');
            """
        )
```

- [ ] **Step 2: Update the four-thread test to use `tmp_path / "init_race" / "faces.db"`, call `_create_legacy_database()` before launching the barrier workers, and assert no exceptions, all four compatibility columns, one seeded row, and `original_text == text`.**
- [ ] **Step 3: Update the spawn test to use a separate `tmp_path / "process_race" / "faces.db"`, seed the same legacy schema before starting workers, and assert zero process errors plus preserved/backfilled legacy data. Keep the existing terminate/join cleanup in `finally`.**
- [ ] **Step 4: Run the focused initializer tests before changing production code.**

Run: `d:/Git/FaceAndVoiceRecV2/directmlvenv/Scripts/python.exe -m pytest tests/unit/test_database_initialization.py -q`

Expected: RED on unprotected legacy schema migration, demonstrating the tests cover concurrent `ALTER TABLE` and backfill rather than a preinitialized DB.

## Task 2: Serialize Schema Initialization

**Files:**
- Modify: `src/face_voice_rec/config/constants.py`
- Modify: `src/face_voice_rec/config/__init__.py`
- Modify: `src/face_voice_rec/storage/database.py`
- Test: `tests/unit/test_database_initialization.py`

**Interfaces:**
- `DB_BUSY_TIMEOUT_SECONDS = 30` remains exported from `face_voice_rec.config`.
- `_DB_LOCK` is shared by `init_db()` and `get_conn()`.

- [ ] **Step 1: Retain the lock-wait test: hold `_DB_LOCK` in the test thread, start an initializer worker, assert the sqlite connection spy is not called until the test releases the lock, then assert init succeeds.**
- [ ] **Step 2: In `init_db()`, acquire `_DB_LOCK` for the full operation; connect with `timeout=DB_BUSY_TIMEOUT_SECONDS`, set `PRAGMA busy_timeout` in milliseconds, set WAL and foreign keys before `BEGIN IMMEDIATE`, then run existing DDL/backfills and commit once. Roll back/re-raise on error and always close the connection. Do not weaken or swallow lock errors.**
- [ ] **Step 3: Run the thread/spawn regressions and existing DB tests.**

Run: `d:/Git/FaceAndVoiceRecV2/directmlvenv/Scripts/python.exe -m pytest tests/unit/test_database_initialization.py tests/unit/test_database_attendance.py tests/unit/test_database_audio_logs.py tests/unit/test_database_unknown_cleanup.py tests/unit/test_database_voice.py -q`

Expected: all tests pass, including legacy-column upgrade/backfill under thread/process contention.

- [ ] **Step 4: Commit schema locking.**

```powershell
git add src/face_voice_rec/storage/database.py src/face_voice_rec/config/constants.py src/face_voice_rec/config/__init__.py tests/unit/test_database_initialization.py
git commit -m "fix: serialize sqlite schema initialization"
```

## Task 3: Serialize SQLite Path-Reference Migration

**Files:**
- Modify: `src/face_voice_rec/config/migration.py`
- Test: `tests/integration/test_data_migration.py`

**Interfaces:**
- `_migrate_path_references()` uses `DB_BUSY_TIMEOUT_SECONDS`, matching `PRAGMA busy_timeout`, and `BEGIN IMMEDIATE`.
- Existing path maps, markers, separator preservation, and transaction semantics remain intact.

- [ ] **Step 1: Keep the trace-callback test asserting connect timeout, busy-timeout pragma, `BEGIN IMMEDIATE` before path-table introspection/select/update, exactly one rewrite, and marker written after commit.**
- [ ] **Step 2: Run the targeted test, then the migration suite after applying the same timeout and immediate transaction to `_migrate_path_references()`.**

Run: `d:/Git/FaceAndVoiceRecV2/directmlvenv/Scripts/python.exe -m pytest tests/integration/test_data_migration.py -q`

Expected: all migration tests pass with temporary SQLite DBs only.

- [ ] **Step 3: Commit path-reference writer serialization.**

```powershell
git add src/face_voice_rec/config/migration.py tests/integration/test_data_migration.py
git commit -m "fix: serialize sqlite path migration"
```

## Task 4: Harden Cross-Process File Migration

**Files:**
- Modify: `src/face_voice_rec/config/migration.py`
- Modify: `.gitignore`
- Test: `tests/integration/test_data_migration.py`

**Interfaces:**
- Add a persistent lock DB path under `data/`, e.g. `data/.legacy_migration_lock.sqlite`; serialize the complete file-move + path-reference migration with `BEGIN IMMEDIATE` and `DB_BUSY_TIMEOUT_SECONDS`.
- Use unique temporary marker names and no-clobber marker publication under the process lock.
- Conflicts leave all source contents untouched and never overwrite existing destinations.

- [ ] **Step 1: Add tests for a nested source symlink when destination root is absent and for a symlinked destination ancestor.** Use a portable monkeypatched `Path.is_symlink()` fixture for deterministic Windows behavior. Assert full preflight detects each conflict before any source move, marker write, or destination content change.
- [ ] **Step 2: Expand `_collect_conflicts_for_path()` to reject dangling source symlinks before `exists()`, reject destination symlinks and symlink ancestors, and recurse every source directory even when its destination directory does not exist. Check all nested sources/filesystem devices in preflight. Revalidate no-clobber conditions immediately before each file publication.**
- [ ] **Step 3: Add a concurrent-spawn migration test.** Create a temp project with representative artifacts and no markers; synchronize two spawned workers calling `migrate_legacy_data(project_root, data_root)`; assert both succeed, each file lands once, both markers are complete, and database path refs are repaired once. Use a synthetic temp SQLite DB only.
- [ ] **Step 4: Serialize `migrate_legacy_data()` across processes using a persistent SQLite lock database under the target `data/` root. Set timeout and busy timeout, acquire `BEGIN IMMEDIATE`, recheck markers and rerun preflight under the lock, then move files/repair references/write markers before releasing. Add the lock database and sidecars to `.gitignore`.**
- [ ] **Step 5: Replace the fixed marker `.tmp` name and `Path.replace` with a unique temp file created exclusively, flush/fsync it, then publish the marker via `os.link` without replacing an existing marker; accept an already-complete marker idempotently and always remove the temp link.**
- [ ] **Step 6: Run migration integration tests.**

Run: `d:/Git/FaceAndVoiceRecV2/directmlvenv/Scripts/python.exe -m pytest tests/integration/test_data_migration.py -q`

Expected: race, retry, symlink, and marker tests pass without production data access.

- [ ] **Step 7: Commit migration hardening and ignore rule.**

```powershell
git add src/face_voice_rec/config/migration.py tests/integration/test_data_migration.py .gitignore
git commit -m "fix: serialize legacy data migration across processes"
```

## Task 5: Cache App Schema Initialization

**Files:**
- Modify: `app/main.py`
- Modify: `tests/unit/test_app_navigation.py`

**Interfaces:**
- `initialize_application_database() -> None`, decorated with `@st.cache_resource`, calls `init_db()` once per Streamlit process.
- `main()` order remains `initialize_runtime_paths()` then cached app DB init then navigation.
- Realtime remains an independent process initializer.

- [ ] **Step 1: Keep the cache test that clears cache, calls the wrapper twice with patched `init_db`, asserts one call, and clears cache in `finally`.**
- [ ] **Step 2: Run navigation/cache tests, then commit only app/main and navigation test changes.**

Run: `d:/Git/FaceAndVoiceRecV2/directmlvenv/Scripts/python.exe -m pytest tests/unit/test_app_navigation.py tests/unit/test_database_initialization.py -q`

```powershell
git add app/main.py tests/unit/test_app_navigation.py
git commit -m "perf: cache app database initialization"
```

## Task 6: Preserve State for Previously Visited Lazy Tabs

**Files:**
- Modify: `app/main.py`
- Modify: `tests/unit/test_app_lazy_loading.py`

**Interfaces:**
- Store the set of page module paths ever activated in `st.session_state["_loaded_tab_modules"]`.
- On stateful-tab reruns, import/render the active page plus previously visited pages, while never importing/rendering an unvisited inactive page. On old Streamlit fallback, render all seven as before.

- [ ] **Step 1: Add a test that selects Realtime on the first render, then selects Video on the second render with the same loaded set. Assert the second render includes Realtime and Video but not any other unvisited page.**
- [ ] **Step 2: Update `_render_tab_pages()` and `render_application()` to add the active page to the per-session loaded set and render every page in that set. Preserve lazy import on first visit.**
- [ ] **Step 3: Run the lazy-loading and navigation test suites.**

Run: `d:/Git/FaceAndVoiceRecV2/directmlvenv/Scripts/python.exe -m pytest tests/unit/test_app_lazy_loading.py tests/unit/test_app_navigation.py -q`

- [ ] **Step 4: Commit widget-state preservation.**

```powershell
git add app/main.py tests/unit/test_app_lazy_loading.py
git commit -m "fix: preserve visited tab widget state"
```

## Final Verification

- [ ] Run: `d:/Git/FaceAndVoiceRecV2/directmlvenv/Scripts/python.exe -m pytest tests/unit tests/integration -q`.
- [ ] Verify every generated DB and filesystem test uses `tmp_path` and never touches live `data/`.
- [ ] Review migration marker race/symlink/retry tests and visited-tab state tests.
- [ ] Do not execute the app or v2 migration against the live database during development.