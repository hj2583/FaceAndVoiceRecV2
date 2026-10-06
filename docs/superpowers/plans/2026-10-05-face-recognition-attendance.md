# Face-Recognition Attendance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record one attendance event per known, confidently recognized tracked appearance from realtime and uploaded-video recognition, and display/filter/export those events in Streamlit.

**Architecture:** Add an `attendance_events` SQLite table and small database APIs. Realtime writes one event the first time a known track passes the existing confidence threshold, using a per-process run ID and UTC wall time. Video processing emits one candidate per track with a clip-relative offset; the workflow replaces attendance rows for that video only after the complete workflow returns successfully. The UI reads the dedicated table and keeps wall-clock timestamps distinct from clip offsets.

**Tech Stack:** Python, SQLite, OpenCV realtime/video pipelines, Streamlit, pytest. No new dependencies.

## Global Constraints

- Use the existing `config.RECOGNITION_THRESHOLD` (currently `0.70`); do not record unknown or below-threshold faces.
- Preserve the distinction between a persistent `person_id` and a run-local `track_id`.
- Realtime attendance uses UTC wall-clock time converted from the pipeline's monotonic clock.
- Uploaded-video attendance stores clip-relative milliseconds and no calendar observation timestamp.
- Reprocessing an uploaded video replaces its prior events only after the complete workflow succeeds; failures leave prior rows intact.
- Existing recognition and audio logs retain their current diagnostic meanings and schemas.
- An appearance is unique by `(source, source_ref, track_id, person_id)`; repeated samples for that identity/track do not create duplicate rows.
- Unknown and low-confidence recognitions never create attendance records.

---

### Task 1: Attendance persistence API

**Files:** Modify `database.py`, test `tests/test_database_attendance.py`.

**Interface:**

```python
def record_attendance_event(
    person_id: int,
    person_name: str,
    source: str,
    source_ref: str,
    track_id: int,
    confidence: float,
    observed_at_utc: str | None = None,
    media_offset_ms: int | None = None,
) -> bool: ...


def replace_video_attendance(source_ref: str, events: Sequence[Mapping[str, object]]) -> int: ...


def fetch_attendance(
    source: str | None = None,
    person_id: int | None = None,
    source_ref: str | None = None,
    observed_from_utc: str | None = None,
    observed_to_utc: str | None = None,
) -> list[sqlite3.Row]: ...
```

`record_attendance_event` returns true only when a new row was inserted; an identical source/run/track/person tuple is ignored. It normalizes aware realtime ISO-8601 input to UTC and rejects naive or malformed datetimes, confidence outside `[0, 1]`, and invalid offsets/track IDs. `replace_video_attendance` transactionally deletes prior `source='video'` rows for the exact normalized source path and inserts the supplied candidate set. Its caller invokes it only after successful video workflow completion. `fetch_attendance` orders realtime rows by normalized observation timestamp and video rows by import/creation time, newest first, and optionally filters by source/person.

- [x] **Step 1: Write migration and persistence tests**

Add tests in `tests/test_database_attendance.py`; define a local `_setup_database(tmp_path, monkeypatch)` helper that patches `database.DB_PATH` and calls `database.init_db()`. Cover idempotent `init_db`, source/timestamp constraints, UTC normalization/rejection, confidence/offset validation, one-time insert behavior, realtime/video timestamp fields, filtered reads, observation ordering, and transactional replacement. Build video paths under `tmp_path` and pass `Path.resolve()` rather than hard-coded Windows paths so tests are portable.

```python
import sqlite3
from pathlib import Path

import pytest

import database


def _setup_database(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "faces.db")
    database.init_db()


def test_attendance_event_is_idempotent_per_track(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alice")
    event = {
        "person_id": person_id,
        "person_name": "Alice",
        "source": "realtime",
        "source_ref": "run-1",
        "track_id": 4,
        "confidence": 0.91,
        "observed_at_utc": "2026-10-05T12:30:00+00:00",
        "media_offset_ms": None,
    }

    assert database.record_attendance_event(**event) is True
    assert database.record_attendance_event(**event) is False
    rows = database.fetch_attendance(source="realtime", person_id=person_id)
    assert len(rows) == 1
    assert rows[0]["track_id"] == 4
    assert rows[0]["observed_at_utc"] == event["observed_at_utc"]


def test_replace_video_attendance_replaces_only_matching_video(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alice")
    video_a = (tmp_path / "videos" / "a.mp4").resolve()
    video_b = (tmp_path / "videos" / "b.mp4").resolve()
    first = [{
        "person_id": person_id, "person_name": "Alice", "track_id": 1,
        "confidence": 0.9, "media_offset_ms": 1200,
    }]
    second = [{
        "person_id": person_id, "person_name": "Alice", "track_id": 2,
        "confidence": 0.95, "media_offset_ms": 2400,
    }]

    database.replace_video_attendance(str(video_a), first)
    database.replace_video_attendance(str(video_b), first)
    assert database.replace_video_attendance(str(video_a), second) == 1

    rows = database.fetch_attendance(source="video")
    assert [(row["source_ref"], row["track_id"], row["media_offset_ms"]) for row in rows] == [
        (video_a.as_posix(), 2, 2400), (video_b.as_posix(), 1, 1200),
    ]
```

Add this rollback test: the invalid foreign key fails after the prior source rows have been deleted and a new row has been inserted inside the transaction, proving the whole replacement rolls back.

```python
def test_replace_video_attendance_rolls_back_on_invalid_candidate(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alice")
    old = [{
        "person_id": person_id, "person_name": "Alice", "track_id": 1,
        "confidence": 0.9, "media_offset_ms": 1000,
    }]
    video_a = (tmp_path / "videos" / "a.mp4").resolve()
    database.replace_video_attendance(str(video_a), old)

    candidates = [
        {"person_id": person_id, "person_name": "Alice", "track_id": 2,
         "confidence": 0.95, "media_offset_ms": 2000},
        {"person_id": 999999, "person_name": "Missing", "track_id": 3,
         "confidence": 0.9, "media_offset_ms": 3000},
    ]
    with pytest.raises(sqlite3.IntegrityError):
        database.replace_video_attendance(str(video_a), candidates)

    rows = database.fetch_attendance(source="video")
    assert [(row["track_id"], row["media_offset_ms"]) for row in rows] == [(1, 1000)]
```

Add tests for observation-time ordering and reprocessing edge cases:

```python
def test_fetch_attendance_orders_realtime_by_observed_time(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    first = database.create_person("First")
    second = database.create_person("Second")
    database.record_attendance_event(
        person_id=second, person_name="Second", source="realtime",
        source_ref="run", track_id=2, confidence=0.9,
        observed_at_utc="2026-10-05T12:31:00+00:00",
    )
    database.record_attendance_event(
        person_id=first, person_name="First", source="realtime",
        source_ref="run", track_id=1, confidence=0.9,
        observed_at_utc="2026-10-05T12:30:00+00:00",
    )

    assert [row["person_name"] for row in database.fetch_attendance()] == [
        "Second", "First",
    ]


def test_empty_video_replacement_clears_only_that_video(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alice")
    video_a = str((tmp_path / "videos" / "a.mp4").resolve())
    video_b = str((tmp_path / "videos" / "b.mp4").resolve())
    event = {
        "person_id": person_id, "person_name": "Alice", "track_id": 1,
        "confidence": 0.9, "media_offset_ms": 1000,
    }
    database.replace_video_attendance(video_a, [event])
    database.replace_video_attendance(video_b, [event])

    assert database.replace_video_attendance(video_a, []) == 0
    assert [row["source_ref"] for row in database.fetch_attendance(source="video")] == [
        Path(video_b).as_posix(),
    ]


def test_duplicate_candidates_in_one_video_replacement_are_inserted_once(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alice")
    video_path = str((tmp_path / "videos" / "a.mp4").resolve())
    event = {
        "person_id": person_id, "person_name": "Alice", "track_id": 1,
        "confidence": 0.9, "media_offset_ms": 1000,
    }

    assert database.replace_video_attendance(video_path, [event, event]) == 1
    assert len(database.fetch_attendance(source="video")) == 1
```

Add parametrized validation cases asserting an offset-aware non-UTC realtime timestamp is stored normalized to `+00:00`, a naive timestamp raises `ValueError`, confidence rejects NaN and values outside `[0, 1]`, and fractional/negative/bool track IDs and fractional/negative offsets raise `ValueError`.

- [x] **Step 2: Run tests and verify RED**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_database_attendance.py -q`

Expected: collection or assertions fail because the attendance table/APIs do not exist.

- [x] **Step 3: Add the schema and database APIs**

Add `attendance_events` in `init_db()` using the existing foreign-key/SQLite conventions. Include source `CHECK`, non-null person/source/source_ref/track/confidence/created time, nullable source-specific time fields, and a unique constraint on `(source, source_ref, track_id, person_id)`. Use `get_conn()` so an exception rolls back the replacement delete and inserts. Reject malformed source/time combinations with `ValueError` before insert. Validate confidence as a finite numeric value in `[0, 1]` and `media_offset_ms` as a non-negative integer (reject bools/fractional values). Parse realtime timestamps with `datetime.fromisoformat`, require timezone awareness, normalize to UTC, and store a canonical `+00:00` ISO value. Use `ON CONFLICT(source, source_ref, track_id, person_id) DO NOTHING` instead of broad `INSERT OR IGNORE`, so unrelated CHECK/FK/NOT NULL errors are not silently hidden. Do not modify `recognition_logs` or `audio_logs` schemas.

- [x] **Step 4: Run database attendance tests**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_database_attendance.py -q`

Expected: migration, idempotency, timestamp validation, filtering, replacement, and rollback tests pass.

- [x] **Step 5: Commit**

```powershell
git add database.py tests/test_database_attendance.py
git commit -m "feat(attendance): add attendance event persistence"
```

---

### Task 2: Realtime attendance events

**Files:** Modify `realtime.py`, test `tests/test_realtime.py`.

**Consumes:** `database.record_attendance_event(...)`; `config.RECOGNITION_THRESHOLD`.

**Behavior:** Generate one UUID run ID per `realtime.run()` invocation. Keep a set of `(track_id, person_id)` pairs already recorded for attendance. At the first recognition sample for a pair where `person_id is not None` and confidence is at least `RECOGNITION_THRESHOLD`, record one event with source `realtime`, that run ID, track/person/name/confidence, and `_mono_to_wall_clock(timestamp)` converted to UTC ISO-8601. Keep existing per-second `log_recognition()` calls unchanged. A new track ID in the same run is a new appearance.

- [x] **Step 1: Add failing realtime tests**

Extend the realtime run tests with a fake tracker/person track and monkeypatched `record_attendance_event`. Provide enough frames for repeated recognition samples. Assert exactly one attendance call for the same track, a different track produces a second call, and unknown/low-confidence tracks produce none. Assert source/ref/track/person/confidence fields and an ISO-8601 UTC time derived from the existing monotonic-to-wall-clock offset. Keep existing recognition/audio-log assertions.

- [x] **Step 2: Run the focused test and verify RED**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_realtime.py -k attendance -q`

Expected: fails because realtime has no attendance recording yet.

- [x] **Step 3: Add first-per-track attendance recording**

Create the run ID once after realtime starts. Create `attendance_logged_pairs` beside the existing `last_recognition_log` runtime state. At an eligible known recognition, check/add `(track.track_id, track.person_id)` and call `record_attendance_event` once per pair with `observed_at_utc=datetime.fromtimestamp(_mono_to_wall_clock(timestamp), timezone.utc).isoformat()`. Do not repurpose `log_recognition` or change existing log cadence.

- [x] **Step 4: Run realtime tests**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_realtime.py -q`

Expected: attendance tests and existing realtime tests pass.

- [x] **Step 5: Commit**

```powershell
git add realtime.py tests/test_realtime.py
git commit -m "feat(attendance): record realtime recognized appearances"
```

---

### Task 3: Uploaded-video attendance events

**Files:** Modify `video_processor.py`, `video_workflow.py`, tests `tests/test_video_processor.py` and `tests/test_video_workflow.py`.

**Consumes:** `database.replace_video_attendance(source_ref, events)` from Task 1.

**Behavior:** Add a top-level `attendance` candidate list to the existing video JSON log, leaving `recognition` and `speech` unchanged. Capture only the first frame per `(track_id, person_id)` pair at which its `person_id` is set and confidence meets `RECOGNITION_THRESHOLD`; candidate fields are `person_id`, `person_name`, `track_id`, `confidence`, and `media_offset_ms=round(timestamp_sec*1000)`. After subtitle/audio finalization succeeds and immediately before returning the meeting ID, `process_video_and_transcribe()` resolves the source path and calls `replace_video_attendance()` once. If face processing, transcription, cue writing, subtitle/audio mux, or fallback finalization raises, do not replace prior attendance for that video.

- [x] **Step 1: Write failing video candidate tests**

Add a video-processor test with multiple frames on one known track, a second known track, an unknown track, and one low-confidence known track. Assert the new `attendance` array contains one first-seen candidate for each eligible known track with the expected media offset, while existing `recognition` and `speech` keys retain their current meanings.

Add workflow tests: successful workflow calls replacement once with the resolved video path and parsed candidates; transcription/final-mux failure does not call replacement and leaves prior data untouched. Keep the existing partial-output and one-transcription assertions.

- [x] **Step 2: Run focused tests and verify RED**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_video_processor.py tests/test_video_workflow.py -k attendance -q`

Expected: candidate log and post-success database replacement assertions fail.

- [x] **Step 3: Emit first-seen candidates in the video log**

Maintain an `attendance_candidates` list and `attendance_pairs` set in `process_video_pipeline()`. At each detected track, form `(track.track_id, track.person_id)` and append only if that pair is newly eligible under the global constraints. Add the list as the top-level `attendance` key; do not change `recognition` row fields or active-speaker events.

- [x] **Step 4: Replace video attendance only after full workflow success**

After final mux succeeds, read the candidate list from the JSON log, normalize the input video with `Path.resolve()`, and call `database.replace_video_attendance(...)`. Do not call it in any exception/fallback branch. An empty successful candidate list intentionally clears prior attendance for that video.

- [x] **Step 5: Run video attendance tests**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_video_processor.py tests/test_video_workflow.py -q`

Expected: event capture, success-only replacement, retry/reprocess, and existing workflow tests pass.

- [x] **Step 6: Commit**

```powershell
git add video_processor.py video_workflow.py tests/test_video_processor.py tests/test_video_workflow.py
git commit -m "feat(attendance): record uploaded video appearances"
```

---

### Task 4: Attendance view and CSV export

**Files:** Modify `database.py` and `app.py`; create `attendance_view.py`; test `tests/test_database_attendance.py` and `tests/test_attendance_view.py`.

**Consumes:** Extend `database.fetch_attendance(source=None, person_id=None, source_ref=None, observed_from_utc=None, observed_to_utc=None)`. Bounds apply inclusively to realtime `observed_at_utc`; `source_ref` selects a single uploaded video.

**Produces:** `local_datetime_to_utc_iso(value: datetime) -> str`, `local_datetime_range_to_utc_iso(start: datetime, end: datetime) -> tuple[str, str]`, `filter_visible_attendance(rows) -> list`, `build_attendance_frame(rows) -> pandas.DataFrame`, and `attendance_frame_to_csv(frame) -> bytes` in `attendance_view.py`. Row helpers accept mapping rows and `sqlite3.Row` objects returned by `fetch_attendance()`.

**Behavior:** Add an `Attendance` tab with a horizontal source selector (`Realtime`, `Uploaded video`) and a person filter. In Realtime mode, show exact `st.datetime_input` start/end controls labeled machine-local time, defaulting to the last 24 hours. Convert both inclusive boundaries with `local_datetime_to_utc_iso()` before querying; show a warning and no rows if start is after end. In Uploaded video mode, require one source reference selected from distinct stored video paths; label options with basename plus parent path to distinguish duplicate filenames. Show a dense table with Person, Source, Observed (UTC), Video, Source Ref, Clip Time, Confidence, and Track ID. Realtime rows show run ID in Source Ref; video rows show full normalized path in Source Ref and clip-relative `mm:ss` without a calendar date. The CSV exports exactly the selected source, person, and period/video rows. Empty video history and empty query results have explicit states. Exclude unknown/empty identities, but do not reapply the current confidence threshold at display time. Escape formula-leading text cells in CSV.

- [x] **Step 1: Add attendance query/view tests**

Add `tests/test_attendance_view.py`:

```python
from attendance_view import (
    attendance_frame_to_csv,
    build_attendance_frame,
    filter_visible_attendance,
    format_clip_offset,
)


def test_attendance_frame_keeps_realtime_dates_separate_from_video_offsets():
    frame = build_attendance_frame([
        {
            "person_name": "Alice", "source": "realtime",
            "observed_at_utc": "2026-10-05T12:30:00+00:00",
            "media_offset_ms": None, "confidence": 0.91, "track_id": 4,
            "source_ref": "run-1",
        },
        {
            "person_name": "Alice", "source": "video",
            "observed_at_utc": None, "media_offset_ms": 60000,
            "confidence": 0.95, "track_id": 2,
            "source_ref": "C:/videos/a.mp4",
        },
    ])

    assert list(frame.columns) == [
        "Person", "Source", "Observed (UTC)", "Video", "Source Ref",
        "Clip Time", "Confidence", "Track ID",
    ]
    assert frame.iloc[0]["Observed (UTC)"] == "2026-10-05T12:30:00+00:00"
    assert frame.iloc[0]["Source Ref"] == "run-1"
    assert frame.iloc[0]["Clip Time"] == ""
    assert frame.iloc[1]["Observed (UTC)"] == ""
    assert frame.iloc[1]["Video"] == "a.mp4"
    assert frame.iloc[1]["Source Ref"] == "C:/videos/a.mp4"
    assert frame.iloc[1]["Clip Time"] == "01:00"


def test_clip_offset_formatting_handles_minute_and_hour_boundaries():
    assert format_clip_offset(0) == "00:00"
    assert format_clip_offset(59999) == "00:59"
    assert format_clip_offset(60000) == "01:00"
    assert format_clip_offset(3600000) == "1:00:00"


def test_attendance_filter_excludes_unresolved_but_keeps_historical_confidence():
    rows = [
        {"person_id": 1, "person_name": "Alice", "confidence": 0.40},
        {"person_id": None, "person_name": "UNKNOWN", "confidence": 0.95},
        {"person_id": None, "person_name": "Bob", "confidence": 0.95},
        {"person_id": 2, "person_name": "", "confidence": 0.95},
        {"person_id": 3, "person_name": "Missing confidence", "confidence": None},
    ]

    assert filter_visible_attendance(rows) == [rows[0]]


def test_attendance_csv_exports_exact_columns_and_sanitizes_formulas():
    frame = build_attendance_frame([{
        "person_name": "=1+1", "person_id": 1, "source": "video", "observed_at_utc": None,
        "media_offset_ms": 0, "confidence": 0.9, "track_id": 1,
        "source_ref": "C:/videos/a.mp4",
    }])

    csv_text = attendance_frame_to_csv(frame).decode("utf-8-sig")
    assert csv_text.splitlines()[0] == (
        "Person,Source,Observed (UTC),Video,Source Ref,Clip Time,Confidence,Track ID"
    )
    assert "'=1+1,video,,a.mp4,C:/videos/a.mp4,00:00,0.9,1" in csv_text
```

Add a DB integration test in `tests/test_attendance_view.py`: create a person/event with `database.record_attendance_event`, call `database.fetch_attendance()` to get actual `sqlite3.Row` objects, pass them through `filter_visible_attendance`, `build_attendance_frame`, and `attendance_frame_to_csv`, and assert the row survives and appears in the CSV. Add filter tests that exclude null `person_id`, empty/UNKNOWN names, and missing confidence, but retain a known historical confidence below the current threshold. Add a CSV test that prefixes text cells beginning with `=`, `+`, `-`, or `@` with an apostrophe; do not alter numeric confidence/track cells.

Add tests in `tests/test_database_attendance.py` for `source_ref` video selection and inclusive realtime `observed_from_utc`/`observed_to_utc` bounds, including events exactly on both boundaries. Verify an invalid reversed range raises `ValueError`. Add tests in `tests/test_attendance_view.py` for `local_datetime_to_utc_iso()` using an aware non-UTC datetime and a naive datetime interpreted in machine-local timezone.

```python
import database


def test_sqlite_rows_flow_through_filter_frame_and_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "faces.db")
    database.init_db()
    person_id = database.create_person("Alice")
    database.record_attendance_event(
        person_id=person_id, person_name="Alice", source="realtime",
        source_ref="run-1", track_id=1, confidence=0.9,
        observed_at_utc="2026-10-05T12:30:00+00:00",
    )

    rows = database.fetch_attendance()
    visible = filter_visible_attendance(rows)
    frame = build_attendance_frame(visible)
    csv_text = attendance_frame_to_csv(frame).decode("utf-8-sig")

    assert len(frame) == 1
    assert "Alice,realtime,2026-10-05T12:30:00+00:00,run-1" in csv_text
```

- [x] **Step 2: Run focused tests and verify RED**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_attendance_view.py -q`

Expected: the new SQLite-row integration test fails when `.get()` is called on `sqlite3.Row`; the Source Ref assertion and CSV formula-safety assertion also fail against the current frame/export helpers.

- [x] **Step 3: Add the tab and view**

Add `local_datetime_to_utc_iso`, `filter_visible_attendance`, `format_clip_offset`, `build_attendance_frame`, and `attendance_frame_to_csv` in `attendance_view.py`. Filter to non-null person IDs, non-empty/non-UNKNOWN names, and present confidence; do not reapply the current threshold to historical rows. Format offsets as `MM:SS`, switching to `H:MM:SS` at one hour; preserve UTC values in realtime rows and keep video date fields empty. Extend `database.fetch_attendance()` with optional source reference and inclusive UTC bounds, validating `start <= end`. In `app.py`, add `render_attendance()` and register an `Attendance` tab. Unpack all four values returned by `list_persons()`. Use Realtime/Uploaded video mode selection. Realtime mode uses two machine-local `st.datetime_input` controls (default last 24 hours), converts them to UTC, and queries inclusively; reversed values show a warning without querying. Video mode requires selecting one stored source reference, with options labeled by basename plus parent path. Keep the person filter in both modes, apply only identity/presence filtering, and export exactly the displayed frame.

- [x] **Step 4: Run attendance/database UI-preparation tests**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_attendance_view.py -q`

Expected: SQLite-row integration, historical visibility, time formatting, and CSV-safety tests pass.

- [x] **Step 5: Commit**

```powershell
git add app.py attendance_view.py tests/test_attendance_view.py
git commit -m "feat(attendance): add attendance list and export"
```

---

### Task 5: Filter attendance by video or realtime period

**Files:** Modify `database.py`, `app.py`, and `attendance_view.py`; test `tests/test_database_attendance.py` and `tests/test_attendance_view.py`.

**Interface:** Extend `fetch_attendance` without breaking existing calls:

```python
def fetch_attendance(
    source: str | None = None,
    person_id: int | None = None,
    source_ref: str | None = None,
    observed_from_utc: str | None = None,
    observed_to_utc: str | None = None,
) -> list[sqlite3.Row]: ...


def local_datetime_to_utc_iso(value: datetime) -> str: ...
def local_datetime_range_to_utc_iso(start: datetime, end: datetime) -> tuple[str, str]: ...
```

- [ ] **Step 1: Write failing database query tests**

Insert realtime attendance at 09:00, 10:00, and 11:00 UTC for one person, plus another person's row at 10:00. Query source `realtime`, that person, and inclusive bounds 10:00 through 11:00; assert the exact-boundary 10:00 and 11:00 rows return, but 09:00 and the other person do not. Insert video rows for two source references; assert `source_ref` selects only the requested video. Assert `observed_from_utc > observed_to_utc` raises `ValueError`. Keep the existing no-argument and source/person query tests passing.

Use this database query test shape in `tests/test_database_attendance.py`:

```python
def test_fetch_attendance_filters_inclusive_period_and_video_path(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alice")
    video_a = str((tmp_path / "videos" / "a.mp4").resolve())
    video_b = str((tmp_path / "videos" / "b.mp4").resolve())

    for minute in (9, 10, 11):
        database.record_attendance_event(
            person_id=person_id, person_name="Alice", source="realtime",
            source_ref="run-1", track_id=minute, confidence=0.9,
            observed_at_utc=f"2026-10-06T10:{minute:02d}:00+00:00",
        )
    video_event = {
        "person_id": person_id, "person_name": "Alice", "track_id": 1,
        "confidence": 0.9, "media_offset_ms": 1000,
    }
    database.replace_video_attendance(video_a, [video_event])
    database.replace_video_attendance(video_b, [video_event])

    realtime_rows = database.fetch_attendance(
        source="realtime", person_id=person_id,
        observed_from_utc="2026-10-06T10:10:00+00:00",
        observed_to_utc="2026-10-06T10:11:00+00:00",
    )
    assert [row["track_id"] for row in realtime_rows] == [11, 10]
    assert [row["source_ref"] for row in database.fetch_attendance(
        source="video", source_ref=video_a,
    )] == [video_a]

    with pytest.raises(ValueError, match="start.*end"):
        database.fetch_attendance(
            source="realtime",
            observed_from_utc="2026-10-06T11:00:00+00:00",
            observed_to_utc="2026-10-06T10:00:00+00:00",
        )
```

- [ ] **Step 2: Run database query tests and verify RED**

Run: `.\directmlvenv\\Scripts\\python.exe -m pytest tests/test_database_attendance.py -k "period or source_ref" -q`

Expected: `fetch_attendance` rejects the new filter arguments or lacks inclusive bounds/source-ref filtering.

- [ ] **Step 3: Extend the attendance query**

Add optional `source_ref`, `observed_from_utc`, and `observed_to_utc` parameters to `database.fetch_attendance`. Add parameterized SQL predicates, require datetime bounds only for realtime queries, and reject reversed bounds before executing SQL. Use inclusive `>=`/`<=` comparisons against canonical UTC ISO values. Existing callers with only source/person continue to work.

- [ ] **Step 4: Add failing local-time conversion and mode tests**

Add `local_datetime_to_utc_iso` tests: an aware datetime with a non-UTC offset converts to UTC; a naive datetime is interpreted in the machine's local timezone and converted to UTC. Add `local_datetime_range_to_utc_iso(start, end)` tests for inclusive UTC output and `ValueError` on reversed input. In `app.py`, make the Attendance source selector choose exactly `Realtime` or `Uploaded video`. Realtime shows two `st.datetime_input` controls labeled machine-local time, defaulting from now minus 24 hours to now. Video mode requires one selection from stored video source refs, with option labels combining basename and parent path to distinguish equal filenames. Both modes retain person filtering. A reversed realtime range shows a warning and performs no attendance query; a selected video calls the query with that exact source ref. Keep empty states and CSV export of the exact displayed rows.

Use these helper tests in `tests/test_attendance_view.py`:

```python
from datetime import datetime, timedelta, timezone

import pytest

from attendance_view import local_datetime_range_to_utc_iso, local_datetime_to_utc_iso


def test_local_datetime_range_converts_to_utc_inclusively():
    start = datetime(2026, 10, 6, 9, 0, tzinfo=timezone(timedelta(hours=8)))
    end = datetime(2026, 10, 6, 10, 0, tzinfo=timezone(timedelta(hours=8)))
    assert local_datetime_to_utc_iso(start) == "2026-10-06T01:00:00+00:00"
    assert local_datetime_range_to_utc_iso(start, end) == (
        "2026-10-06T01:00:00+00:00", "2026-10-06T02:00:00+00:00",
    )


def test_local_datetime_range_rejects_reversed_endpoints():
    start = datetime(2026, 10, 6, 11, 0)
    end = datetime(2026, 10, 6, 10, 0)
    with pytest.raises(ValueError, match="start.*end"):
        local_datetime_range_to_utc_iso(start, end)
```

Also assert that a naive datetime converts to `value.astimezone(timezone.utc).isoformat()` in the current host timezone.

- [ ] **Step 5: Run attendance query and view tests**

Run: `.\directmlvenv\\Scripts\\python.exe -m pytest tests/test_database_attendance.py tests/test_attendance_view.py -q`

Expected: inclusive bounds, path filtering, local-to-UTC conversion, source-specific controls, and existing SQLite-row/CSV tests pass.

- [ ] **Step 6: Commit**

```powershell
git add database.py app.py attendance_view.py tests/test_database_attendance.py tests/test_attendance_view.py
git commit -m "feat(attendance): filter by video or realtime period"
```

---

### Task 6: Full regression and bounded smoke test

**Files:** No new feature surface; run validation and record evidence in the task report.

- [ ] **Step 1: Compile and run the full suite**

Run: `.\directmlvenv\Scripts\python.exe -m py_compile app.py database.py realtime.py video_processor.py video_workflow.py`

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests -q`

Expected: all tests pass; record actual count.

- [ ] **Step 2: Verify a bounded realtime attendance event**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_realtime.py -k attendance -q`

Expected: duplicate samples for one pair create one UTC attendance row; a second track creates a second row; unknown and below-threshold observations create none. Then run the full realtime module.

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_realtime.py -q`

- [ ] **Step 3: Verify uploaded-video replacement behavior**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_video_processor.py tests/test_video_workflow.py -k attendance -q`

Then run both complete test modules. Also perform one bounded run on `initialVideo/NEWS Why LPI Capital and not other insurers - The Edge TV (1080p).mp4` with temporary DB/output/log paths. Verify first-seen track offsets, persisted rows only after full workflow success, reprocessing replaces rather than duplicates rows, and a forced workflow failure leaves prior rows unchanged.

- [ ] **Step 4: Verify the attendance view/export**

Confirm realtime UTC dates and video clip offsets render in separate columns; filter by source/person and verify CSV contains the same filtered rows.

Do not add generated DBs, videos, or exports to git. Preserve any unrelated worktree changes.
