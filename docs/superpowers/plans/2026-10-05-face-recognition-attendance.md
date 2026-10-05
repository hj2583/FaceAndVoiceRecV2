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
record_attendance_event(
    person_id: int,
    person_name: str,
    source: str,
    source_ref: str,
    track_id: int,
    confidence: float,
    observed_at_utc: str | None = None,
    media_offset_ms: int | None = None,
) -> bool

replace_video_attendance(source_ref: str, events: Sequence[Mapping[str, object]]) -> int
fetch_attendance(source: str | None = None, person_id: int | None = None) -> list[sqlite3.Row]
```

`record_attendance_event` returns true only when a new row was inserted; an identical source/run/track/person tuple is ignored. `replace_video_attendance` transactionally deletes prior `source='video'` rows for the exact normalized source path and inserts the supplied candidate set. Its caller invokes it only after successful video workflow completion. `fetch_attendance` returns newest observations first and optionally filters by source/person.

- [ ] **Step 1: Write migration and persistence tests**

Add tests in `tests/test_database_attendance.py` using the existing `_setup_database(tmp_path, monkeypatch)` pattern. Cover idempotent `init_db`, source/timestamp constraints, one-time insert behavior, realtime/video timestamp fields, filtered reads, and transactional replacement.

```python
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
    first = [{
        "person_id": person_id, "person_name": "Alice", "track_id": 1,
        "confidence": 0.9, "media_offset_ms": 1200,
    }]
    second = [{
        "person_id": person_id, "person_name": "Alice", "track_id": 2,
        "confidence": 0.95, "media_offset_ms": 2400,
    }]

    database.replace_video_attendance("C:/videos/a.mp4", first)
    database.replace_video_attendance("C:/videos/b.mp4", first)
    assert database.replace_video_attendance("C:/videos/a.mp4", second) == 1

    rows = database.fetch_attendance(source="video")
    assert [(row["source_ref"], row["track_id"], row["media_offset_ms"]) for row in rows] == [
        ("C:/videos/a.mp4", 2, 2400), ("C:/videos/b.mp4", 1, 1200),
    ]
```

Add this rollback test: the invalid foreign key fails after the prior source rows have been deleted and a new row has been inserted inside the transaction, proving the whole replacement rolls back.

```python
import sqlite3

import pytest
import database


def test_replace_video_attendance_rolls_back_on_invalid_candidate(tmp_path, monkeypatch):
    _setup_database(tmp_path, monkeypatch)
    person_id = database.create_person("Alice")
    old = [{
        "person_id": person_id, "person_name": "Alice", "track_id": 1,
        "confidence": 0.9, "media_offset_ms": 1000,
    }]
    database.replace_video_attendance("C:/videos/a.mp4", old)

    candidates = [
        {"person_id": person_id, "person_name": "Alice", "track_id": 2,
         "confidence": 0.95, "media_offset_ms": 2000},
        {"person_id": 999999, "person_name": "Missing", "track_id": 3,
         "confidence": 0.9, "media_offset_ms": 3000},
    ]
    with pytest.raises(sqlite3.IntegrityError):
        database.replace_video_attendance("C:/videos/a.mp4", candidates)

    rows = database.fetch_attendance(source="video")
    assert [(row["track_id"], row["media_offset_ms"]) for row in rows] == [(1, 1000)]
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_database_attendance.py -q`

Expected: collection or assertions fail because the attendance table/APIs do not exist.

- [ ] **Step 3: Add the schema and database APIs**

Add `attendance_events` in `init_db()` using the existing foreign-key/SQLite conventions. Include source `CHECK`, non-null person/source/source_ref/track/confidence/created time, nullable source-specific time fields, and a unique constraint on `(source, source_ref, track_id, person_id)`. Use `get_conn()` so an exception rolls back the replacement delete and inserts. Reject malformed source/time combinations with `ValueError` before insert. Do not modify `recognition_logs` or `audio_logs` schemas.

- [ ] **Step 4: Run database attendance tests**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_database_attendance.py -q`

Expected: migration, idempotency, timestamp validation, filtering, replacement, and rollback tests pass.

- [ ] **Step 5: Commit**

```powershell
git add database.py tests/test_database_attendance.py
git commit -m "feat(attendance): add attendance event persistence"
```

---

### Task 2: Realtime attendance events

**Files:** Modify `realtime.py`, test `tests/test_realtime.py`.

**Consumes:** `database.record_attendance_event(...)`; `config.RECOGNITION_THRESHOLD`.

**Behavior:** Generate one UUID run ID per `realtime.run()` invocation. Keep a set of `(track_id, person_id)` pairs already recorded for attendance. At the first recognition sample for a pair where `person_id is not None` and confidence is at least `RECOGNITION_THRESHOLD`, record one event with source `realtime`, that run ID, track/person/name/confidence, and `_mono_to_wall_clock(timestamp)` converted to UTC ISO-8601. Keep existing per-second `log_recognition()` calls unchanged. A new track ID in the same run is a new appearance.

- [ ] **Step 1: Add failing realtime tests**

Extend the realtime run tests with a fake tracker/person track and monkeypatched `record_attendance_event`. Provide enough frames for repeated recognition samples. Assert exactly one attendance call for the same track, a different track produces a second call, and unknown/low-confidence tracks produce none. Assert source/ref/track/person/confidence fields and an ISO-8601 UTC time derived from the existing monotonic-to-wall-clock offset. Keep existing recognition/audio-log assertions.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_realtime.py -k attendance -q`

Expected: fails because realtime has no attendance recording yet.

- [ ] **Step 3: Add first-per-track attendance recording**

Create the run ID once after realtime starts. Create `attendance_logged_pairs` beside the existing `last_recognition_log` runtime state. At an eligible known recognition, check/add `(track.track_id, track.person_id)` and call `record_attendance_event` once per pair with `observed_at_utc=datetime.fromtimestamp(_mono_to_wall_clock(timestamp), timezone.utc).isoformat()`. Do not repurpose `log_recognition` or change existing log cadence.

- [ ] **Step 4: Run realtime tests**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_realtime.py -q`

Expected: attendance tests and existing realtime tests pass.

- [ ] **Step 5: Commit**

```powershell
git add realtime.py tests/test_realtime.py
git commit -m "feat(attendance): record realtime recognized appearances"
```

---

### Task 3: Uploaded-video attendance events

**Files:** Modify `video_processor.py`, `video_workflow.py`, tests `tests/test_video_processor.py` and `tests/test_video_workflow.py`.

**Consumes:** `database.replace_video_attendance(source_ref, events)` from Task 1.

**Behavior:** Add a top-level `attendance` candidate list to the existing video JSON log, leaving `recognition` and `speech` unchanged. Capture only the first frame per `(track_id, person_id)` pair at which its `person_id` is set and confidence meets `RECOGNITION_THRESHOLD`; candidate fields are `person_id`, `person_name`, `track_id`, `confidence`, and `media_offset_ms=round(timestamp_sec*1000)`. After subtitle/audio finalization succeeds and immediately before returning the meeting ID, `process_video_and_transcribe()` resolves the source path and calls `replace_video_attendance()` once. If face processing, transcription, cue writing, subtitle/audio mux, or fallback finalization raises, do not replace prior attendance for that video.

- [ ] **Step 1: Write failing video candidate tests**

Add a video-processor test with multiple frames on one known track, a second known track, an unknown track, and one low-confidence known track. Assert the new `attendance` array contains one first-seen candidate for each eligible known track with the expected media offset, while existing `recognition` and `speech` keys retain their current meanings.

Add workflow tests: successful workflow calls replacement once with the resolved video path and parsed candidates; transcription/final-mux failure does not call replacement and leaves prior data untouched. Keep the existing partial-output and one-transcription assertions.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_video_processor.py tests/test_video_workflow.py -k attendance -q`

Expected: candidate log and post-success database replacement assertions fail.

- [ ] **Step 3: Emit first-seen candidates in the video log**

Maintain an `attendance_candidates` list and `attendance_pairs` set in `process_video_pipeline()`. At each detected track, form `(track.track_id, track.person_id)` and append only if that pair is newly eligible under the global constraints. Add the list as the top-level `attendance` key; do not change `recognition` row fields or active-speaker events.

- [ ] **Step 4: Replace video attendance only after full workflow success**

After final mux succeeds, read the candidate list from the JSON log, normalize the input video with `Path.resolve()`, and call `database.replace_video_attendance(...)`. Do not call it in any exception/fallback branch. An empty successful candidate list intentionally clears prior attendance for that video.

- [ ] **Step 5: Run video attendance tests**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_video_processor.py tests/test_video_workflow.py -q`

Expected: event capture, success-only replacement, retry/reprocess, and existing workflow tests pass.

- [ ] **Step 6: Commit**

```powershell
git add video_processor.py video_workflow.py tests/test_video_processor.py tests/test_video_workflow.py
git commit -m "feat(attendance): record uploaded video appearances"
```

---

### Task 4: Attendance view and CSV export

**Files:** Modify `app.py`; create `attendance_view.py`; test `tests/test_attendance_view.py`.

**Consumes:** `database.fetch_attendance(source=None, person_id=None)`.

**Produces:** `build_attendance_frame(rows) -> pandas.DataFrame` and `attendance_frame_to_csv(frame) -> bytes` in `attendance_view.py`.

**Behavior:** Add an `Attendance` tab to `main()`. The view shows source and person filters, then a dense table with Person, Source, Observed (UTC), Video, Clip Time, Confidence, and Track ID. Realtime entries populate Observed (UTC); video entries populate Video and clip-relative `mm:ss` without a calendar date. The download button exports the currently filtered rows as UTF-8 CSV. Empty results show an empty state; do not show unknown or low-confidence observations.

- [ ] **Step 1: Add attendance query/view tests**

Add `tests/test_attendance_view.py`:

```python
from attendance_view import attendance_frame_to_csv, build_attendance_frame, format_clip_offset


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
        "Person", "Source", "Observed (UTC)", "Video", "Clip Time",
        "Confidence", "Track ID",
    ]
    assert frame.iloc[0]["Observed (UTC)"] == "2026-10-05T12:30:00+00:00"
    assert frame.iloc[0]["Clip Time"] == ""
    assert frame.iloc[1]["Observed (UTC)"] == ""
    assert frame.iloc[1]["Video"] == "a.mp4"
    assert frame.iloc[1]["Clip Time"] == "01:00"


def test_clip_offset_formatting_handles_minute_and_hour_boundaries():
    assert format_clip_offset(0) == "00:00"
    assert format_clip_offset(59999) == "00:59"
    assert format_clip_offset(60000) == "01:00"
    assert format_clip_offset(3600000) == "1:00:00"


def test_attendance_csv_exports_exact_frame_columns():
    frame = build_attendance_frame([{
        "person_name": "Alice", "source": "video", "observed_at_utc": None,
        "media_offset_ms": 0, "confidence": 0.9, "track_id": 1,
        "source_ref": "C:/videos/a.mp4",
    }])

    csv_text = attendance_frame_to_csv(frame).decode("utf-8-sig")
    assert csv_text.splitlines()[0] == (
        "Person,Source,Observed (UTC),Video,Clip Time,Confidence,Track ID"
    )
    assert "Alice,video,,a.mp4,00:00,0.9,1" in csv_text
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_attendance_view.py -q`

Expected: module collection fails because `attendance_view` and its formatters do not exist.

- [ ] **Step 3: Add the tab and view**

Add `format_clip_offset`, `build_attendance_frame`, and `attendance_frame_to_csv` in `attendance_view.py`. Format offsets as `MM:SS`, switching to `H:MM:SS` at one hour; preserve UTC values in realtime rows and keep video date fields empty. In `app.py`, add `render_attendance()` and register an `Attendance` tab. Provide source and person filters, fetch the matching events from the database, render the returned frame, and wire a CSV download button to `attendance_frame_to_csv(frame)`. Include an empty state.

- [ ] **Step 4: Run attendance/database UI-preparation tests**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/test_attendance_view.py -q`

Expected: the frame, time formatting, and CSV tests pass.

- [ ] **Step 5: Commit**

```powershell
git add app.py attendance_view.py tests/test_attendance_view.py
git commit -m "feat(attendance): add attendance list and export"
```

---

### Task 5: Full regression and bounded smoke test

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
