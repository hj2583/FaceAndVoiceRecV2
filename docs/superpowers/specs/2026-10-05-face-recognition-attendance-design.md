# Face-Recognition Attendance

## Problem

The app recognizes known people in realtime webcam sessions and uploaded videos, but it does not provide an attendance list. Existing recognition logs are sampling/debug records: realtime emits repeated samples, while video processing writes per-frame records to a JSON file. Those records are unsuitable as attendance rows without event-level filtering and stable source identity.

## Goals

- Record one attendance event per known, confidently recognized tracked appearance in both realtime and uploaded-video workflows.
- Use the existing `config.RECOGNITION_THRESHOLD` (currently 0.70); do not record unknown or below-threshold faces.
- Preserve the distinction between a persistent `person_id` and a run-local `track_id`.
- Display realtime events with their UTC wall-clock timestamp.
- Display uploaded-video events with the video reference and clip-relative offset; do not invent a calendar date for historical footage.
- Reprocessing an uploaded video replaces its previous attendance events only after the new processing run succeeds.
- Keep existing recognition logs and audio logs available for their current diagnostic purposes.

## Non-Goals

- Marking people absent or comparing attendance against a roster.
- Class/session management, manual corrections, or manual attendance entry.
- Treating every frame or every one-second realtime recognition sample as a separate attendance event.
- Inferring an uploaded video's recording date from its filename or filesystem timestamp.

## Event Semantics

An appearance is identified by the tuple `(source, source_ref, track_id, person_id)`:

- `source` is `realtime` or `video`.
- `source_ref` is a generated realtime run ID, or a normalized absolute uploaded-video path.
- `track_id` is local to that run/video and separates distinct tracked appearances.
- `person_id` links to a known person in the database.

Only observations with `person_id` set and confidence at or above `RECOGNITION_THRESHOLD` are eligible. The first eligible observation for a track creates an attendance event; repeated observations of the same person/track/source do not create more events. If the tracker loses and reacquires a person as a new track, that is a new appearance.

Realtime timestamps are stored as UTC ISO-8601 values converted from the pipeline's monotonic clock. Uploaded-video events store the first eligible clip offset in milliseconds; their calendar timestamp remains null.

## Data Model

Add an `attendance_events` table through the existing idempotent `init_db()` migration pattern. Fields:

- `attendance_id`: integer primary key.
- `person_id`: required foreign key to `persons`.
- `person_name`: name snapshot for historical display.
- `source`: constrained to `realtime` or `video`.
- `source_ref`: required run ID or normalized video path.
- `track_id`: required source-local track identifier.
- `observed_at_utc`: nullable ISO-8601 timestamp; populated for realtime only.
- `media_offset_ms`: nullable nonnegative offset; populated for uploaded video only.
- `confidence`: required recognition confidence.
- `created_at_utc`: required insertion timestamp.

Enforce uniqueness on `(source, source_ref, track_id, person_id)` to make repeated observations idempotent within a source run. Store attendance separately from `recognition_logs` and `audio_logs`.

For realtime, generate a fresh run ID when a camera recognition process starts and record eligible tracks once during that run. For uploaded video, collect first eligible recognition per track while processing. Replace rows for that normalized video path transactionally only after the pipeline completes successfully; on failure keep the prior attendance rows intact.

## User Experience

Add an `Attendance` tab to the existing Streamlit navigation with a source mode selector: `Realtime` or `Uploaded video`. In `Uploaded video` mode, require selection of one source video by its normalized path. In `Realtime` mode, show exact start/end date and time controls in machine-local time, defaulting to the last 24 hours; include both endpoints and convert them to UTC for querying. Default an `Up to now` option on so the end boundary advances when the view reruns; allow it to be turned off to enter a fixed end date/time. Use Streamlit date/time widgets available at the project's minimum supported version. Keep the person filter in both modes.

Show a dense table with person, source, observation time or clip offset, video name, source reference/run ID, confidence, and track ID. Realtime rows use wall-clock time; video rows visibly use clip-relative `mm:ss` and the input video name. Provide CSV download of exactly the selected source, time interval, and person rows. Protect spreadsheet CSV exports by escaping formula-leading text cells. Do not present video clip offsets as dates. Attendance eligibility is determined at capture time using the configured threshold; do not hide historical rows when that threshold is later changed.

## Testing

- Database migration is idempotent and attendance rows enforce their source/run/track identity.
- Known, above-threshold realtime observations create one event per track; repeated log samples do not duplicate it; unknown and low-confidence observations are ignored.
- Realtime event timestamps are UTC wall-clock values, not monotonic process values.
- Video logs carry `person_id`, `track_id`, confidence, and first clip offset for known appearances; repeated frames on a track create one row.
- A successful reprocessing run atomically replaces events for that video path; a failed run preserves prior events.
- Attendance queries return current/snapshotted person names, source, time representation, confidence, and track ID with filters.
- Streamlit attendance view requires one video selection in video mode; realtime mode filters with inclusive machine-local datetimes converted to UTC and defaults to the last 24 hours. Both modes retain person filtering and export exactly the selected rows.
- Query tests cover source_ref filtering, inclusive realtime UTC bounds, and empty/reversed time ranges.
- Existing recognition, audio-log, face, and video workflows remain unchanged when no attendance-eligible recognition is produced.

## Limitations

Face recognition can produce false matches; attendance records reflect the configured recognition threshold and are not proof of identity. Uploaded-video offsets identify when the appearance occurs in the clip, not when the footage was recorded. Reprocessing the same normalized video path replaces its prior attendance events; moving or renaming the video creates a different source reference.
