# Refinement Split-on-Failure and Application Logging Design

## Goal

1. Stop losing LLM refinement on very long transcript chunks.
2. Record what happened, when, where, and for how long, in log files, with
   errors in a separate file.

## Current state

### Chunks rejected by validation

Meeting 7 (244 segments, 71 chunks) logged
`Local OpenAI-compatible server output failed validation for 4 of 71 chunks`.
A read-only replay showed that chunks 14, 28, 67 and 68 failed with
"Refinement omitted, duplicated, or reordered source segment references". All
four were among the largest chunks (5,587 to 6,745 characters; the median chunk
is 927). Probing chunk 14 directly returned `finish_reason=stop`, 870 completion
tokens (cap 8192) and only 7 of 13 segments, identically on a repeat request.
The model stops early on long inputs; it is not cut off, and retrying the same
request cannot help. The validator is right to reject such output, and the
chunk then keeps its source text.

### Logging

No logging is configured for the Streamlit app. Only `face_core.py` calls
`logging.basicConfig` as an import side effect. Every module already uses
`logging.getLogger(__name__)`. `LOG_DIR` (`data/artifacts/logs`) exists and is
created by `initialize_runtime_paths()`. The failure warning does not say which
chunk failed, why, or how long it took.

## Part 1: split on failure

In `transcript_refiner._refine_with_provider`, local providers only (the cloud
provider path is unchanged):

- When a chunk's response fails `_normalize_and_validate_response` and the chunk
  has more than one segment, split the target segments at the midpoint and
  request each half, recursively.
- A single segment that is still rejected falls back to its source text, as
  today.
- Each half is a same-speaker run, so every existing check (speaker change,
  overlap, long pause, word-for-word equality, segment coverage and order)
  applies unchanged.
- Each half receives the same read-only context as a normal chunk: two segments
  before and two after, taken from the full transcript.
- Request errors (timeout, HTTP error, connection error) are not retried by
  splitting and still abort the whole refinement.
- Only failing chunks add requests.

## Part 2: logging setup

New module `src/face_voice_rec/logging_setup.py` with `setup_logging()`.
`app/main.py` calls it in `main()` immediately after `initialize_runtime_paths()`.

- Idempotent: Streamlit reruns the script on every interaction, so the function
  does not attach duplicate handlers to the root logger.
- Files in `LOG_DIR`, not colliding with the existing `realtime-*.log` files:
  - `app.log`: INFO and above from every module.
  - `errors.log`: ERROR and above only.
  - Both rotate at 5 MB with 5 backups.
- Console output is unchanged.
- Line format, with time in local time:
  `2026-10-23 10:44:09,256 WARNING transcript_refiner._refine_with_provider:530 | message`
  (time, level, module, function, line, then the message with `key=value`
  fields for what and duration).
- The `basicConfig` call in `face_core.py` becomes a no-op because the root
  logger is already configured; it is not modified.
- Known limit: two Streamlit instances running at once can make a Windows file
  rotation fail. Logging reports this on the console and does not crash the app.

## Part 3: refinement log events

`refine_transcript(segments, meeting_id=None)` gains an optional `meeting_id`
that is passed to `_refine_with_provider` for log fields. Existing callers and
tests are unaffected.

| Event | Level | Fields |
|---|---|---|
| Run started | INFO | `meeting`, `provider`, `model`, `endpoint` (host:port only), `timeout`, `segments`, `chunks` |
| Request ok | INFO | `chunk=N/M`, `depth`, `segments`, `chars`, `duration`, `outcome=ok` |
| Request rejected (will split) | WARNING | same fields plus `outcome=rejected` and `reason` (the validation message) |
| Single segment kept as source | WARNING | `chunk`, `segment` |
| Request failed (timeout, HTTP, connection) | ERROR | `chunk`, `duration`, `reason` |
| Run finished | INFO, or WARNING if any source text was kept | `meeting`, `status`, `requests`, `rejected`, `repaired_by_split`, `kept_source_segments`, `duration` |
| Run aborted (existing log line in `transcription_core.py`) | raised from WARNING to ERROR | `meeting`, `reason`, `duration` |

The `endpoint` field is logged so that a stale launch environment (the app
running with a different URL or timeout than `.env`) is visible at a glance.

Never logged: transcript text, the API key, or full URLs.

Because the error file is ERROR and above, per-chunk rejections that are
repaired by splitting appear only in `app.log`. A failed request and an aborted
run appear in both files.

## Out of scope

- A `LOG_LEVEL` setting.
- Detailed what/when/duration events for voice, face, database and the
  Streamlit pages. The central setup already routes their existing log calls to
  `app.log` and their errors to `errors.log`; detailed events are a later step.
- Changing chunk size limits.

## Testing (written first)

- `logging_setup`: calling it twice does not duplicate handlers; `app.log`
  receives INFO and above; `errors.log` receives ERROR only; the line contains
  time, level and where.
- Split on failure: a fake server that drops segments for any request over six
  segments yields all segments through the halves; a single segment that keeps
  failing falls back to its source text; request errors are not retried by
  splitting and still raise.
- Log events: each event above is emitted with its fields; a timeout produces
  the ERROR line; a sentinel string from the transcript never appears in log
  output.
- Existing refinement, transcription and database tests keep passing.
