# Lazy Loading and Database Startup Design

Date: 2026-10-13

## Goal

Reduce Streamlit startup/rerun work by loading only the selected tab's module and deferring the Torch/ONNX runtime probe until requested, while eliminating intermittent SQLite `database is locked` errors during app/realtime initialization. Preserve the current seven-tab UI, recognition behavior, and database schema.

## Current Evidence

- `app/main.py` imports all seven page modules at module load and invokes every page renderer under `st.tabs` on each rerun.
- Stateful tabs can defer inactive content using `on_change="rerun"` and each tab container's `.open` property. The selected interpreter has Streamlit 1.64.0 and exposes these parameters. `requirements.txt` supports Streamlit 1.40+, so an eager compatibility fallback is required for versions without the stateful API.
- `get_runtime_status()` imports Torch when called; `main()` currently calls it before tab rendering on every rerun.
- `init_db()` performs DDL and compatibility backfills, including an `UPDATE transcription_segments SET original_text=text WHERE original_text IS NULL`. It currently has no process lock, busy timeout, or explicit write transaction. `_DB_LOCK` protects `get_conn()` only.
- A temporary-database reproduction with four synchronized threads over 50 trials produced 40 exceptions across 200 `init_db()` calls: 23 `database is locked`, 14 duplicate `track_id`, and 3 duplicate `word_timestamps` errors.
- `init_db()` is called during Streamlit startup and realtime worker startup. Repeated Streamlit reruns also call it again.

## Approaches

1. **Stateful tabs with compatibility fallback (recommended):** use `st.tabs(..., on_change="rerun", key=...)` when supported, import/render only the tab whose `.open` is true, and use the existing eager behavior on older Streamlit. This preserves tabs and the current version floor while making supported versions genuinely lazy.
2. **Lazy models only:** keep importing/rendering all pages and defer only model creation. This is smaller, but leaves page imports, pandas work, database reads, and status probing on startup.
3. **Sidebar page navigation:** Streamlit naturally executes only one page, but changes the existing seven-tab interaction and is not selected.

## Lazy-Loading Design

`app.main` will import no page modules at module scope. It builds the same seven labels in the same order. Before constructing navigation it sets Streamlit's scriptable `client.showSidebarNavigation` option to `False`, preventing the conventional `app/pages/` directory from creating duplicate sidebar links. When the installed `st.tabs` supports `on_change="rerun"`, it enables reruns on selection and imports/calls only the active page renderer. If the API is unavailable, the app imports and renders all pages as it does today; the project will not raise its Streamlit minimum solely for this optimization.

The runtime status panel becomes an explicit **Check runtime** action. Before the action, the UI shows no Torch/ONNX probe result and imports neither provider. On click, it calls the existing `get_runtime_status()` once and stores the result in `st.session_state` for subsequent reruns. The FaceIndex stays behind its existing cached getter and is reached only when the Face Database tab is selected.

The tab event test will assert exact label/order preservation, lazy import/render of only the open tab on the stateful branch, eager fallback behavior for an older API, and that runtime status is not called until the action is activated. UI text and recognition/model thresholds remain unchanged.

## SQLite Initialization Design

`init_db()` will hold the existing `_DB_LOCK` across initialization, serializing it with `get_conn()` in the same process. Connections use a bounded 30-second SQLite timeout and matching `PRAGMA busy_timeout`. Set WAL mode before beginning the schema transaction, then use `BEGIN IMMEDIATE` before schema inspection, DDL, and compatibility backfills. Commit once on success; roll back and re-raise on failure. This lets SQLite serialize competing app/realtime processes and prevents partial schema updates.

Within the Streamlit process, a cached application initialization wrapper calls `init_db()` only once per server lifetime, instead of on every tab-selection rerun. Realtime remains a separate process and calls `init_db()` once at worker startup. A timeout still returns a clear initialization error; the app must not swallow the error or continue with a partial schema.

No database schema or recognition policy changes are part of this work.

## Tests and Validation

- Add a concurrency test using a temporary database and multiple threads that previously reproduced lock/duplicate-column failures; assert every initializer completes and the compatibility backfills are idempotent.
- Add a separate-process initialization test against a temporary database to verify SQLite's busy timeout/write lock behavior beyond the in-process `RLock`.
- Test that app reruns initialize the database once per Streamlit process while realtime startup still initializes independently.
- Test selected-tab lazy import/render, older-Streamlit eager fallback, and runtime-status unloaded/loaded states using fakes; no test runs app initialization against the live database.
- Run focused database/UI tests and the full suite. All database concurrency and migration tests use temporary paths. Do not run Streamlit or migration against the live project data during implementation validation.

## Out of Scope

- Changing the seven tab labels/order or switching to sidebar navigation.
- Changing recognition thresholds, model selection, transcription, or database schema.
- Removing the eager fallback or raising the supported Streamlit minimum.
- Running path-reference migration against the live production database during development.