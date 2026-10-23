# Refinement Split-on-Failure and Application Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop losing LLM refinement on long transcript chunks by resending rejected chunks as halves, and add app-wide file logging (when, what, duration, where) with ERROR-and-above lines in a separate file.

**Architecture:** `transcript_refiner._refine_with_provider` gains a recursive per-chunk helper that splits a rejected chunk in halves. A new `logging_setup.setup_logging()` attaches rotating `app.log`, `errors.log` and console handlers to the root logger once at app startup. The refiner emits `key=value` log events; `transcription_core` raises its existing failure line to ERROR with a duration.

**Tech Stack:** Python 3, stdlib `logging` / `logging.handlers.RotatingFileHandler`, pytest (`caplog`, `monkeypatch`), Streamlit app entry `app/main.py`.

**Spec:** `docs/superpowers/specs/2026-10-23-refinement-split-and-logging-design.md`

## Global Constraints

- Run Python and tests with the project venv from `D:\Git\FaceAndVoiceRecV2` in PowerShell: `.\directmlvenv\Scripts\python.exe -m pytest ...`.
- Split-on-failure applies to local providers only (`ollama`, `openai`); the cloud (`foundry`) path is unchanged and still raises on a rejected response.
- Request errors (timeout, HTTP error, connection error, invalid structured response from `post`) are never retried by splitting and still abort the whole refinement.
- Every existing validation check (speaker change, overlap, long pause, word-for-word equality, segment coverage and order) applies unchanged to each half.
- Each half gets the same read-only context as a normal chunk: two segments before and two after, taken from the full transcript.
- `app.log` receives INFO and above from every module; `errors.log` receives ERROR and above only; both rotate at 5 MB with 5 backups and live in `config.LOG_DIR` (`data/artifacts/logs`).
- Log line format: `%(asctime)s %(levelname)s %(module)s.%(funcName)s:%(lineno)d | %(message)s`, local time with milliseconds.
- A console `StreamHandler` (INFO and above, same format) is kept so terminal output continues.
- Never log transcript text, the API key, or full URLs. The `endpoint` field is host and port only (no user info, no path).
- `refine_transcript(segments, meeting_id=None)`: `meeting_id` is optional so existing callers and tests keep working.
- Do not modify the `logging.basicConfig` call in `face_core.py`. No `LOG_LEVEL` setting. Chunk size limits (`TRANSCRIPT_REFINEMENT_CHUNK_SEGMENTS`, `TRANSCRIPT_REFINEMENT_CHUNK_CHARS`) are unchanged.
- Comments: only where the code cannot show the reason, one short line.

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/face_voice_rec/logging_setup.py` | Create | `setup_logging()`: idempotent root-logger handlers |
| `tests/unit/test_logging_setup.py` | Create | Tests for `setup_logging()` |
| `app/main.py` | Modify | Call `setup_logging()` after `initialize_runtime_paths()` |
| `tests/unit/test_app_navigation.py` | Modify | Startup order test includes `setup_logging` |
| `src/face_voice_rec/domain/transcription/transcript_refiner.py` | Modify | Split-on-failure, log events, `meeting_id` |
| `tests/unit/test_transcript_refiner.py` | Modify | Split and log-event tests |
| `src/face_voice_rec/domain/transcription/transcription_core.py` | Modify | Pass `meeting_id`; ERROR line with duration |
| `tests/unit/test_transcription_core.py` | Modify | ERROR-line test; test doubles accept `meeting_id` |

---

### Task 1: Central logging setup and app wiring

**Files:**
- Create: `src/face_voice_rec/logging_setup.py`
- Create: `tests/unit/test_logging_setup.py`
- Modify: `app/main.py` (imports near line 21-23; `main()` near line 156-158)
- Modify: `tests/unit/test_app_navigation.py` (`test_main_renders_tabs_in_expected_order_with_startup_sequence`, near lines 111 and 145)

**Interfaces:**
- Consumes: `face_voice_rec.config.LOG_DIR` (a `Path`, read at call time so tests can monkeypatch it).
- Produces: `face_voice_rec.logging_setup.setup_logging() -> None`, idempotent; attaches exactly three handlers to the root logger on first call (`app.log` INFO, `errors.log` ERROR, console INFO) and sets the root level to INFO if it was higher.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_logging_setup.py`:

```python
import logging
import re

import pytest

from face_voice_rec import config, logging_setup


@pytest.fixture
def root_logger(monkeypatch, tmp_path):
    root = logging.getLogger()
    handlers_before = list(root.handlers)
    level_before = root.level
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs")
    yield root
    for handler in list(root.handlers):
        if handler not in handlers_before:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level_before)


def test_setup_logging_is_idempotent(root_logger):
    before = len(root_logger.handlers)

    logging_setup.setup_logging()
    after_first = len(root_logger.handlers)
    logging_setup.setup_logging()

    assert after_first == before + 3
    assert len(root_logger.handlers) == after_first


def test_app_log_gets_info_and_errors_log_gets_only_errors(root_logger, tmp_path):
    logging_setup.setup_logging()
    log = logging.getLogger("face_voice_rec.logging_demo")

    log.info("info line")
    log.warning("warning line")
    log.error("error line")

    app_text = (tmp_path / "logs" / "app.log").read_text(encoding="utf-8")
    errors_text = (tmp_path / "logs" / "errors.log").read_text(encoding="utf-8")
    assert "info line" in app_text
    assert "warning line" in app_text
    assert "error line" in app_text
    assert "error line" in errors_text
    assert "info line" not in errors_text
    assert "warning line" not in errors_text


def test_log_line_shows_when_level_and_where(root_logger, tmp_path):
    logging_setup.setup_logging()

    logging.getLogger("face_voice_rec.logging_demo").info("hello")

    last_line = (tmp_path / "logs" / "app.log").read_text(encoding="utf-8").splitlines()[-1]
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} INFO "
        r"test_logging_setup\.test_log_line_shows_when_level_and_where:\d+ \| hello",
        last_line,
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/unit/test_logging_setup.py -q`
Expected: FAIL (collection error) with `ImportError: cannot import name 'logging_setup' from 'face_voice_rec'`.

- [ ] **Step 3: Write the minimal implementation**

Create `src/face_voice_rec/logging_setup.py`:

```python
"""Application logging: a general log, an errors-only log, and the console."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from face_voice_rec import config

_FORMAT = "%(asctime)s %(levelname)s %(module)s.%(funcName)s:%(lineno)d | %(message)s"
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 5
_HANDLER_FLAG = "_face_voice_rec_handler"


def _file_handler(path: Path, level: int) -> logging.Handler:
    handler = RotatingFileHandler(
        path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
        delay=True,
    )
    handler.setLevel(level)
    return handler


def _console_handler() -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    return handler


def setup_logging() -> None:
    root = logging.getLogger()
    # Streamlit reruns the script on every interaction.
    if any(getattr(handler, _HANDLER_FLAG, False) for handler in root.handlers):
        return

    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_FORMAT)
    for handler in (
        _file_handler(log_dir / "app.log", logging.INFO),
        _file_handler(log_dir / "errors.log", logging.ERROR),
        _console_handler(),
    ):
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_FLAG, True)
        root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/unit/test_logging_setup.py -q`
Expected: `3 passed`.

- [ ] **Step 5: Write the failing startup-order test change**

In `tests/unit/test_app_navigation.py`, inside `test_main_renders_tabs_in_expected_order_with_startup_sequence`, add this line directly after the existing `monkeypatch.setattr(main, "initialize_runtime_paths", ...)` line:

```python
    monkeypatch.setattr(main, "setup_logging", lambda: events.append(("setup_logging",)))
```

and in the expected `events` list, insert `("setup_logging",),` between `("initialize_runtime_paths",),` and `("initialize_application_database",),` so that part reads:

```python
        ("initialize_runtime_paths",),
        ("setup_logging",),
        ("initialize_application_database",),
```

- [ ] **Step 6: Run the test to verify it fails**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/unit/test_app_navigation.py::test_main_renders_tabs_in_expected_order_with_startup_sequence -q`
Expected: FAIL with `AttributeError: <module 'app.main' ...> has no attribute 'setup_logging'`.

- [ ] **Step 7: Wire `setup_logging()` into `app/main.py`**

In `app/main.py`, change the imports:

```python
from face_voice_rec import config
from face_voice_rec.config import initialize_runtime_paths
from face_voice_rec.logging_setup import setup_logging
from face_voice_rec.storage.database import init_db
```

and in `main()`, call it right after `initialize_runtime_paths()`:

```python
    initialize_runtime_paths()
    setup_logging()
    initialize_application_database()
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/unit/test_app_navigation.py tests/unit/test_logging_setup.py -q`
Expected: all pass (no failures).

- [ ] **Step 9: Commit**

```bash
git add src/face_voice_rec/logging_setup.py tests/unit/test_logging_setup.py app/main.py tests/unit/test_app_navigation.py
git commit -m "feat: add central app.log and errors.log logging setup"
```

---

### Task 2: Split a rejected refinement chunk in halves

**Files:**
- Modify: `src/face_voice_rec/domain/transcription/transcript_refiner.py` (replace `_refine_with_provider`, currently starting at `def _refine_with_provider(normalized: Sequence[dict], provider: str) -> RefinementResult:`)
- Test: `tests/unit/test_transcript_refiner.py` (append to the end of the file)

**Interfaces:**
- Consumes: existing `_chunk_ranges`, `_normalize_and_validate_response(response, targets, normalized)`, `_LOCAL_PROVIDER_NAMES`, `RefinedGroup`, `TranscriptRefinementError`.
- Produces: `_source_group(targets: Sequence[dict]) -> RefinedGroup` and `_refine_chunk(post, provider, normalized, start, end) -> tuple[list[RefinedGroup], list[SpeakerChangeRecommendation], bool]` where the bool is `True` when every segment in the returned groups fell back to source text. Task 3 replaces both functions' signatures.

- [ ] **Step 1: Write the failing tests**

Append to the end of `tests/unit/test_transcript_refiner.py`:

```python
def test_rejected_chunk_is_resent_as_halves(monkeypatch):
    items = _segments(
        ("Alice", 0, 500, "one two"),
        ("Alice", 600, 900, "three four"),
        ("Alice", 1000, 1400, "five six"),
        ("Alice", 1500, 1900, "seven eight"),
    )
    requests = []

    def chat(payload):
        targets = json.loads(payload)["target_segments"]
        requests.append([item["segment_id"] for item in targets])
        # Mimics the model dropping segments on a long chunk.
        return _reply(targets[:2] if len(targets) > 2 else targets)

    monkeypatch.setattr(transcript_refiner, "_post_ollama_chat", chat)

    result = transcript_refiner.refine_transcript(items)

    assert requests == [[1, 2, 3, 4], [1, 2], [3, 4]]
    assert [group.source_segment_ids for group in result.groups] == [(1,), (2,), (3,), (4,)]
    assert [group.text for group in result.groups] == [
        "one two", "three four", "five six", "seven eight",
    ]


def test_halves_that_all_fail_merge_back_into_one_source_group(monkeypatch):
    items = _segments(
        ("Alice", 0, 500, "one two"),
        ("Alice", 600, 900, "three four"),
    )
    requests = []

    def chat(payload):
        targets = json.loads(payload)["target_segments"]
        requests.append([item["segment_id"] for item in targets])
        return {"groups": [], "speaker_change_recommendations": []}

    monkeypatch.setattr(transcript_refiner, "_post_ollama_chat", chat)

    result = transcript_refiner.refine_transcript(items)

    assert requests == [[1, 2], [1], [2]]
    assert [group.source_segment_ids for group in result.groups] == [(1, 2)]
    assert result.groups[0].text == "one two three four"


def test_request_errors_are_not_retried_by_splitting(monkeypatch):
    items = _segments(
        ("Alice", 0, 500, "one two"),
        ("Alice", 600, 900, "three four"),
    )
    requests = []

    def chat(_payload):
        requests.append(1)
        raise TranscriptRefinementError("Local Ollama request timed out")

    monkeypatch.setattr(transcript_refiner, "_post_ollama_chat", chat)

    with pytest.raises(TranscriptRefinementError, match="timed out"):
        transcript_refiner.refine_transcript(items)

    assert requests == [1]
```

- [ ] **Step 2: Run the tests to verify the first two fail**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/unit/test_transcript_refiner.py -k "resent_as_halves or merge_back_into_one_source_group or not_retried_by_splitting" -q`
Expected: `test_rejected_chunk_is_resent_as_halves` and `test_halves_that_all_fail_merge_back_into_one_source_group` FAIL (`requests == [[1, 2, 3, 4]]` and `[[1, 2]]` respectively, because the current code makes one request and falls back). `test_request_errors_are_not_retried_by_splitting` passes already; it guards existing behaviour.

- [ ] **Step 3: Implement split-on-failure**

In `src/face_voice_rec/domain/transcription/transcript_refiner.py`, replace the whole `_refine_with_provider` function (from `def _refine_with_provider(` through its final `return RefinementResult(...)`) with:

```python
def _source_group(targets: Sequence[dict]) -> RefinedGroup:
    # A chunk is one same-speaker run, so its source text is a valid group.
    return RefinedGroup(
        tuple(int(item["segment_id"]) for item in targets),
        " ".join(item["text"] for item in targets),
    )


def _refine_chunk(post, provider: str, normalized: Sequence[dict], start: int, end: int):
    """Return (groups, recommendations, all_source) for normalized[start:end]."""
    targets = normalized[start:end]
    context = normalized[max(0, start - 2):start] + normalized[end:end + 2]
    response = post(json.dumps(
        {
            "target_segments": targets,
            "read_only_context_segments": context,
            "target_segment_ids": [item["segment_id"] for item in targets],
        },
        ensure_ascii=False,
    ))
    try:
        groups, recommendations = _normalize_and_validate_response(
            response,
            targets,
            normalized,
        )
    except TranscriptRefinementError:
        if provider not in _LOCAL_PROVIDER_NAMES:
            raise
        if len(targets) == 1:
            return [_source_group(targets)], [], True
        # Models drop segments on long chunks; shorter halves usually pass.
        middle = start + len(targets) // 2
        left_groups, left_recommendations, left_source = _refine_chunk(
            post, provider, normalized, start, middle
        )
        right_groups, right_recommendations, right_source = _refine_chunk(
            post, provider, normalized, middle, end
        )
        if left_source and right_source:
            return [_source_group(targets)], [], True
        return (
            left_groups + right_groups,
            left_recommendations + right_recommendations,
            False,
        )
    return groups, recommendations, False


def _refine_with_provider(normalized: Sequence[dict], provider: str) -> RefinementResult:
    if provider == "foundry":
        post, model = _post_foundry_chat, config.MODEL
    elif provider == "ollama":
        post, model = _post_ollama_chat, config.TRANSCRIPT_REFINEMENT_MODEL
    elif provider == "openai":
        post, model = _post_openai_compatible_chat, config.TRANSCRIPT_REFINEMENT_MODEL
    else:
        raise ValueError(f"Unknown refinement provider: {provider}")

    output_groups = []
    output_recommendations = []
    deterministic_chunks = 0
    ranges = _chunk_ranges(normalized)
    for chunk_index, (start, end) in enumerate(ranges):
        groups, recommendations, all_source = _refine_chunk(
            post, provider, normalized, start, end
        )
        if all_source:
            deterministic_chunks += 1
        output_groups.extend(groups)
        output_recommendations.extend(recommendations)
        logger.debug(
            "Refined transcript chunk %d/%d (%d segments) via %s",
            chunk_index + 1,
            len(ranges),
            end - start,
            provider,
        )
    if deterministic_chunks:
        logger.warning(
            "Local %s output failed validation for %d of %d chunks; "
            "kept source text for those chunks",
            _LOCAL_PROVIDER_NAMES[provider],
            deterministic_chunks,
            len(ranges),
        )
    return RefinementResult(
        tuple(output_groups),
        tuple(output_recommendations),
        provider=provider,
        model=model,
    )
```

- [ ] **Step 4: Run the whole refiner test module**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/unit/test_transcript_refiner.py -q`
Expected: all pass, including the three new tests and the existing `test_local_chunk_validation_failure_uses_deterministic_text_for_that_chunk_only` (its two-segment chunk is split, both halves fail, and the halves merge back into the single `(1, 2)` group it expects).

- [ ] **Step 5: Commit**

```bash
git add src/face_voice_rec/domain/transcription/transcript_refiner.py tests/unit/test_transcript_refiner.py
git commit -m "feat: resend rejected refinement chunks as halves"
```

---

### Task 3: Refinement log events and meeting context

**Files:**
- Modify: `src/face_voice_rec/domain/transcription/transcript_refiner.py` (imports; `_refine_chunk` / `_refine_with_provider` from Task 2; `refine_transcript` signature and its three `_refine_with_provider` calls)
- Modify: `src/face_voice_rec/domain/transcription/transcription_core.py` (imports; `refresh_readable_transcript`)
- Modify: `tests/unit/test_transcription_core.py` (two `refine_transcript` test doubles; one new test)
- Modify: `tests/unit/test_transcript_refiner.py` (imports at top; append new tests)

**Interfaces:**
- Consumes: Task 2's `_source_group`, `_refine_chunk`, `_refine_with_provider`.
- Produces:
  - `refine_transcript(segments: Sequence[dict], meeting_id: int | None = None) -> RefinementResult`
  - `_refine_with_provider(normalized, provider, meeting_id=None)`
  - `_refine_chunk(post, normalized, start, end, stats, chunk_number, depth)` where `stats` is a `_RunStats`
  - `_endpoint_label(provider: str) -> str`
  - Log events (logger `face_voice_rec.domain.transcription.transcript_refiner`):
    - `refinement started meeting=%s provider=%s model=%s endpoint=%s timeout=%ss segments=%d chunks=%d` (INFO)
    - `refinement request meeting=%s chunk=%d/%d depth=%d segments=%d chars=%d duration=%.1fs outcome=ok` (INFO)
    - the same line with `outcome=rejected reason=<validation message>` (WARNING)
    - `refinement request failed meeting=%s chunk=%d/%d depth=%d segments=%d chars=%d duration=%.1fs reason=<message>` (ERROR)
    - `refinement kept source text meeting=%s chunk=%d/%d depth=%d segment=%d` (WARNING)
    - `refinement finished meeting=%s status=complete requests=%d rejected=%d repaired_by_split=%d kept_source_segments=%d duration=%.1fs` (INFO, or WARNING when `kept_source_segments` > 0)
  - In `transcription_core`: `Optional transcript refinement failed for meeting %s (%s) after %.1fs; keeping the deterministic readable transcript` at ERROR.

- [ ] **Step 1: Let the existing test doubles accept `meeting_id`**

`transcription_core` will call `refine_transcript(..., meeting_id=meeting_id)`. In `tests/unit/test_transcription_core.py`:

Change the lambda in `test_refresh_readable_transcript_persists_refinement_result_provenance`:

```python
        lambda _payload, **_kwargs: transcript_refiner.RefinementResult(
```

and in `test_refresh_readable_transcript_empty_meeting_stays_deterministic_without_network`:

```python
    def fail_if_called(_segments, **_kwargs):
```

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/unit/test_transcription_core.py -q`
Expected: all pass (these changes are compatible with the current code).

- [ ] **Step 2: Write the failing refiner log tests**

In `tests/unit/test_transcript_refiner.py`, change the top imports from:

```python
import json

import pytest
```

to:

```python
import json
import logging
import re

import pytest
```

Append to the end of the file:

```python
def _refiner_records(caplog):
    return [r for r in caplog.records if r.name == transcript_refiner.logger.name]


def test_refinement_logs_run_and_request_events(monkeypatch, caplog):
    monkeypatch.setattr(
        config, "TRANSCRIPT_REFINEMENT_URL", "http://sentinel-user:sentinel-pw@lan-host.invalid:30000/x"
    )
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_MODEL", "log-model")
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_TIMEOUT_SECONDS", 123)
    items = _segments(("Alice", 0, 500, "Hello there."))
    monkeypatch.setattr(
        transcript_refiner,
        "_post_ollama_chat",
        lambda payload: _reply(json.loads(payload)["target_segments"]),
    )
    caplog.set_level(logging.INFO)

    transcript_refiner.refine_transcript(items, meeting_id=7)

    started, request, finished = [r.getMessage() for r in _refiner_records(caplog)]
    assert started == (
        "refinement started meeting=7 provider=ollama model=log-model "
        "endpoint=lan-host.invalid:30000 timeout=123s segments=1 chunks=1"
    )
    assert re.fullmatch(
        r"refinement request meeting=7 chunk=1/1 depth=0 segments=1 chars=12 "
        r"duration=\d+\.\ds outcome=ok",
        request,
    )
    assert re.fullmatch(
        r"refinement finished meeting=7 status=complete requests=1 rejected=0 "
        r"repaired_by_split=0 kept_source_segments=0 duration=\d+\.\ds",
        finished,
    )
    assert "sentinel-user" not in caplog.text
    assert "sentinel-pw" not in caplog.text


def test_rejected_chunk_split_is_logged(monkeypatch, caplog):
    items = _segments(
        ("Alice", 0, 500, "one two"),
        ("Alice", 600, 900, "three four"),
        ("Alice", 1000, 1400, "five six"),
        ("Alice", 1500, 1900, "seven eight"),
    )
    monkeypatch.setattr(
        transcript_refiner,
        "_post_ollama_chat",
        lambda payload: _reply(
            json.loads(payload)["target_segments"][:2]
            if len(json.loads(payload)["target_segments"]) > 2
            else json.loads(payload)["target_segments"]
        ),
    )
    caplog.set_level(logging.INFO)

    transcript_refiner.refine_transcript(items, meeting_id=3)

    records = _refiner_records(caplog)
    rejected = [r for r in records if "outcome=rejected" in r.getMessage()]
    assert len(rejected) == 1
    assert rejected[0].levelno == logging.WARNING
    assert "chunk=1/1 depth=0 segments=4" in rejected[0].getMessage()
    assert "reason=Refinement omitted, duplicated, or reordered source segment references" in (
        rejected[0].getMessage()
    )
    ok_depths = [
        re.search(r"depth=(\d+)", r.getMessage()).group(1)
        for r in records
        if "outcome=ok" in r.getMessage()
    ]
    assert ok_depths == ["1", "1"]
    finished = records[-1]
    assert finished.levelno == logging.INFO
    assert re.fullmatch(
        r"refinement finished meeting=3 status=complete requests=3 rejected=1 "
        r"repaired_by_split=1 kept_source_segments=0 duration=\d+\.\ds",
        finished.getMessage(),
    )


def test_segments_kept_as_source_are_logged_and_counted(monkeypatch, caplog):
    items = _segments(
        ("Alice", 0, 500, "one two"),
        ("Alice", 600, 900, "three four"),
    )
    monkeypatch.setattr(
        transcript_refiner,
        "_post_ollama_chat",
        lambda _payload: {"groups": [], "speaker_change_recommendations": []},
    )
    caplog.set_level(logging.INFO)

    transcript_refiner.refine_transcript(items, meeting_id=5)

    records = _refiner_records(caplog)
    kept = [r for r in records if "kept source text" in r.getMessage()]
    assert len(kept) == 2
    assert all(r.levelno == logging.WARNING for r in kept)
    assert "chunk=1/1 depth=1 segment=1" in kept[0].getMessage()
    finished = records[-1]
    assert finished.levelno == logging.WARNING
    assert re.fullmatch(
        r"refinement finished meeting=5 status=complete requests=3 rejected=3 "
        r"repaired_by_split=0 kept_source_segments=2 duration=\d+\.\ds",
        finished.getMessage(),
    )


def test_request_failure_is_logged_as_error_with_chunk_and_duration(monkeypatch, caplog):
    items = _segments(("Alice", 0, 500, "Hello."))

    def chat(_payload):
        raise TranscriptRefinementError("Local Ollama request timed out")

    monkeypatch.setattr(transcript_refiner, "_post_ollama_chat", chat)

    with pytest.raises(TranscriptRefinementError):
        transcript_refiner.refine_transcript(items, meeting_id=9)

    errors = [r for r in _refiner_records(caplog) if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert re.fullmatch(
        r"refinement request failed meeting=9 chunk=1/1 depth=0 segments=1 chars=6 "
        r"duration=\d+\.\ds reason=Local Ollama request timed out",
        errors[0].getMessage(),
    )


def test_logs_never_contain_transcript_text(monkeypatch, caplog):
    items = _segments(
        ("Alice", 0, 500, "SECRETWORD one"),
        ("Alice", 600, 900, "SECRETWORD two"),
    )
    monkeypatch.setattr(
        transcript_refiner,
        "_post_ollama_chat",
        lambda _payload: {"groups": [], "speaker_change_recommendations": []},
    )
    caplog.set_level(logging.DEBUG)

    transcript_refiner.refine_transcript(items, meeting_id=1)

    assert "SECRETWORD" not in caplog.text
```

- [ ] **Step 3: Write the failing core test**

In `tests/unit/test_transcription_core.py`, change the top imports from:

```python
from pathlib import Path
import json
import sqlite3
```

to:

```python
from pathlib import Path
import json
import logging
import re
import sqlite3
```

Append to the end of the file:

```python
def test_refinement_failure_is_logged_as_error_with_duration(monkeypatch, caplog):
    from face_voice_rec.domain.transcription import transcript_refiner

    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_ENABLED", True)
    with database.get_conn() as connection:
        connection.execute(
            "INSERT INTO meetings(video_path, created_at, transcription_status) "
            "VALUES ('m.mp4', 'now', 'completed')"
        )
        meeting_id = connection.execute("SELECT meeting_id FROM meetings").fetchone()[0]
    save_transcripts(
        [TranscriptionSegment("Alice", 0, 1000, "Hello there.", 0.9)],
        config.TRANSCRIPTS_DIR / str(meeting_id),
        meeting_id=meeting_id,
    )

    def fail(_segments, **_kwargs):
        raise transcript_refiner.TranscriptRefinementError(
            "Local OpenAI-compatible server request timed out"
        )

    monkeypatch.setattr(transcript_refiner, "refine_transcript", fail)

    transcription_core.refresh_readable_transcript(meeting_id, use_llm=True)

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert re.fullmatch(
        rf"Optional transcript refinement failed for meeting {meeting_id} "
        r"\(Local OpenAI-compatible server request timed out\) after \d+\.\ds; "
        r"keeping the deterministic readable transcript",
        errors[0].getMessage(),
    )
```

- [ ] **Step 4: Run the new tests to verify they fail**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/unit/test_transcript_refiner.py tests/unit/test_transcription_core.py -k "logs_run_and_request or split_is_logged or kept_as_source or request_failure_is_logged or never_contain_transcript or logged_as_error_with_duration" -q`
Expected: the five refiner tests FAIL with `TypeError: refine_transcript() got an unexpected keyword argument 'meeting_id'`; the core test FAILS because the existing line is logged at WARNING with different text.

- [ ] **Step 5: Implement the refiner log events**

In `src/face_voice_rec/domain/transcription/transcript_refiner.py`:

Add to the imports (next to the other `from` imports at the top):

```python
from time import perf_counter
```

Replace Task 2's `_source_group`, `_refine_chunk` and `_refine_with_provider` (everything from `def _source_group(` through the end of `_refine_with_provider`) with:

```python
@dataclass
class _RunStats:
    meeting_id: int | None
    provider: str
    total_chunks: int
    requests: int = 0
    rejected: int = 0
    kept_source_segments: int = 0


def _endpoint_label(provider: str) -> str:
    url = config.ANTHROPIC_API_URL if provider == "foundry" else config.TRANSCRIPT_REFINEMENT_URL
    parts = urlsplit(url)
    try:
        port = parts.port
    except ValueError:
        port = None
    host = parts.hostname or "unknown"
    return f"{host}:{port}" if port else host


def _source_group(targets: Sequence[dict]) -> RefinedGroup:
    # A chunk is one same-speaker run, so its source text is a valid group.
    return RefinedGroup(
        tuple(int(item["segment_id"]) for item in targets),
        " ".join(item["text"] for item in targets),
    )


def _refine_chunk(
    post,
    normalized: Sequence[dict],
    start: int,
    end: int,
    stats: _RunStats,
    chunk_number: int,
    depth: int,
):
    """Return (groups, recommendations, all_source) for normalized[start:end]."""
    targets = normalized[start:end]
    context = normalized[max(0, start - 2):start] + normalized[end:end + 2]
    fields = (
        stats.meeting_id,
        chunk_number,
        stats.total_chunks,
        depth,
        len(targets),
        sum(len(item["text"]) for item in targets),
    )
    started = perf_counter()
    try:
        response = post(json.dumps(
            {
                "target_segments": targets,
                "read_only_context_segments": context,
                "target_segment_ids": [item["segment_id"] for item in targets],
            },
            ensure_ascii=False,
        ))
    except TranscriptRefinementError as error:
        logger.error(
            "refinement request failed meeting=%s chunk=%d/%d depth=%d segments=%d "
            "chars=%d duration=%.1fs reason=%s",
            *fields, perf_counter() - started, error,
        )
        raise
    duration = perf_counter() - started
    stats.requests += 1
    try:
        groups, recommendations = _normalize_and_validate_response(
            response,
            targets,
            normalized,
        )
    except TranscriptRefinementError as error:
        logger.warning(
            "refinement request meeting=%s chunk=%d/%d depth=%d segments=%d chars=%d "
            "duration=%.1fs outcome=rejected reason=%s",
            *fields, duration, error,
        )
        if stats.provider not in _LOCAL_PROVIDER_NAMES:
            raise
        stats.rejected += 1
        if len(targets) == 1:
            stats.kept_source_segments += 1
            logger.warning(
                "refinement kept source text meeting=%s chunk=%d/%d depth=%d segment=%d",
                stats.meeting_id, chunk_number, stats.total_chunks, depth,
                targets[0]["segment_id"],
            )
            return [_source_group(targets)], [], True
        # Models drop segments on long chunks; shorter halves usually pass.
        middle = start + len(targets) // 2
        left_groups, left_recommendations, left_source = _refine_chunk(
            post, normalized, start, middle, stats, chunk_number, depth + 1
        )
        right_groups, right_recommendations, right_source = _refine_chunk(
            post, normalized, middle, end, stats, chunk_number, depth + 1
        )
        if left_source and right_source:
            return [_source_group(targets)], [], True
        return (
            left_groups + right_groups,
            left_recommendations + right_recommendations,
            False,
        )
    logger.info(
        "refinement request meeting=%s chunk=%d/%d depth=%d segments=%d chars=%d "
        "duration=%.1fs outcome=ok",
        *fields, duration,
    )
    return groups, recommendations, False


def _refine_with_provider(
    normalized: Sequence[dict],
    provider: str,
    meeting_id: int | None = None,
) -> RefinementResult:
    if provider == "foundry":
        post, model = _post_foundry_chat, config.MODEL
    elif provider == "ollama":
        post, model = _post_ollama_chat, config.TRANSCRIPT_REFINEMENT_MODEL
    elif provider == "openai":
        post, model = _post_openai_compatible_chat, config.TRANSCRIPT_REFINEMENT_MODEL
    else:
        raise ValueError(f"Unknown refinement provider: {provider}")

    ranges = _chunk_ranges(normalized)
    stats = _RunStats(meeting_id, provider, len(ranges))
    run_started = perf_counter()
    logger.info(
        "refinement started meeting=%s provider=%s model=%s endpoint=%s timeout=%ss "
        "segments=%d chunks=%d",
        meeting_id, provider, model, _endpoint_label(provider),
        config.TRANSCRIPT_REFINEMENT_TIMEOUT_SECONDS, len(normalized), len(ranges),
    )

    output_groups = []
    output_recommendations = []
    repaired_by_split = 0
    for chunk_number, (start, end) in enumerate(ranges, start=1):
        rejected_before = stats.rejected
        kept_before = stats.kept_source_segments
        groups, recommendations, _all_source = _refine_chunk(
            post, normalized, start, end, stats, chunk_number, 0
        )
        if stats.rejected > rejected_before and stats.kept_source_segments == kept_before:
            repaired_by_split += 1
        output_groups.extend(groups)
        output_recommendations.extend(recommendations)

    logger.log(
        logging.WARNING if stats.kept_source_segments else logging.INFO,
        "refinement finished meeting=%s status=complete requests=%d rejected=%d "
        "repaired_by_split=%d kept_source_segments=%d duration=%.1fs",
        meeting_id, stats.requests, stats.rejected, repaired_by_split,
        stats.kept_source_segments, perf_counter() - run_started,
    )
    return RefinementResult(
        tuple(output_groups),
        tuple(output_recommendations),
        provider=provider,
        model=model,
    )
```

Change the `refine_transcript` signature and its three `_refine_with_provider` calls:

```python
def refine_transcript(
    segments: Sequence[dict],
    meeting_id: int | None = None,
) -> RefinementResult:
```

```python
    local_provider = _local_provider()
    if not config.TRANSCRIPT_REFINEMENT_USE_CLOUD:
        return _refine_with_provider(normalized, local_provider, meeting_id)

    try:
        return _refine_with_provider(normalized, "foundry", meeting_id)
    except TranscriptRefinementError as cloud_error:
        cloud_reason = str(cloud_error)
    logger.warning("Foundry refinement failed (%s); retrying locally", cloud_reason)
    try:
        local_result = _refine_with_provider(normalized, local_provider, meeting_id)
```

(Leave the rest of `refine_transcript` unchanged.)

- [ ] **Step 6: Implement the core changes**

In `src/face_voice_rec/domain/transcription/transcription_core.py`:

Add next to the other stdlib imports:

```python
from time import perf_counter
```

In `refresh_readable_transcript`, start a timer just before the `try:` inside `if should_use_llm and detailed:`:

```python
    if should_use_llm and detailed:
        transcript_refiner = None
        refinement_started = perf_counter()
        try:
```

pass the meeting id, closing the call with it:

```python
            result = transcript_refiner.refine_transcript([
                {
                    "segment_id": segment.source_segment_ids[0],
                    "speaker_label": segment.speaker_label,
                    "start_ms": segment.start_ms,
                    "end_ms": segment.end_ms,
                    "text": segment.text,
                }
                for segment in detailed
            ], meeting_id=meeting_id)
```

and replace the `logger.warning(...)` in the `except` block with:

```python
            logger.error(
                "Optional transcript refinement failed for meeting %s (%s) after %.1fs; "
                "keeping the deterministic readable transcript",
                meeting_id,
                category,
                perf_counter() - refinement_started,
            )
```

- [ ] **Step 7: Run the refiner and core test modules**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/unit/test_transcript_refiner.py tests/unit/test_transcription_core.py tests/unit/test_transcript_refinement_ui.py -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/face_voice_rec/domain/transcription/transcript_refiner.py src/face_voice_rec/domain/transcription/transcription_core.py tests/unit/test_transcript_refiner.py tests/unit/test_transcription_core.py
git commit -m "feat: log refinement requests, splits and failures with timings"
```

---

### Task 4: Full verification and live check

**Files:** none changed (verification only; commit only if a fix is needed).

- [ ] **Step 1: Run the full unit suite**

Run: `.\directmlvenv\Scripts\python.exe -m pytest tests/unit -q`
Expected: all pass. If a test outside the files this plan touches fails, report it to the user with its name and error rather than changing it.

- [ ] **Step 2: Check editor diagnostics on the touched source files**

Use the editor's problems view (or `get_errors`) for `logging_setup.py`, `transcript_refiner.py`, `transcription_core.py` and `app/main.py`.
Expected: no errors.

- [ ] **Step 3: Live check on meeting 7 (the meeting that had 4 rejected chunks)**

This regenerates meeting 7's readable view only (source segments are not changed) and writes to `data/artifacts/logs/`.

```powershell
Remove-Item Env:TRANSCRIPT_REFINEMENT_URL,Env:TRANSCRIPT_REFINEMENT_TIMEOUT_SECONDS,Env:TRANSCRIPT_REFINEMENT_API,Env:TRANSCRIPT_REFINEMENT_MODEL,Env:TRANSCRIPT_REFINEMENT_USE_CLOUD -ErrorAction SilentlyContinue
@'
from face_voice_rec.config import initialize_runtime_paths
from face_voice_rec.logging_setup import setup_logging
from face_voice_rec.domain.transcription import transcription_core
initialize_runtime_paths()
setup_logging()
transcription_core.refresh_readable_transcript(7, use_llm=True)
'@ | .\directmlvenv\Scripts\python.exe -
Get-Content data\artifacts\logs\app.log | Select-String "meeting=7"
Test-Path data\artifacts\logs\errors.log
```

Expected in `app.log`:
- one `refinement started meeting=7 ... endpoint=192.168.137.46:30000 timeout=900s segments=244 chunks=71` line;
- `outcome=rejected` WARNING lines for the large chunks, followed by `depth=1` request lines;
- one `refinement finished meeting=7 status=complete requests=<more than 71> rejected=<n> repaired_by_split=<n> kept_source_segments=<n>` line.

Success criterion: `kept_source_segments` is well below the 40 segments that were kept as source before (chunks 14, 28, 67 and 68). Report the actual numbers. `errors.log` should not exist or should be empty, because no request failed.

- [ ] **Step 4: Tell the user how to pick up the change in the running app**

Stop the Streamlit app with Ctrl+C and start it again from a PowerShell terminal where `TRANSCRIPT_REFINEMENT_URL` and `TRANSCRIPT_REFINEMENT_TIMEOUT_SECONDS` are not set (so `.env` is read), then open the Transcripts tab and click the refine button once. New log lines appear in `data/artifacts/logs/app.log`.
