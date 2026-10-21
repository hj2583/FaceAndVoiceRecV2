# Cloud Transcript Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select Foundry or Ollama transcript formatting, fall back from cloud to a complete local rerun, and display the saved output's actual provider/model.

**Architecture:** Keep environment configuration in settings and provider HTTP/parsing in the existing refiner. Normalize once, split targets at mandatory boundaries, and validate every provider's output identically. Extend refinement results with provenance, then persist it using existing run fields and render provider-neutral controls.

**Tech Stack:** Python 3.10+, urllib, python-dotenv, SQLite, Streamlit, pytest, Anthropic Messages REST, Ollama chat REST.

## Global Constraints

- Approved spec: `docs/superpowers/specs/2026-10-16-cloud-transcript-refinement-design.md`.
- `TRANSCRIPT_REFINEMENT_ENABLED` remains the master switch; default false.
- `TRANSCRIPT_REFINEMENT_USE_CLOUD` defaults false. True selects Foundry first with automatic Ollama fallback; false uses Ollama only.
- Existing process environment variables take precedence over repository-root `.env`.
- Never stage, commit, display, or change `.env` credentials. Preserve pre-existing `.gitignore` and `app/main.py` changes.
- No real cloud requests, live app startup, live database access, or runtime migration during implementation/testing.
- Discard all cloud chunks and recommendations before a local fallback; never combine providers in one result.
- Preserve exact spoken-word order, detailed source rows, speaker labels, timestamps, and existing validator rejection guarantees.
- Keep existing timeout (90 seconds), chunk limits (30 segments/9000 characters), and pause limit (1500 ms).
- Do not change schema, recognition, enrollment, attendance, tabs, or historical attribution.
- Tests use mocked HTTP and temporary fixtures; no real model names or credential values are needed.
- Do not automatically commit implementation work unless separately authorized. Commit commands below apply only after authorization; otherwise leave each tested task as a scoped working-tree change.

## File Map

Modify `src/face_voice_rec/config/settings.py` and `config/__init__.py` for dotenv and exported settings; declare python-dotenv in both requirements files.

Modify `src/face_voice_rec/domain/transcription/transcript_refiner.py` for shared instructions, boundary-safe chunking, Foundry transport, whole-transcript fallback, and result provenance. Keep the existing `_post_ollama_chat(str) -> dict` seam for compatibility.

Modify `src/face_voice_rec/domain/transcription/transcription_core.py` only around readable-result attribution and sanitized failure reporting.

Modify `app/pages/transcripts.py` for configured-provider action wording and historical provider attribution. Add `tests/unit/test_transcript_refinement_ui.py` because there is no existing test module owning this page's refinement status presentation. Reuse the existing config, refiner, and transcription tests for all other coverage.

Update README refinement setup and troubleshooting; do not create a credential-bearing sample file.

## Official Protocol Evidence

Confirmed during planning from Microsoft's REST API example:
https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-models/how-to/use-foundry-models-claude?view=foundry-classic

Use the full configured `/anthropic/v1/messages` endpoint unchanged, `x-api-key`,
`anthropic-version`, and `Content-Type: application/json`. Send a deployment name
unchanged in `model`, top-level `system`, user `messages`, `max_tokens`, and
`stream: false`. Do not send Ollama's `format` or `options` fields to Foundry.
Use `max_tokens=8192` as a finite output cap; reject truncated responses rather
than accepting incomplete JSON. Do not add adaptive thinking or optional model
features. No assumption is made about the public validity of a deployment name.

---

### Task 1: Root Dotenv And Provider Configuration

**Files:**
- Modify: `src/face_voice_rec/config/settings.py`
- Modify: `src/face_voice_rec/config/__init__.py`
- Modify: `requirements.txt`, `requirements-gpu.txt`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `_load_environment(project_root: Path | None = None) -> None` in settings, called before environment reads.
- Produces config exports: `TRANSCRIPT_REFINEMENT_USE_CLOUD: bool`, `ANTHROPIC_API_URL: str`, `API_KEY: str`, `ANTHROPIC_VERSION: str`, `MODEL: str`.
- Existing local URL/model/enabled settings retain their names and defaults. Cloud strings default empty; missing credentials are checked only on cloud use.

- [ ] **Step 1: Add failing tests for dotenv location and precedence.** Import settings and call the helper with a temporary root; do not reload configuration from the user's real `.env`. Add a loader spy asserting the import-time call precedes setting reads. Use subprocesses with a temporary copied settings module under `project/src/face_voice_rec/config/settings.py` to test root versus app working directories without touching real data.

```python
def test_dotenv_keeps_process_values(tmp_path, monkeypatch):
    import face_voice_rec.config.settings as settings
    (tmp_path / ".env").write_text(
        "TRANSCRIPT_REFINEMENT_USE_CLOUD=true\nMODEL=file-deployment\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TRANSCRIPT_REFINEMENT_USE_CLOUD", raising=False)
    monkeypatch.setenv("MODEL", "process-deployment")
    settings._load_environment(tmp_path)
    assert os.environ["MODEL"] == "process-deployment"
    assert os.environ["TRANSCRIPT_REFINEMENT_USE_CLOUD"] == "true"
```

Add `import os` to the test file. Parameterize true/false values using the existing accepted boolean tokens. Verify absent `.env` and empty cloud settings are inert. Restore any changed environment/module state using fixtures.

- [ ] **Step 2: Run red check.**

Run: `directmlvenv/Scripts/python.exe -m pytest tests/unit/test_config.py -q`
Expected: the new helper/export checks fail; existing tests remain unchanged.

- [ ] **Step 3: Implement dotenv and exports.** Install `python-dotenv>=1.0,<2` in the selected environment and list it in both requirements files. Use an explicit path rather than dotenv's upward search. Mirror new names in both `__all__` lists.

```python
from pathlib import Path
from dotenv import load_dotenv

def _load_environment(project_root: Path | None = None) -> None:
    root = project_root if project_root is not None else Path(__file__).resolve().parents[3]
    load_dotenv(root / ".env", override=False)

_load_environment()
TRANSCRIPT_REFINEMENT_USE_CLOUD = os.environ.get(
    "TRANSCRIPT_REFINEMENT_USE_CLOUD", "0"
).strip().lower() in {"1", "true", "yes", "on"}
ANTHROPIC_API_URL = os.environ.get("ANTHROPIC_API_URL", "").strip()
API_KEY = os.environ.get("API_KEY", "").strip()
ANTHROPIC_VERSION = os.environ.get("ANTHROPIC_VERSION", "").strip()
MODEL = os.environ.get("MODEL", "").strip()
```

- [ ] **Step 4: Run green config checks and verify no filesystem/runtime initialization is introduced.** Re-run the command from Step 2; all config tests must pass. Ensure test subprocess output contains assertions only, not configuration dumps.
- [ ] **Step 5: Review scoped changes.** Candidate commit when authorized: `feat: load transcript provider settings from root dotenv`. Stage only Task 1 files, never the user's `.gitignore` or entrypoint edits.

### Task 2: Mandatory Boundary Chunking And Shared Instructions

**Files:**
- Modify: `src/face_voice_rec/domain/transcription/transcript_refiner.py`
- Test: `tests/unit/test_transcript_refiner.py`

**Interfaces:**
- Preserve: `_chunk_ranges(segments: Sequence[dict]) -> list[tuple[int, int]]`, returning half-open target ranges.
- Preserve: `_normalize_and_validate_response(response, targets, all_segments)` and all existing rejection behavior.
- Produces: shared `_SYSTEM_PROMPT` containing the configured gap limit, exact-word preservation, JSON output shape, and untrusted/read-only input constraints.

- [ ] **Step 1: Add boundary range tests without changing existing malicious-response tests.**

```python
def test_chunks_stop_at_mandatory_boundaries(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_MERGE_MAX_GAP_MS", 1500)
    items = _segments(
        ("Alice", 0, 500, "One."),
        ("Alice", 500, 900, "Two."),
        ("Bob", 1000, 1400, "Three."),
        ("Bob", 1300, 1800, "Four."),
        ("Bob", 3301, 3800, "Five."),
    )
    assert transcript_refiner._chunk_ranges(items) == [(0, 2), (2, 3), (3, 4), (4, 5)]
```

Add exact-1500-ms and zero-gap acceptance, character/segment caps within a run, single oversized target, and read-only context exclusion tests. Existing fakes that return refs outside a newly bounded target may now fail with `non-target`; update those tests to call the validator directly with all original targets so they still specifically exercise speaker/overlap rejection rather than weaken assertions.

- [ ] **Step 2: Run red check.**

Run: `directmlvenv/Scripts/python.exe -m pytest tests/unit/test_transcript_refiner.py -q`
Expected: new boundary splitting tests fail against current size-only chunking.

- [ ] **Step 3: Add the boundary check inside the chunk accumulator before adding a nonfirst target.** Preserve existing size limits and use the same rule as validation.

```python
if end > start:
    previous, current = segments[end - 1], segments[end]
    gap = int(current["start_ms"]) - int(previous["end_ms"])
    if (
        current["speaker_label"] != previous["speaker_label"]
        or gap < 0
        or gap > config.TRANSCRIPT_MERGE_MAX_GAP_MS
    ):
        break
```

Extract system wording from `_post_ollama_chat` into the shared prompt and state the numeric configured gap, read-only context, exact target coverage, and JSON shape. Preserve Ollama's schema request field and unchanged validator.

- [ ] **Step 4: Re-run the refiner tests; every malicious merge, word-change, duplicate, omission, and chronology check still rejects.**
- [ ] **Step 5: Review scoped changes.** Candidate commit when authorized: `fix: split refinement requests at mandatory transcript boundaries`.

### Task 3: Foundry Transport, Whole-Transcript Fallback, And Provenance

**Files:**
- Modify: `src/face_voice_rec/domain/transcription/transcript_refiner.py`
- Test: `tests/unit/test_transcript_refiner.py`

**Interfaces:**
- Add: `_post_foundry_chat(user_content: str) -> dict` using new config exports.
- Add: `_refine_with_provider(normalized: Sequence[dict], provider: str) -> RefinementResult` where provider is `foundry` or `ollama`; it owns fresh chunk accumulators per call.
- Extend frozen `RefinementResult` after its existing fields with `provider: str = "ollama"`, `model: str = ""`, `used_local_fallback: bool = False`, `fallback_reason: str | None = None`. Defaults preserve existing two-positional-argument construction in tests/callers.
- `refine_transcript(segments) -> RefinementResult` still normalizes once and performs no I/O for empty input. Nonempty successful production results always fill provider/model.

- [ ] **Step 1: Add red HTTP-contract and fallback tests.** Use fake context-managed responses and patch the module HTTP opener; never contact actual hosts. Assert the full configured URL is unchanged and headers include the fake API key/version. Set `MODEL` to a deployment alias, not a guessed model ID.

```python
def test_cloud_failure_restarts_every_chunk_locally(monkeypatch):
    import json
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_USE_CLOUD", True)
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_CHUNK_SEGMENTS", 1)
    monkeypatch.setattr(config, "MODEL", "test-deployment")
    calls = []
    items = _segments(("Alice", 0, 500, "Hello."), ("Alice", 600, 900, "Again."))

    def cloud(payload):
        targets = json.loads(payload)["target_segments"]
        calls.append(("cloud", targets[0]["segment_id"]))
        if targets[0]["segment_id"] == 2:
            raise TranscriptRefinementError("Foundry request timed out")
        return _reply(targets, recommendations=[
            {"after_segment_id": 1, "reason": "Cloud-only recommendation"}
        ])

    def local(payload):
        targets = json.loads(payload)["target_segments"]
        calls.append(("local", targets[0]["segment_id"]))
        return _reply(targets)

    monkeypatch.setattr(transcript_refiner, "_post_foundry_chat", cloud)
    monkeypatch.setattr(transcript_refiner, "_post_ollama_chat", local)
    result = transcript_refiner.refine_transcript(items)
    assert calls == [("cloud", 1), ("cloud", 2), ("local", 1), ("local", 2)]
    assert result.provider == "ollama"
    assert result.model == config.TRANSCRIPT_REFINEMENT_MODEL
    assert result.used_local_fallback is True
    assert result.speaker_change_recommendations == ()
```

Also test cloud success never calls local; local-only never calls cloud; missing configuration, invalid HTTPS URL, 401/429/500, timeout, malformed JSON, nonobject JSON, bad structure, changed words, truncation, and tool-use-only output trigger local fallback. Reject credential-bearing URL userinfo, fragments, and query strings rather than risk credential leakage. Test empty input and invalid source data never call either provider. Test direct local errors and dual failures remain sanitized and leave callers able to use deterministic output.

- [ ] **Step 2: Run the refiner red check using Task 2's command.** New transport/provenance tests fail before implementation.
- [ ] **Step 3: Implement transport and orchestration.** Use `urllib.request.build_opener` with a `HTTPRedirectHandler` subclass returning `None` from `redirect_request` for authenticated cloud traffic; reject redirects rather than resend secrets. Keep a patchable module-level cloud opener seam for tests. Validate config using `urlsplit`: HTTPS, nonempty hostname, no userinfo/query/fragment, path ending `/anthropic/v1/messages`, nonempty key/version/model, and no header control characters.

```python
payload = {
    "model": config.MODEL,
    "max_tokens": 8192,
    "stream": False,
    "system": _SYSTEM_PROMPT,
    "messages": [{"role": "user", "content": user_content}],
}
headers = {
    "Content-Type": "application/json",
    "x-api-key": config.API_KEY,
    "anthropic-version": config.ANTHROPIC_VERSION,
}
```

Require `stop_reason == "end_turn"`, collect only string-valued text blocks from `content`, join them, parse strict JSON, and require an object. Reject tool-use blocks even if text exists. Do not silently strip explanations or fenced Markdown. Catch expected HTTP/network/decode/shape errors and raise provider/category-only `TranscriptRefinementError` messages with suppressed exception chaining. Do not catch programmer exceptions broadly. Sanitize existing Ollama errors too: remove URLs and arbitrary transport exception text from public errors.

Move the current chunk request loop into `_refine_with_provider`, selecting exactly one request function for its entire run. After source normalization, cloud-first routing is:

```python
try:
    return _refine_with_provider(normalized, "foundry")
except TranscriptRefinementError as cloud_error:
    cloud_reason = str(cloud_error)
try:
    local_result = _refine_with_provider(normalized, "ollama")
except TranscriptRefinementError:
    raise TranscriptRefinementError("Foundry and local Ollama refinement failed") from None
return replace(local_result, used_local_fallback=True, fallback_reason=cloud_reason)
```

Import `replace` from dataclasses. Local-only calls `_refine_with_provider` once. Empty input returns empty groups with provider `deterministic`, model empty, and no fallback flag. Keep normalization failures outside the cloud try block.

- [ ] **Step 4: Run all refiner tests and verify redaction using secret sentinels in exception bodies.** Include redirect tests that assert a second HTTP destination is never invoked and headers never reach it.
- [ ] **Step 5: Review scoped changes.** Candidate commit when authorized: `feat: add Foundry refinement with complete local fallback`.

### Task 4: Persist Actual Attribution, Update UI, And Document Setup

**Files:**
- Modify: `src/face_voice_rec/domain/transcription/transcription_core.py`
- Modify: `app/pages/transcripts.py`
- Modify: `README.md`
- Test: `tests/unit/test_transcription_core.py`
- Create: `tests/unit/test_transcript_refinement_ui.py`

**Interfaces:**
- Consume `RefinementResult.provider`, `.model`, `.used_local_fallback`, `.fallback_reason`.
- Preserve `refresh_readable_transcript(meeting_id: int, use_llm: Optional[bool] = None) -> int` and SQLite schema.
- Add page presentation helper `_refinement_status_caption(status: str, method: str) -> str`; it parses saved methods, never consults current provider config for historical labels.
- Persist methods `foundry:<model>`, `ollama:<model>`, or `deterministic`; status `complete`, `complete_local_fallback`, `fallback`, or `disabled` as specified.

- [ ] **Step 1: Add temporary-DB persistence tests and pure caption tests.** Reuse the existing `test_ollama_failure_falls_back_without_losing_detailed_transcript` fixture pattern: fake extraction/transcription, patch the refiner result, query run and detailed rows. Parameterize cloud/direct-local/local-fallback provenance. Build the original `RefinementResult` fixtures with real metadata and ensure stored models match results even when current config differs.

```python
@pytest.mark.parametrize("status,method,expected", [
    ("complete", "foundry:test-deployment", "Cloud"),
    ("complete", "ollama:test-local", "Local Ollama"),
    ("complete_local_fallback", "ollama:test-local", "cloud fallback"),
    ("fallback", "deterministic", "Deterministic"),
    ("disabled", "deterministic", "Deterministic"),
])
def test_caption_uses_saved_provenance(status, method, expected):
    from app.pages.transcripts import _refinement_status_caption
    assert expected in _refinement_status_caption(status, method)
```

Keep UI tests import-safe, using the repository's existing Streamlit stubs if required. Add a lightweight fake-Streamlit render test for configured-provider action text and successful local-fallback info; no real browser/server or live DB. Confirm UI never labels `complete_local_fallback` as deterministic. Add an empty-meeting test that keeps deterministic attribution and makes no network call. Keep existing tests demonstrating source text survives dual provider failure.

- [ ] **Step 2: Run red checks.**

Run: `directmlvenv/Scripts/python.exe -m pytest tests/unit/test_transcription_core.py tests/unit/test_transcript_refinement_ui.py -q`
Expected: new metadata and UI checks fail against hard-coded Ollama/local-only wording.

- [ ] **Step 3: Use result provenance in persistence and provider-neutral UI.**

```python
method = f"{result.provider}:{result.model}"
status = "complete_local_fallback" if result.used_local_fallback else "complete"
error_message = result.fallback_reason
```

Guard the no-detailed-input path so it remains deterministic. Replace `logger.exception` for expected refinement rejection with a warning containing only meeting ID and sanitized category; never log raw provider exceptions. Keep deterministic persistence after any expected refinement failure.

The page's action names the configured primary provider; its saved caption names the actual historical provider. Split method on the first colon. Show `Local Ollama: <model> (cloud fallback)` after successful fallback, with sanitized cloud reason as informational text. Keep a warning only for deterministic failure. Preserve display of existing historical `ollama:*` runs and generic unknown stored methods rather than crashing.

Update README with root-based startup and placeholder-only dotenv settings:

```dotenv
TRANSCRIPT_REFINEMENT_ENABLED=true
TRANSCRIPT_REFINEMENT_USE_CLOUD=true
ANTHROPIC_API_URL=https://example.services.ai.azure.com/anthropic/v1/messages
API_KEY=your-foundry-key
ANTHROPIC_VERSION=2023-06-01
MODEL=your-deployment-name
TRANSCRIPT_REFINEMENT_URL=http://localhost:11434
TRANSCRIPT_REFINEMENT_MODEL=qwen2.5:7b
```

Explain `USE_CLOUD=false`, process-variable precedence, restarting Streamlit after changes, cloud transcript disclosure, no audio upload, strict validation, cloud-to-local-to-deterministic behavior, and real saved attribution. Clarify that changing settings does not automatically reprocess old meetings; the refinement action regenerates their readable view. Use `streamlit run app/main.py` from repository root or `streamlit run main.py` from app, with no dependency on the cwd for dotenv loading.

- [ ] **Step 4: Run focused green checks, then the packaged suite.**

Run: `directmlvenv/Scripts/python.exe -m pytest tests/unit/test_config.py tests/unit/test_transcript_refiner.py tests/unit/test_transcription_core.py tests/unit/test_transcript_refinement_ui.py -q`

Then run: `directmlvenv/Scripts/python.exe -m pytest tests/unit tests/integration -q`

Expected: all packaged checks pass. If the user's real dotenv enables cloud mode, isolate refinement flags/settings in test fixtures with monkeypatch before requests; never permit real network calls as a workaround. Run editor diagnostics on changed Python files. Record exact results, avoid asserting a real cloud deployment was verified.

- [ ] **Step 5: Review final scope and summarize.** Verify only planned files changed, pre-existing edits remain untouched, no secrets appear in diff, and no `.env` is staged. Candidate commit when authorized: `feat: persist and display transcript refinement provider`. Tell the user the config boolean, restart requirement, tested fallback behavior, and that live cloud connectivity/model quality were not tested.

## Plan Self-Review

- Configuration/defaults/precedence/dependencies: Task 1.
- Boundary-safe input and unchanged validation: Task 2.
- Documented Foundry protocol, redirects, sanitization, all-or-nothing fallback: Task 3.
- Historical attribution, deterministic preservation, UI wording, README: Task 4.
- Existing interfaces remain compatible; new result fields have defaults, production attempts populate them explicitly.
- Empty/invalid input performs no model requests; UI attribution does not rely on current configuration.
- No live data, secret reads, real inference, schema changes, or unrelated cleanup are authorized.