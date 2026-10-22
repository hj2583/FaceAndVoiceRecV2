# OpenAI-Compatible Local Transcript Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let local transcript refinement call an OpenAI-compatible server (llama.cpp `llama-server`) instead of Ollama, selected by `TRANSCRIPT_REFINEMENT_API`.

**Architecture:** A validated setting selects the local provider ID (`ollama` or `openai`). A new client posts to `/v1/chat/completions` with a JSON-schema `response_format` and reuses the existing bounded-read, parse, and sanitized-error helpers. Routing, saved attribution (`openai:<model>`), and UI labels use the configured local provider; Ollama remains the default and its behavior does not change.

**Tech Stack:** Python 3.10+, `urllib.request`, Streamlit, pytest.

Spec: `docs/superpowers/specs/2026-10-22-openai-compatible-local-refinement-design.md`

## Global Constraints

- Allowed `TRANSCRIPT_REFINEMENT_API` values: `ollama` (default) and `openai`. Values are trimmed and case-insensitive. An empty value means `ollama`. Any other value raises `ValueError` when settings load.
- The URL is the server root without `/v1`; the client appends `/v1/chat/completions`.
- Error messages must never include URLs, response bodies, transcript text, or exception text; they use `from None`.
- Provider labels:
  - Errors: `Local Ollama` and `Local OpenAI-compatible server`.
  - Combined fallback error: `local Ollama` and `local OpenAI-compatible server`.
  - UI caption: `Local OpenAI-compatible: <model>`.
- Saved method: `openai:<model>`. Existing `ollama:<model>` behavior must not change.
- Python for all commands: `D:\Git\FaceAndVoiceRecV2\directmlvenv\Scripts\python.exe`, written below as `PY`.
- Commit only the files listed for each task. Never commit `.env`.

## File Map

| File | Change |
|---|---|
| `src/face_voice_rec/config/settings.py` | Add the `_refinement_api()` validator, the `TRANSCRIPT_REFINEMENT_API` setting, and its `__all__` entry |
| `src/face_voice_rec/config/__init__.py` | Add `TRANSCRIPT_REFINEMENT_API` to `__all__` |
| `src/face_voice_rec/domain/transcription/transcript_refiner.py` | Add the OpenAI client, local-provider helpers, and routing |
| `src/face_voice_rec/domain/transcription/transcription_core.py` | Accept `openai` when saving refinement methods |
| `app/pages/transcripts.py` | Update the caption and the action, spinner, and hint labels |
| `README.md` | Document the setting and add a `llama-server` example |
| `tests/unit/test_config.py`, `tests/unit/test_transcript_refiner.py`, `tests/unit/test_transcription_core.py`, `tests/unit/test_transcript_refinement_ui.py` | Add tests |

---

### Task 1: `TRANSCRIPT_REFINEMENT_API` setting

**Files:**
- Modify: `src/face_voice_rec/config/settings.py` (after `TRANSCRIPT_REFINEMENT_MODEL`, and in `__all__`)
- Modify: `src/face_voice_rec/config/__init__.py` (in `__all__`, after `"TRANSCRIPT_REFINEMENT_MODEL"`)
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces:
  - `settings._refinement_api(value: str | None) -> str`, which returns `"ollama"` or `"openai"`.
  - `config.TRANSCRIPT_REFINEMENT_API: str`.

- [ ] **Step 1: Write failing tests** (append to `tests/unit/test_config.py`):

```python
@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, "ollama"), ("", "ollama"), ("ollama", "ollama"), (" OpenAI ", "openai")],
)
def test_refinement_api_accepts_known_values(raw, expected):
    from face_voice_rec.config import settings

    assert settings._refinement_api(raw) == expected


@pytest.mark.parametrize("raw", ["vllm", "openai-compatible", "llama"])
def test_refinement_api_rejects_unknown_values(raw):
    from face_voice_rec.config import settings

    with pytest.raises(ValueError, match="TRANSCRIPT_REFINEMENT_API"):
        settings._refinement_api(raw)


def test_refinement_api_is_exported():
    from face_voice_rec.config import settings

    assert config.TRANSCRIPT_REFINEMENT_API in {"ollama", "openai"}
    assert "TRANSCRIPT_REFINEMENT_API" in settings.__all__
    assert "TRANSCRIPT_REFINEMENT_API" in config.__all__
```

Also isolate copied-settings subprocesses from the developer's `.env`. In `_run_copied_settings`, add this after `env.pop("TRANSCRIPT_REFINEMENT_ENABLED", None)`:

```python
    env.pop("TRANSCRIPT_REFINEMENT_API", None)
```

Then add this line to the body of `test_cloud_settings_default_inert_without_dotenv`:

```python
        assert copied.TRANSCRIPT_REFINEMENT_API == "ollama"
```

- [ ] **Step 2: Run and confirm failure**

Run: `PY -m pytest tests/unit/test_config.py -q`
Expected: FAIL with `AttributeError: ... '_refinement_api'` and a missing `TRANSCRIPT_REFINEMENT_API`.

- [ ] **Step 3: Implement** in `settings.py`, directly after the `TRANSCRIPT_REFINEMENT_MODEL` assignment:

```python
_REFINEMENT_APIS = ("ollama", "openai")


def _refinement_api(value: str | None) -> str:
    api = (value or "").strip().lower() or "ollama"
    if api not in _REFINEMENT_APIS:
        raise ValueError(
            "TRANSCRIPT_REFINEMENT_API must be one of: " + ", ".join(_REFINEMENT_APIS)
        )
    return api


TRANSCRIPT_REFINEMENT_API = _refinement_api(os.environ.get("TRANSCRIPT_REFINEMENT_API"))
```

Add `"TRANSCRIPT_REFINEMENT_API",` after `"TRANSCRIPT_REFINEMENT_MODEL",` in `__all__` in `settings.py` and in `config/__init__.py`. `__init__.py` uses tabs for indentation.

- [ ] **Step 4: Run and confirm pass**

Run: `PY -m pytest tests/unit/test_config.py -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add -- src/face_voice_rec/config/settings.py src/face_voice_rec/config/__init__.py tests/unit/test_config.py
git commit -m "feat: add TRANSCRIPT_REFINEMENT_API setting" -- src/face_voice_rec/config/settings.py src/face_voice_rec/config/__init__.py tests/unit/test_config.py
```

---

### Task 2: OpenAI-compatible chat client

**Files:**
- Modify: `src/face_voice_rec/domain/transcription/transcript_refiner.py` (add the client after `_post_ollama_chat`; add `_THINK_PREFIX` near `_MAX_RESPONSE_BYTES`)
- Test: `tests/unit/test_transcript_refiner.py` (append at the end; reuse `_FakeOpener`, `_FakeResponse`, `_http_error`, `_assert_sanitized`, `_GOOD_REPLY_TEXT`, `_BODY_SENTINEL`, and `_ERROR_SENTINEL`)

**Interfaces:**
- Consumes: `config.TRANSCRIPT_REFINEMENT_URL`, `config.TRANSCRIPT_REFINEMENT_MODEL`, `config.TRANSCRIPT_REFINEMENT_TIMEOUT_SECONDS`, `_read_bounded_body`, `_parse_structured_response`, `_transport_failure_kind`, `_OUTPUT_SCHEMA`, and `_SYSTEM_PROMPT`.
- Produces: `_post_openai_compatible_chat(user_content: str) -> dict`, which uses module-level `urlopen` so tests can monkeypatch it.

- [ ] **Step 1: Write failing tests** (append):

```python
# ---- OpenAI-compatible local server ----


def _openai_envelope(content):
    return json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": content}}]}
    ).encode("utf-8")


def test_openai_compatible_request_contract(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_URL", "http://lan-host.invalid:30000")
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_MODEL", "remote-model")
    sent = {}

    def fake_urlopen(request, timeout):
        sent["request"] = request
        sent["timeout"] = timeout
        return _FakeResponse(_openai_envelope(_GOOD_REPLY_TEXT))

    monkeypatch.setattr(transcript_refiner, "urlopen", fake_urlopen)

    result = transcript_refiner._post_openai_compatible_chat('{"target_segments": []}')

    assert result["groups"][0]["text"] == "Hello."
    request = sent["request"]
    assert request.full_url == "http://lan-host.invalid:30000/v1/chat/completions"
    assert request.get_method() == "POST"
    headers = {key.lower(): value for key, value in request.header_items()}
    assert headers["content-type"] == "application/json"
    assert "authorization" not in headers
    assert json.loads(request.data.decode("utf-8")) == {
        "model": "remote-model",
        "temperature": 0,
        "stream": False,
        "messages": [
            {"role": "system", "content": transcript_refiner._SYSTEM_PROMPT},
            {"role": "user", "content": '{"target_segments": []}'},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "transcript_refinement",
                "schema": transcript_refiner._OUTPUT_SCHEMA,
            },
        },
    }
    assert sent["timeout"] == config.TRANSCRIPT_REFINEMENT_TIMEOUT_SECONDS


@pytest.mark.parametrize(
    "content",
    [
        f"<think>reasoning about it</think>\n{_GOOD_REPLY_TEXT}",
        f"<think>\nmulti\nline\n</think>\n\n```json\n{_GOOD_REPLY_TEXT}\n```",
        f"```json\n{_GOOD_REPLY_TEXT}\n```",
    ],
    ids=["think", "think-fenced", "fenced"],
)
def test_openai_compatible_strips_think_and_fences(monkeypatch, content):
    monkeypatch.setattr(
        transcript_refiner,
        "urlopen",
        lambda request, timeout: _FakeResponse(_openai_envelope(content)),
    )

    assert transcript_refiner._post_openai_compatible_chat("{}")["groups"][0]["text"] == "Hello."


@pytest.mark.parametrize(
    "outcome",
    [
        urllib.error.URLError(f"http://lan-host.invalid {_ERROR_SENTINEL}"),
        _http_error(500),
        TimeoutError(_ERROR_SENTINEL),
        OSError(_ERROR_SENTINEL),
        b"not json " + _BODY_SENTINEL.encode(),
        b"\xff\xfe",
        b"[]",
        json.dumps({"choices": []}).encode(),
        json.dumps({"choices": [{"message": {"content": None}}]}).encode(),
        json.dumps({"choices": [{"message": {"content": 5}}]}).encode(),
        _openai_envelope(f"not json {_BODY_SENTINEL}"),
        b"[" * 200000,
    ],
    ids=[
        "url-error", "http-500", "timeout", "os-error", "malformed", "non-utf8",
        "non-object", "no-choices", "null-content", "non-string-content",
        "bad-content", "deep-nesting",
    ],
)
def test_openai_compatible_errors_are_sanitized(monkeypatch, outcome):
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_URL", "http://lan-host.invalid:30000")

    def fake_urlopen(request, timeout):
        return _FakeOpener(outcome).open(request, timeout)

    monkeypatch.setattr(transcript_refiner, "urlopen", fake_urlopen)

    with pytest.raises(TranscriptRefinementError) as caught:
        transcript_refiner._post_openai_compatible_chat("{}")

    assert "lan-host" not in str(caught.value)
    assert str(caught.value).startswith(
        ("Local OpenAI-compatible server", "Refinement response must be")
    )
    _assert_sanitized(caught.value)


def test_openai_compatible_http_error_reports_status_only(monkeypatch):
    monkeypatch.setattr(
        transcript_refiner,
        "urlopen",
        lambda request, timeout: _FakeOpener(_http_error(503)).open(request, timeout),
    )

    with pytest.raises(TranscriptRefinementError) as caught:
        transcript_refiner._post_openai_compatible_chat("{}")

    assert str(caught.value) == "Local OpenAI-compatible server request failed (HTTP 503)"


def test_openai_compatible_oversized_response_is_rejected(monkeypatch):
    monkeypatch.setattr(transcript_refiner, "_MAX_RESPONSE_BYTES", 64)
    body = _openai_envelope(_BODY_SENTINEL + "x" * 200)
    monkeypatch.setattr(
        transcript_refiner, "urlopen", lambda request, timeout: _FakeResponse(body)
    )

    with pytest.raises(TranscriptRefinementError, match="size limit") as caught:
        transcript_refiner._post_openai_compatible_chat("{}")

    _assert_sanitized(caught.value)
```

- [ ] **Step 2: Run and confirm failure**

Run: `PY -m pytest tests/unit/test_transcript_refiner.py -q -k openai_compatible`
Expected: FAIL with `AttributeError: ... '_post_openai_compatible_chat'`.

- [ ] **Step 3: Implement.** Add this next to `_MAX_RESPONSE_BYTES`:

```python
_THINK_PREFIX = re.compile(r"^\s*<think>.*?</think>\s*", flags=re.DOTALL)
```

Add this function directly after `_post_ollama_chat`:

```python
def _post_openai_compatible_chat(user_content: str) -> dict:
    provider = "Local OpenAI-compatible server"
    payload = {
        "model": config.TRANSCRIPT_REFINEMENT_MODEL,
        "temperature": 0,
        "stream": False,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "transcript_refinement", "schema": _OUTPUT_SCHEMA},
        },
    }
    request = Request(
        f"{config.TRANSCRIPT_REFINEMENT_URL}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=config.TRANSCRIPT_REFINEMENT_TIMEOUT_SECONDS) as response:
            envelope = json.loads(_read_bounded_body(response, provider).decode("utf-8"))
    except HTTPError as error:
        raise TranscriptRefinementError(
            f"{provider} request failed (HTTP {error.code})"
        ) from None
    except (URLError, TimeoutError, OSError, HTTPException) as error:
        raise TranscriptRefinementError(
            f"{provider} request {_transport_failure_kind(error)}"
        ) from None
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
        raise TranscriptRefinementError(
            f"{provider} returned an invalid structured response"
        ) from None

    try:
        content = envelope["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise TranscriptRefinementError(
            f"{provider} returned an invalid structured response"
        ) from None
    if not isinstance(content, str):
        raise TranscriptRefinementError(
            f"{provider} returned an invalid structured response"
        )
    return _parse_structured_response(_THINK_PREFIX.sub("", content, count=1), provider)
```

`_read_bounded_body` raises `TranscriptRefinementError`. Because that is a `RuntimeError`, not an `OSError` or `ValueError`, it reaches the caller unchanged, matching the Ollama client.

- [ ] **Step 4: Run and confirm pass**

Run: `PY -m pytest tests/unit/test_transcript_refiner.py -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add -- src/face_voice_rec/domain/transcription/transcript_refiner.py tests/unit/test_transcript_refiner.py
git commit -m "feat: add OpenAI-compatible local refinement client" -- src/face_voice_rec/domain/transcription/transcript_refiner.py tests/unit/test_transcript_refiner.py
```

---

### Task 3: Routing and saved attribution

**Files:**
- Modify: `src/face_voice_rec/domain/transcription/transcript_refiner.py`: module docstring, `_refine_with_provider`, and `refine_transcript`
- Modify: `src/face_voice_rec/domain/transcription/transcription_core.py`: the line containing `if result_provider in {"foundry", "ollama"} and result_model:`
- Test: `tests/unit/test_transcript_refiner.py` and `tests/unit/test_transcription_core.py`

**Interfaces:**
- Consumes: `config.TRANSCRIPT_REFINEMENT_API` from Task 1 and `_post_openai_compatible_chat` from Task 2.
- Produces:
  - `_LOCAL_PROVIDER_NAMES: dict[str, str]`, which maps `ollama` to `Ollama` and `openai` to `OpenAI-compatible server`.
  - `_local_provider() -> str`.
  - `RefinementResult.provider == "openai"` for runs that use the OpenAI-compatible server.

- [ ] **Step 1: Write failing tests.**

In `tests/unit/test_transcript_refiner.py`, add this line inside the existing autouse fixture `_isolated_refiner_config`:

```python
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_API", "ollama")
```

Then append:

```python
def test_openai_api_routes_local_only_refinement(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_API", "openai")
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_MODEL", "remote-model")
    monkeypatch.setattr(
        transcript_refiner, "_post_ollama_chat",
        lambda _payload: pytest.fail("Ollama must not be called"),
    )
    monkeypatch.setattr(
        transcript_refiner,
        "_post_openai_compatible_chat",
        lambda payload: _reply(json.loads(payload)["target_segments"]),
    )

    result = transcript_refiner.refine_transcript(_segments(("Alice", 0, 500, "Hello.")))

    assert (result.provider, result.model) == ("openai", "remote-model")
    assert result.used_local_fallback is False


def test_cloud_failure_falls_back_to_openai_compatible(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_USE_CLOUD", True)
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_API", "openai")

    def cloud(_payload):
        raise TranscriptRefinementError("Foundry request timed out")

    monkeypatch.setattr(transcript_refiner, "_post_foundry_chat", cloud)
    monkeypatch.setattr(
        transcript_refiner,
        "_post_openai_compatible_chat",
        lambda payload: _reply(json.loads(payload)["target_segments"]),
    )

    result = transcript_refiner.refine_transcript(_segments(("Alice", 0, 500, "Hello.")))

    assert result.provider == "openai"
    assert result.used_local_fallback is True
    assert result.fallback_reason == "Foundry request timed out"


def test_both_failing_names_openai_compatible_local_provider(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_USE_CLOUD", True)
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_API", "openai")

    def fail_cloud(_payload):
        raise TranscriptRefinementError("Foundry request timed out")

    def fail_local(_payload):
        raise TranscriptRefinementError("Local OpenAI-compatible server request timed out")

    monkeypatch.setattr(transcript_refiner, "_post_foundry_chat", fail_cloud)
    monkeypatch.setattr(transcript_refiner, "_post_openai_compatible_chat", fail_local)

    with pytest.raises(TranscriptRefinementError) as caught:
        transcript_refiner.refine_transcript(_segments(("Alice", 0, 500, "Hello.")))

    assert str(caught.value) == (
        "Foundry refinement failed (Foundry request timed out); "
        "local OpenAI-compatible server refinement failed "
        "(Local OpenAI-compatible server request timed out)"
    )


def test_openai_invalid_chunk_keeps_source_text(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_API", "openai")
    items = _segments(("Alice", 0, 500, "Hello."))
    monkeypatch.setattr(
        transcript_refiner,
        "_post_openai_compatible_chat",
        lambda _payload: _reply(items, text_by_id={1: "Goodbye."}),
    )

    result = transcript_refiner.refine_transcript(items)

    assert result.provider == "openai"
    assert [(group.source_segment_ids, group.text) for group in result.groups] == [
        ((1,), "Hello.")
    ]
```

In `tests/unit/test_transcription_core.py`, add this `pytest.param` to the list for `test_refresh_readable_transcript_persists_refinement_result_provenance`, after the `local-primary` entry:

```python
        pytest.param(
            (
                "openai",
                "unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL",
                False,
                None,
            ),
            "complete",
            "openai:unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL",
            None,
            id="local-openai-compatible",
        ),
```

- [ ] **Step 2: Run and confirm failure**

Run: `PY -m pytest tests/unit/test_transcript_refiner.py tests/unit/test_transcription_core.py -q -k "openai or provenance"`
Expected:
- Refiner tests fail because local-only routing calls Ollama.
- The `local-openai-compatible` provenance case fails because the saved method is `deterministic`.

- [ ] **Step 3: Implement.**

In `transcript_refiner.py`, change the module docstring to:

```python
"""Optional, validated readable-transcript refinement via cloud or local LLMs."""
```

Add these definitions directly above `_refine_with_provider`:

```python
_LOCAL_PROVIDER_NAMES = {"ollama": "Ollama", "openai": "OpenAI-compatible server"}


def _local_provider() -> str:
    return config.TRANSCRIPT_REFINEMENT_API
```

In `_refine_with_provider`, replace the provider selection with:

```python
    if provider == "foundry":
        post, model = _post_foundry_chat, config.MODEL
    elif provider == "ollama":
        post, model = _post_ollama_chat, config.TRANSCRIPT_REFINEMENT_MODEL
    elif provider == "openai":
        post, model = _post_openai_compatible_chat, config.TRANSCRIPT_REFINEMENT_MODEL
    else:
        raise ValueError(f"Unknown refinement provider: {provider}")
```

In the same function, replace `if provider != "ollama":` with:

```python
            if provider not in _LOCAL_PROVIDER_NAMES:
```

Then replace the warning call with:

```python
    if deterministic_chunks:
        logger.warning(
            "Local %s output failed validation for %d of %d chunks; "
            "kept source text for those chunks",
            _LOCAL_PROVIDER_NAMES[provider],
            deterministic_chunks,
            len(ranges),
        )
```

In `refine_transcript`, replace the provider-dispatch tail, starting at `if not config.TRANSCRIPT_REFINEMENT_USE_CLOUD:`, with:

```python
    local_provider = _local_provider()
    if not config.TRANSCRIPT_REFINEMENT_USE_CLOUD:
        return _refine_with_provider(normalized, local_provider)

    try:
        return _refine_with_provider(normalized, "foundry")
    except TranscriptRefinementError as cloud_error:
        cloud_reason = str(cloud_error)
    logger.warning("Foundry refinement failed (%s); retrying locally", cloud_reason)
    try:
        local_result = _refine_with_provider(normalized, local_provider)
    except TranscriptRefinementError as local_error:
        raise TranscriptRefinementError(
            "Foundry refinement failed "
            f"({cloud_reason}); local {_LOCAL_PROVIDER_NAMES[local_provider]} "
            f"refinement failed ({local_error})"
        ) from None
    return replace(local_result, used_local_fallback=True, fallback_reason=cloud_reason)
```

In `transcription_core.py`, change:

```python
            if result_provider in {"foundry", "ollama"} and result_model:
```

to:

```python
            if result_provider in {"foundry", "ollama", "openai"} and result_model:
```

- [ ] **Step 4: Run and confirm pass**

Run: `PY -m pytest tests/unit/test_transcript_refiner.py tests/unit/test_transcription_core.py -q`
Expected: all tests pass, including the existing Ollama test that expects `"local Ollama refinement failed (...)"`.

- [ ] **Step 5: Commit**

```powershell
git add -- src/face_voice_rec/domain/transcription/transcript_refiner.py src/face_voice_rec/domain/transcription/transcription_core.py tests/unit/test_transcript_refiner.py tests/unit/test_transcription_core.py
git commit -m "feat: route local refinement through configured API" -- src/face_voice_rec/domain/transcription/transcript_refiner.py src/face_voice_rec/domain/transcription/transcription_core.py tests/unit/test_transcript_refiner.py tests/unit/test_transcription_core.py
```

---

### Task 4: Meeting Transcripts UI labels

**Files:**
- Modify: `app/pages/transcripts.py`: `_provider_caption`, plus the label block under `if TRANSCRIPT_REFINEMENT_ENABLED:` and its `else` hint
- Test: `tests/unit/test_transcript_refinement_ui.py`

**Interfaces:**
- Consumes: `config.TRANSCRIPT_REFINEMENT_API`.
- Produces: `_local_backend_label() -> str`, which returns `"local Ollama"` or `"local OpenAI-compatible server"`.

- [ ] **Step 1: Write failing tests.** Add these rows to the `test_caption_uses_saved_provenance` parameter list:

```python
        ("complete", "openai:remote-model", "Local OpenAI-compatible: remote-model"),
        ("complete_local_fallback", "openai:remote-model", "cloud fallback"),
```

Append:

```python
@pytest.mark.parametrize(
    ("api", "use_cloud", "expected"),
    [
        ("ollama", False, "Refine readable transcript with local Ollama"),
        ("openai", False, "Refine readable transcript with local OpenAI-compatible server"),
        (
            "openai",
            True,
            "Refine readable transcript with cloud Foundry "
            "(fallback to local OpenAI-compatible server)",
        ),
    ],
)
def test_render_action_label_names_configured_local_backend(
    tmp_path, monkeypatch, api, use_cloud, expected
):
    db_path = tmp_path / "meeting.db"
    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(database, "DB_PATH", db_path)
    database.init_db()
    _seed_minimal_meeting(
        db_path,
        refinement_status="complete",
        refinement_method="deterministic",
    )
    module, fake_st = _load_transcripts_module(monkeypatch)
    monkeypatch.setattr(module, "TRANSCRIPT_REFINEMENT_ENABLED", True)
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_USE_CLOUD", use_cloud)
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_API", api)

    module.render()

    assert expected in fake_st.button_labels
```

- [ ] **Step 2: Run and confirm failure**

Run: `PY -m pytest tests/unit/test_transcript_refinement_ui.py -q`
Expected:
- The new caption row returns `openai: remote-model`, which fails.
- The OpenAI action-label cases fail.

- [ ] **Step 3: Implement.** In `_provider_caption`, add these lines after the `ollama` branch:

```python
    if provider == "openai":
        return f"Local OpenAI-compatible: {model}" if model else "Local OpenAI-compatible"
```

Add this function directly above `def render() -> None:`:

```python
def _local_backend_label() -> str:
    if config.TRANSCRIPT_REFINEMENT_API == "openai":
        return "local OpenAI-compatible server"
    return "local Ollama"
```

Replace the action-label block and the disabled hint with:

```python
    local_label = _local_backend_label()
    if TRANSCRIPT_REFINEMENT_ENABLED:
        if config.TRANSCRIPT_REFINEMENT_USE_CLOUD:
            action_label = (
                "Refine readable transcript with cloud Foundry "
                f"(fallback to {local_label})"
            )
            spinner_label = (
                "Asking the configured cloud Foundry model to format the transcript "
                "(local fallback on failure)..."
            )
        else:
            action_label = f"Refine readable transcript with {local_label}"
            spinner_label = (
                f"Asking the configured {local_label} model to format the transcript..."
            )
```

The `if st.button(...)` block stays unchanged. In the `else:` branch, use:

```python
        action_hint = (
            f"cloud Foundry (with {local_label} fallback)"
            if config.TRANSCRIPT_REFINEMENT_USE_CLOUD
            else local_label
        )
```

- [ ] **Step 4: Run and confirm pass**

Run: `PY -m pytest tests/unit/test_transcript_refinement_ui.py -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add -- app/pages/transcripts.py tests/unit/test_transcript_refinement_ui.py
git commit -m "feat: label OpenAI-compatible local refinement in UI" -- app/pages/transcripts.py tests/unit/test_transcript_refinement_ui.py
```

---

### Task 5: README, local `.env`, and full verification

**Files:**
- Modify: `README.md`, section "Optional readable transcript refinement"
- Modify locally without committing: `.env`

- [ ] **Step 1: Update the README.**

In the placeholder `dotenv` block, add `TRANSCRIPT_REFINEMENT_API=ollama` after `TRANSCRIPT_REFINEMENT_ENABLED=true`.

Replace the two behavior bullets that mention "local Ollama" with:

```markdown
- `TRANSCRIPT_REFINEMENT_ENABLED=true` and `TRANSCRIPT_REFINEMENT_USE_CLOUD=false` (default): local refinement only, through the configured local API.
- `TRANSCRIPT_REFINEMENT_ENABLED=true` and `TRANSCRIPT_REFINEMENT_USE_CLOUD=true`: cloud Foundry first, then automatic local fallback through the configured local API, then deterministic fallback if both fail.
```

After the existing "To enable local refinement" PowerShell block, add:

````markdown
`TRANSCRIPT_REFINEMENT_API` selects the local server API: `ollama` (default,
native `/api/chat`) or `openai` (OpenAI-compatible `/v1/chat/completions`, for
example llama.cpp `llama-server`). Set `TRANSCRIPT_REFINEMENT_URL` to the
server root without `/v1`, and set `TRANSCRIPT_REFINEMENT_MODEL` to the model
name the server exposes. Example for a `llama-server` on another machine:

```dotenv
TRANSCRIPT_REFINEMENT_ENABLED=true
TRANSCRIPT_REFINEMENT_API=openai
TRANSCRIPT_REFINEMENT_URL=http://192.168.1.50:30000
TRANSCRIPT_REFINEMENT_MODEL=your-served-model-name
TRANSCRIPT_REFINEMENT_TIMEOUT_SECONDS=300
```

Transcript text is sent to that server, and plain `http://` is unencrypted, so
use only a trusted network and server. Large models may need a longer timeout.
Runs are saved as `openai:<model>` and shown as "Local OpenAI-compatible".
````

- [ ] **Step 2: Update the user's `.env`** (local only; do not commit). Replace the existing `TRANSCRIPT_REFINEMENT_MODEL` and `TRANSCRIPT_REFINEMENT_URL` lines and add the API setting:

```dotenv
TRANSCRIPT_REFINEMENT_API=openai
TRANSCRIPT_REFINEMENT_MODEL=unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL
TRANSCRIPT_REFINEMENT_URL=http://192.168.137.27:30000
```

Keep `TRANSCRIPT_REFINEMENT_ENABLED=true`, `TRANSCRIPT_REFINEMENT_USE_CLOUD=false`, and the existing `TRANSCRIPT_REFINEMENT_TIMEOUT_SECONDS`. Its current value, `30000`, is already at least 300 seconds.

- [ ] **Step 3: Run the full suite**

Run: `PY -m pytest -q`
Expected: no new failures compared with the suite on the branch before Task 1. Report any pre-existing failures separately.

- [ ] **Step 4: Run a read-only connectivity check**

Run: `Invoke-RestMethod http://192.168.137.27:30000/v1/models`
Expected: JSON that lists the served model. If the listed `id` differs from the configured model, report the `id` so the user can update `TRANSCRIPT_REFINEMENT_MODEL`. `llama-server` usually ignores the model field when it serves one model.

- [ ] **Step 5: Commit the README only**

```powershell
git add -- README.md
git commit -m "docs: document OpenAI-compatible local refinement" -- README.md
```

Restart Streamlit afterward so it reloads `.env`.
