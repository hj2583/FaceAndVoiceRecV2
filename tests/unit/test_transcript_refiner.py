import json

import pytest

from face_voice_rec import config
from face_voice_rec.domain.transcription import transcript_refiner
from face_voice_rec.domain.transcription.transcript_refiner import TranscriptRefinementError


def _segments(*items):
    return [
        {
            "segment_id": index,
            "speaker_label": speaker,
            "start_ms": start,
            "end_ms": end,
            "text": text,
        }
        for index, (speaker, start, end, text) in enumerate(items, start=1)
    ]


def _reply(target_segments, group_refs=None, text_by_id=None, recommendations=None):
    ids = [item["segment_id"] for item in target_segments]
    groups = []
    if group_refs is None:
        group_refs = [[segment_id] for segment_id in ids]
    for refs in group_refs:
        text = " ".join(
            (text_by_id or {}).get(
                segment_id,
                next(item["text"] for item in target_segments if item["segment_id"] == segment_id),
            )
            for segment_id in refs
        )
        groups.append({"source_segment_ids": refs, "text": text})
    return {
        "groups": groups,
        "speaker_change_recommendations": recommendations or [],
    }


def test_refiner_merges_adjacent_fragments_for_same_speaker(monkeypatch):
    items = _segments(
        ("Alice", 0, 900, "Hello"),
        ("Alice", 950, 1800, "there."),
    )
    monkeypatch.setattr(
        transcript_refiner,
        "_post_ollama_chat",
        lambda _payload: _reply(
            items,
            [[1, 2]],
            text_by_id={1: "Hello", 2: "there."},
        ),
    )

    result = transcript_refiner.refine_transcript(items)

    assert result.groups[0].source_segment_ids == (1, 2)
    assert result.groups[0].text == "Hello there."


def test_refiner_never_merges_different_speakers_or_unknown_with_known(monkeypatch):
    items = _segments(
        ("Alice", 0, 500, "Good morning."),
        ("UNKNOWN", 550, 900, "Hello."),
        ("Bob", 950, 1400, "Hi."),
    )

    with pytest.raises(TranscriptRefinementError, match="speaker change"):
        transcript_refiner._normalize_and_validate_response(
            _reply(items, [[1, 2, 3]]),
            items,
            items,
        )


def test_unknown_fragment_between_known_speech_stays_unresolved(monkeypatch):
    items = _segments(
        ("Alice", 0, 500, "Good morning."),
        ("UNKNOWN", 550, 900, "Hello."),
        ("Alice", 950, 1400, "Welcome."),
    )
    monkeypatch.setattr(
        transcript_refiner,
        "_post_ollama_chat",
        lambda payload: _reply(json.loads(payload)["target_segments"]),
    )

    result = transcript_refiner.refine_transcript(items)

    assert [group.source_segment_ids for group in result.groups] == [(1,), (2,), (3,)]
    assert [items[segment_id - 1]["speaker_label"] for group in result.groups
            for segment_id in group.source_segment_ids] == ["Alice", "UNKNOWN", "Alice"]


def test_refiner_rejects_overlapping_speech_merge(monkeypatch):
    items = _segments(
        ("Alice", 0, 800, "We agree"),
        ("Alice", 700, 1200, "today."),
    )

    with pytest.raises(TranscriptRefinementError, match="overlap"):
        transcript_refiner._normalize_and_validate_response(
            _reply(items, [[1, 2]]),
            items,
            items,
        )


def test_refiner_preserves_multilingual_words_without_translation(monkeypatch):
    text = "Selamat pagi, semua."
    items = _segments(("UNKNOWN", 0, 1200, text))
    monkeypatch.setattr(
        transcript_refiner,
        "_post_ollama_chat",
        lambda _payload: _reply(items),
    )

    result = transcript_refiner.refine_transcript(items)

    assert result.groups[0].text == text


@pytest.mark.parametrize(
    "item,match",
    [
        ({"segment_id": 1, "speaker_label": "A", "start_ms": 1, "end_ms": 1, "text": "x"}, "invalid source"),
        ({"segment_id": 1, "speaker_label": "A", "start_ms": None, "end_ms": 5, "text": "x"}, "invalid source"),
    ],
)
def test_refiner_rejects_zero_duration_and_missing_timestamps(item, match):
    with pytest.raises(TranscriptRefinementError, match=match):
        transcript_refiner.refine_transcript([item])


def test_refiner_chunks_long_transcripts_with_context_and_no_duplicate_ids(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_CHUNK_SEGMENTS", 10)
    items = _segments(*[
        ("Alice", index * 1100, index * 1100 + 900, f"Word {index}.")
        for index in range(25)
    ])
    calls = []

    def fake_chat(payload):
        import json

        data = json.loads(payload)
        calls.append(data)
        return _reply(data["target_segments"])

    monkeypatch.setattr(transcript_refiner, "_post_ollama_chat", fake_chat)

    result = transcript_refiner.refine_transcript(items)

    assert len(calls) == 3
    assert all("read_only_context_segments" in call for call in calls)
    source_ids = [
        segment_id
        for group in result.groups
        for segment_id in group.source_segment_ids
    ]
    assert source_ids == list(range(1, 26))


def test_refiner_rejects_omitted_duplicate_and_invented_words():
    items = _segments(
        ("Alice", 0, 500, "Hello"),
        ("Alice", 600, 1000, "there."),
    )
    responses = [
        {"groups": [{"source_segment_ids": [1], "text": "Hello"}],
         "speaker_change_recommendations": []},
        {"groups": [{"source_segment_ids": [1, 1, 2], "text": "Hello there."}],
         "speaker_change_recommendations": []},
        {"groups": [{"source_segment_ids": [1, 2], "text": "Hello everyone."}],
         "speaker_change_recommendations": []},
    ]

    for response, message in zip(
        responses, ("omitted", "invalid references", "changed spoken words")
    ):
        with pytest.raises(TranscriptRefinementError, match=message):
            transcript_refiner._normalize_and_validate_response(response, items, items)


def test_recommendation_for_non_target_segment_is_dropped():
    items = _segments(("Alice", 0, 500, "Hello."))

    groups, recommendations = transcript_refiner._normalize_and_validate_response(
        _reply(
            items,
            recommendations=[{"after_segment_id": 999, "reason": "Review audio."}],
        ),
        items,
        items,
    )

    assert recommendations == []
    assert [group.source_segment_ids for group in groups] == [(1,)]


@pytest.mark.parametrize(
    "recommendation",
    [
        {"after_segment_id": 1, "reason": ""},
        {"after_segment_id": "1", "reason": "Review audio."},
        {"after_segment_id": True, "reason": "Review audio."},
        "not an object",
    ],
)
def test_malformed_recommendation_is_still_rejected(recommendation):
    items = _segments(("Alice", 0, 500, "Hello."))

    with pytest.raises(TranscriptRefinementError, match="recommendation"):
        transcript_refiner._normalize_and_validate_response(
            _reply(items, recommendations=[recommendation]),
            items,
            items,
        )


def test_local_chunk_validation_failure_uses_deterministic_text_for_that_chunk_only(
    monkeypatch,
):
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_CHUNK_SEGMENTS", 2)
    items = _segments(
        ("Alice", 0, 500, "one two"),
        ("Alice", 600, 900, "three four"),
        ("Bob", 1000, 1400, "five six"),
    )

    def chat(payload):
        targets = json.loads(payload)["target_segments"]
        if targets[0]["segment_id"] == 1:
            return {
                "groups": [{"source_segment_ids": [1, 2], "text": "one two three four extra"}],
                "speaker_change_recommendations": [],
            }
        return {
            "groups": [{"source_segment_ids": [3], "text": "Five six."}],
            "speaker_change_recommendations": [],
        }

    monkeypatch.setattr(transcript_refiner, "_post_ollama_chat", chat)

    result = transcript_refiner.refine_transcript(items)

    assert [group.source_segment_ids for group in result.groups] == [(1, 2), (3,)]
    assert [group.text for group in result.groups] == ["one two three four", "Five six."]
    assert result.provider == "ollama"
    assert result.used_local_fallback is False


def test_local_transport_failure_is_not_absorbed_per_chunk(monkeypatch):
    items = _segments(("Alice", 0, 500, "Hello."))

    def chat(_payload):
        raise TranscriptRefinementError("Local Ollama request failed")

    monkeypatch.setattr(transcript_refiner, "_post_ollama_chat", chat)

    with pytest.raises(TranscriptRefinementError, match="request failed"):
        transcript_refiner.refine_transcript(items)


def test_refiner_requires_no_audio_file_and_does_not_change_words(monkeypatch):
    items = _segments(("UNKNOWN", 0, 800, "Apa khabar?"))
    monkeypatch.setattr(transcript_refiner, "_post_ollama_chat", lambda _payload: _reply(items))

    result = transcript_refiner.refine_transcript(items)

    assert result.groups[0].text == "Apa khabar?"


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


def test_chunks_accept_zero_gap_and_exact_gap_limit(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_MERGE_MAX_GAP_MS", 1500)
    items = _segments(
        ("Alice", 0, 500, "One."),
        ("Alice", 500, 900, "Two."),
        ("Alice", 2400, 2800, "Three."),
    )
    assert transcript_refiner._chunk_ranges(items) == [(0, 3)]


def test_chunks_split_overlapping_same_speaker_segments(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_MERGE_MAX_GAP_MS", 1500)
    items = _segments(
        ("Alice", 0, 800, "We agree"),
        ("Alice", 700, 1200, "today."),
    )
    assert transcript_refiner._chunk_ranges(items) == [(0, 1), (1, 2)]


def test_chunks_respect_segment_cap_within_one_run(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_CHUNK_SEGMENTS", 2)
    items = _segments(*[
        ("Alice", index * 100, index * 100 + 50, f"W{index}.")
        for index in range(5)
    ])
    assert transcript_refiner._chunk_ranges(items) == [(0, 2), (2, 4), (4, 5)]


def test_chunks_respect_character_cap_within_one_run(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_CHUNK_CHARS", 10)
    items = _segments(
        ("Alice", 0, 100, "aaaa"),
        ("Alice", 100, 200, "bbbb"),
        ("Alice", 200, 300, "cccc"),
    )
    assert transcript_refiner._chunk_ranges(items) == [(0, 2), (2, 3)]


def test_single_oversized_target_forms_its_own_chunk(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_CHUNK_CHARS", 5)
    items = _segments(
        ("Alice", 0, 100, "a"),
        ("Alice", 100, 200, "x" * 20),
        ("Alice", 200, 300, "b"),
    )
    assert transcript_refiner._chunk_ranges(items) == [(0, 1), (1, 2), (2, 3)]


def test_read_only_context_is_sent_but_never_accepted_as_reference(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_CHUNK_SEGMENTS", 1)
    items = _segments(
        ("Alice", 0, 100, "One."),
        ("Alice", 200, 300, "Two."),
        ("Alice", 400, 500, "Three."),
    )
    calls = []

    def fake_chat(payload):
        data = json.loads(payload)
        calls.append(data)
        return _reply(data["target_segments"])

    monkeypatch.setattr(transcript_refiner, "_post_ollama_chat", fake_chat)

    transcript_refiner.refine_transcript(items)

    middle = calls[1]
    assert middle["target_segment_ids"] == [2]
    assert [segment["segment_id"] for segment in middle["read_only_context_segments"]] == [1, 3]
    with pytest.raises(TranscriptRefinementError, match="non-target"):
        transcript_refiner._normalize_and_validate_response(
            {"groups": [{"source_segment_ids": [1], "text": "One."}],
             "speaker_change_recommendations": []},
            [items[1]],
            items,
        )


def test_ollama_request_uses_shared_prompt_and_schema(monkeypatch):
    sent = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, size=-1):
            envelope = {"message": {"content": json.dumps(
                {"groups": [], "speaker_change_recommendations": []}
            )}}
            body = json.dumps(envelope).encode("utf-8")
            return body if size is None or size < 0 else body[:size]

    def fake_urlopen(request, timeout):
        sent["body"] = json.loads(request.data.decode("utf-8"))
        sent["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(transcript_refiner, "urlopen", fake_urlopen)

    transcript_refiner._post_ollama_chat("{}")

    assert sent["body"]["messages"][0] == {
        "role": "system",
        "content": transcript_refiner._SYSTEM_PROMPT,
    }
    assert sent["body"]["format"] == transcript_refiner._OUTPUT_SCHEMA
    assert sent["timeout"] == config.TRANSCRIPT_REFINEMENT_TIMEOUT_SECONDS
    assert f"{config.TRANSCRIPT_MERGE_MAX_GAP_MS} ms" in transcript_refiner._SYSTEM_PROMPT
    assert "untrusted" in transcript_refiner._SYSTEM_PROMPT.lower()
    assert "speaker_change_recommendations" in transcript_refiner._SYSTEM_PROMPT


# ---- Task 3: Foundry transport, whole-transcript fallback, provenance ----

import email.message
import io
import urllib.error
import urllib.request
import urllib.response

_FOUNDRY_URL = "https://foundry-host.invalid/anthropic/v1/messages"
_KEY = "sk-test-KEY-SENTINEL"
_BODY_SENTINEL = "PROVIDER-BODY-SENTINEL"
_ERROR_SENTINEL = "TRANSPORT-ERROR-SENTINEL"


@pytest.fixture(autouse=True)
def _isolated_refiner_config(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_USE_CLOUD", False)
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_API", "ollama")
    monkeypatch.setattr(config, "ANTHROPIC_API_URL", _FOUNDRY_URL)
    monkeypatch.setattr(config, "API_KEY", _KEY)
    monkeypatch.setattr(config, "ANTHROPIC_VERSION", "2023-06-01")
    monkeypatch.setattr(config, "MODEL", "test-deployment")


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size=-1):
        return self._body if size is None or size < 0 else self._body[:size]


class _FakeOpener:
    def __init__(self, outcome):
        self.outcome = outcome
        self.requests = []
        self.timeouts = []

    def open(self, request, timeout=None):
        self.requests.append(request)
        self.timeouts.append(timeout)
        outcome = self.outcome(request) if callable(self.outcome) else self.outcome
        if isinstance(outcome, BaseException):
            raise outcome
        return _FakeResponse(outcome)


def _envelope(text, stop_reason="end_turn", blocks=None):
    content = blocks if blocks is not None else [{"type": "text", "text": text}]
    return json.dumps({"stop_reason": stop_reason, "content": content}).encode("utf-8")


def _http_error(code):
    return urllib.error.HTTPError(
        _FOUNDRY_URL, code, _ERROR_SENTINEL, {}, io.BytesIO(_BODY_SENTINEL.encode())
    )


_GOOD_REPLY_TEXT = json.dumps(
    {"groups": [{"source_segment_ids": [1], "text": "Hello."}],
     "speaker_change_recommendations": []}
)

_FOUNDRY_FAILURES = {
    "http-401": _http_error(401),
    "http-429": _http_error(429),
    "http-500": _http_error(500),
    "timeout": TimeoutError(_ERROR_SENTINEL),
    "url-timeout": urllib.error.URLError(TimeoutError(_ERROR_SENTINEL)),
    "url-error": urllib.error.URLError(f"{_FOUNDRY_URL} {_ERROR_SENTINEL}"),
    "os-error": OSError(f"{_KEY} {_ERROR_SENTINEL}"),
    "malformed-envelope": b"not json " + _BODY_SENTINEL.encode(),
    "non-utf8": b"\xff\xfe",
    "nonobject-envelope": b"[]",
    "malformed-text": _envelope(f"not json {_BODY_SENTINEL}"),
    "nonobject-text": _envelope("[]"),
    "explanation-prefix": _envelope(f"Here you go: {_GOOD_REPLY_TEXT}"),
    "truncated": _envelope(_GOOD_REPLY_TEXT, stop_reason="max_tokens"),
    "tool-use-only": _envelope("", blocks=[{"type": "tool_use", "id": "t", "name": "n", "input": {}}]),
    "tool-use-with-text": _envelope("", blocks=[
        {"type": "text", "text": _GOOD_REPLY_TEXT},
        {"type": "tool_use", "id": "t", "name": "n", "input": {}},
    ]),
    "no-content": json.dumps({"stop_reason": "end_turn"}).encode(),
    "non-string-text": _envelope("", blocks=[{"type": "text", "text": 5}]),
    "bad-structure": _envelope(json.dumps({"groups": "x", "speaker_change_recommendations": []})),
    "changed-words": _envelope(json.dumps(
        {"groups": [{"source_segment_ids": [1], "text": "Goodbye."}],
         "speaker_change_recommendations": []}
    )),
    "truncated-groups": _envelope(json.dumps(
        {"groups": [], "speaker_change_recommendations": []}
    )),
    "deep-nesting-envelope": b"[" * 200000,
    "deep-nesting-text": _envelope("[" * 200000),
    "oversized-int-text": _envelope("9" * 5000),
}


def _assert_sanitized(error):
    text = str(error)
    for secret in (_KEY, _BODY_SENTINEL, _ERROR_SENTINEL, "foundry-host", "http://", "https://"):
        assert secret not in text
    assert error.__cause__ is None
    assert error.__suppress_context__ or error.__context__ is None


def test_cloud_failure_restarts_every_chunk_locally(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_USE_CLOUD", True)
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_CHUNK_SEGMENTS", 1)
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
    assert result.fallback_reason == "Foundry request timed out"
    assert result.speaker_change_recommendations == ()
    assert [group.source_segment_ids for group in result.groups] == [(1,), (2,)]


def test_cloud_success_never_calls_local(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_USE_CLOUD", True)
    items = _segments(("Alice", 0, 500, "Hello."))
    local_calls = []
    monkeypatch.setattr(
        transcript_refiner,
        "_post_foundry_chat",
        lambda payload: _reply(json.loads(payload)["target_segments"]),
    )
    monkeypatch.setattr(
        transcript_refiner, "_post_ollama_chat", lambda payload: local_calls.append(payload)
    )

    result = transcript_refiner.refine_transcript(items)

    assert local_calls == []
    assert (result.provider, result.model) == ("foundry", "test-deployment")
    assert result.used_local_fallback is False
    assert result.fallback_reason is None


def test_local_only_never_calls_cloud(monkeypatch):
    items = _segments(("Alice", 0, 500, "Hello."))
    cloud_calls = []
    monkeypatch.setattr(
        transcript_refiner, "_post_foundry_chat", lambda payload: cloud_calls.append(payload)
    )
    monkeypatch.setattr(
        transcript_refiner,
        "_post_ollama_chat",
        lambda payload: _reply(json.loads(payload)["target_segments"]),
    )

    result = transcript_refiner.refine_transcript(items)

    assert cloud_calls == []
    assert (result.provider, result.model) == ("ollama", config.TRANSCRIPT_REFINEMENT_MODEL)
    assert result.used_local_fallback is False


def test_result_defaults_preserve_two_argument_construction():
    result = transcript_refiner.RefinementResult((), ())
    assert (result.provider, result.model) == ("ollama", "")
    assert result.used_local_fallback is False
    assert result.fallback_reason is None


def test_empty_input_does_no_io_and_reports_deterministic(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_USE_CLOUD", True)
    for name in ("_post_foundry_chat", "_post_ollama_chat"):
        monkeypatch.setattr(
            transcript_refiner, name,
            lambda _payload: pytest.fail("provider must not be called"),
        )

    result = transcript_refiner.refine_transcript([])

    assert result.groups == () and result.speaker_change_recommendations == ()
    assert result.provider == "deterministic"
    assert result.model == ""
    assert result.used_local_fallback is False


def test_invalid_source_data_never_calls_either_provider(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_USE_CLOUD", True)
    for name in ("_post_foundry_chat", "_post_ollama_chat"):
        monkeypatch.setattr(
            transcript_refiner, name,
            lambda _payload: pytest.fail("provider must not be called"),
        )
    bad = [{"segment_id": 1, "speaker_label": "A", "start_ms": 5, "end_ms": 5, "text": "x"}]

    with pytest.raises(TranscriptRefinementError, match="invalid source"):
        transcript_refiner.refine_transcript(bad)


def test_programming_error_in_cloud_is_not_swallowed(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_USE_CLOUD", True)
    items = _segments(("Alice", 0, 500, "Hello."))
    local_calls = []

    def cloud(_payload):
        raise KeyError("programming error")

    monkeypatch.setattr(transcript_refiner, "_post_foundry_chat", cloud)
    monkeypatch.setattr(
        transcript_refiner, "_post_ollama_chat", lambda payload: local_calls.append(payload)
    )

    with pytest.raises(KeyError):
        transcript_refiner.refine_transcript(items)
    assert local_calls == []


def test_both_providers_failing_raises_sanitized_error(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_USE_CLOUD", True)
    items = _segments(("Alice", 0, 500, "Hello."))

    def fail_foundry(_payload):
        raise TranscriptRefinementError("Foundry returned an invalid structured response")

    def fail_local(_payload):
        raise TranscriptRefinementError("Local Ollama returned an invalid structured response")

    monkeypatch.setattr(transcript_refiner, "_post_foundry_chat", fail_foundry)
    monkeypatch.setattr(transcript_refiner, "_post_ollama_chat", fail_local)

    with pytest.raises(TranscriptRefinementError) as caught:
        transcript_refiner.refine_transcript(items)

    assert str(caught.value) == (
        "Foundry refinement failed (Foundry returned an invalid structured response); "
        "local Ollama refinement failed (Local Ollama returned an invalid structured response)"
    )
    _assert_sanitized(caught.value)


def test_foundry_request_contract(monkeypatch):
    opener = _FakeOpener(_envelope(_GOOD_REPLY_TEXT))
    monkeypatch.setattr(transcript_refiner, "_CLOUD_OPENER", opener)

    result = transcript_refiner._post_foundry_chat('{"target_segments": []}')

    assert result["groups"][0]["text"] == "Hello."
    (request,) = opener.requests
    assert request.full_url == _FOUNDRY_URL
    assert request.get_method() == "POST"
    headers = {key.lower(): value for key, value in request.header_items()}
    assert headers["x-api-key"] == _KEY
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["content-type"] == "application/json"
    body = json.loads(request.data.decode("utf-8"))
    assert body == {
        "model": "test-deployment",
        "max_tokens": 8192,
        "stream": False,
        "system": transcript_refiner._SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": '{"target_segments": []}'}],
    }
    assert "format" not in body and "options" not in body
    assert opener.timeouts == [config.TRANSCRIPT_REFINEMENT_TIMEOUT_SECONDS]


def test_foundry_joins_multiple_text_blocks(monkeypatch):
    first, second = _GOOD_REPLY_TEXT[:20], _GOOD_REPLY_TEXT[20:]
    opener = _FakeOpener(_envelope("", blocks=[
        {"type": "text", "text": first}, {"type": "text", "text": second},
    ]))
    monkeypatch.setattr(transcript_refiner, "_CLOUD_OPENER", opener)

    assert transcript_refiner._post_foundry_chat("{}")["groups"][0]["text"] == "Hello."


def test_foundry_accepts_one_fenced_json_object(monkeypatch):
    opener = _FakeOpener(_envelope(f"```json\n{_GOOD_REPLY_TEXT}\n```"))
    monkeypatch.setattr(transcript_refiner, "_CLOUD_OPENER", opener)

    result = transcript_refiner._post_foundry_chat("{}")

    assert result["groups"][0]["text"] == "Hello."


_PROVIDER_FAILURES = [
    name for name in _FOUNDRY_FAILURES
    if name not in {"bad-structure", "changed-words", "truncated-groups"}
]


@pytest.mark.parametrize("name", _PROVIDER_FAILURES)
def test_foundry_transport_failures_are_sanitized(monkeypatch, name):
    monkeypatch.setattr(transcript_refiner, "_CLOUD_OPENER", _FakeOpener(_FOUNDRY_FAILURES[name]))

    with pytest.raises(TranscriptRefinementError) as caught:
        transcript_refiner._post_foundry_chat("{}")

    _assert_sanitized(caught.value)


@pytest.mark.parametrize("name", list(_FOUNDRY_FAILURES))
def test_every_foundry_failure_falls_back_to_local(monkeypatch, name):
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_USE_CLOUD", True)
    monkeypatch.setattr(transcript_refiner, "_CLOUD_OPENER", _FakeOpener(_FOUNDRY_FAILURES[name]))
    monkeypatch.setattr(
        transcript_refiner,
        "_post_ollama_chat",
        lambda payload: _reply(json.loads(payload)["target_segments"]),
    )
    items = _segments(("Alice", 0, 500, "Hello."))

    result = transcript_refiner.refine_transcript(items)

    assert result.provider == "ollama"
    assert result.used_local_fallback is True
    assert result.fallback_reason
    for secret in (_KEY, _BODY_SENTINEL, _ERROR_SENTINEL, "foundry-host"):
        assert secret not in result.fallback_reason


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("ANTHROPIC_API_URL", ""),
        ("API_KEY", ""),
        ("ANTHROPIC_VERSION", ""),
        ("MODEL", ""),
        ("ANTHROPIC_API_URL", "http://foundry-host.invalid/anthropic/v1/messages"),
        ("ANTHROPIC_API_URL", "https://user:pw@foundry-host.invalid/anthropic/v1/messages"),
        ("ANTHROPIC_API_URL", "https://foundry-host.invalid/anthropic/v1/messages?key=abc"),
        ("ANTHROPIC_API_URL", "https://foundry-host.invalid/anthropic/v1/messages#frag"),
        ("ANTHROPIC_API_URL", "https:///anthropic/v1/messages"),
        ("ANTHROPIC_API_URL", "https://foundry-host.invalid/other/path"),
        ("ANTHROPIC_API_URL", "https://foundry-host.invalid:bad/anthropic/v1/messages"),
        ("API_KEY", "key\r\nInjected: header"),
        ("ANTHROPIC_VERSION", "2023-06-01\n"),
    ],
)
def test_invalid_foundry_configuration_is_rejected_before_any_request(
    monkeypatch, attribute, value
):
    monkeypatch.setattr(config, attribute, value)
    opener = _FakeOpener(_envelope(_GOOD_REPLY_TEXT))
    monkeypatch.setattr(transcript_refiner, "_CLOUD_OPENER", opener)

    with pytest.raises(TranscriptRefinementError) as caught:
        transcript_refiner._post_foundry_chat("{}")

    assert opener.requests == []
    assert "foundry-host" not in str(caught.value)
    assert "pw" not in str(caught.value)
    _assert_sanitized(caught.value)


# ---- Fix round: parser errors, bounded provider reads ----


def _fail_nth_json_loads(monkeypatch, call_number, error_type):
    real_loads = json.loads
    count = {"n": 0}

    def failing_loads(*args, **kwargs):
        count["n"] += 1
        if count["n"] == call_number:
            raise error_type("injected parser failure")
        return real_loads(*args, **kwargs)

    monkeypatch.setattr(json, "loads", failing_loads)


@pytest.mark.parametrize("error_type", [ValueError, RecursionError])
@pytest.mark.parametrize("call_number", [1, 2])
def test_ollama_parser_errors_are_sanitized(monkeypatch, call_number, error_type):
    body = json.dumps({"message": {"content": _GOOD_REPLY_TEXT}}).encode("utf-8")
    monkeypatch.setattr(
        transcript_refiner, "urlopen", lambda request, timeout: _FakeResponse(body)
    )
    _fail_nth_json_loads(monkeypatch, call_number, error_type)

    with pytest.raises(TranscriptRefinementError) as caught:
        transcript_refiner._post_ollama_chat("{}")

    _assert_sanitized(caught.value)
    assert "injected" not in str(caught.value)


@pytest.mark.parametrize("error_type", [ValueError, RecursionError])
@pytest.mark.parametrize("call_number", [1, 2])
def test_foundry_parser_errors_are_sanitized(monkeypatch, call_number, error_type):
    monkeypatch.setattr(
        transcript_refiner, "_CLOUD_OPENER", _FakeOpener(_envelope(_GOOD_REPLY_TEXT))
    )
    _fail_nth_json_loads(monkeypatch, call_number, error_type)

    with pytest.raises(TranscriptRefinementError) as caught:
        transcript_refiner._post_foundry_chat("{}")

    _assert_sanitized(caught.value)
    assert "injected" not in str(caught.value)


@pytest.mark.parametrize(
    "body",
    [
        b"[" * 200000,
        json.dumps({"message": {"content": "[" * 200000}}).encode("utf-8"),
    ],
    ids=["envelope", "content"],
)
def test_ollama_deep_nesting_is_sanitized(monkeypatch, body):
    monkeypatch.setattr(
        transcript_refiner, "urlopen", lambda request, timeout: _FakeResponse(body)
    )

    with pytest.raises(TranscriptRefinementError) as caught:
        transcript_refiner._post_ollama_chat("{}")

    _assert_sanitized(caught.value)


def test_response_size_limit_constant_is_four_mib():
    assert transcript_refiner._MAX_RESPONSE_BYTES == 4 * 1024 * 1024


def test_ollama_response_size_boundary(monkeypatch):
    body = json.dumps({"message": {"content": json.dumps(
        {"groups": [], "speaker_change_recommendations": []}
    )}}).encode("utf-8")
    monkeypatch.setattr(
        transcript_refiner, "urlopen", lambda request, timeout: _FakeResponse(body)
    )

    monkeypatch.setattr(transcript_refiner, "_MAX_RESPONSE_BYTES", len(body))
    assert transcript_refiner._post_ollama_chat("{}") == {
        "groups": [], "speaker_change_recommendations": []
    }

    monkeypatch.setattr(transcript_refiner, "_MAX_RESPONSE_BYTES", len(body) - 1)
    with pytest.raises(TranscriptRefinementError, match="size limit") as caught:
        transcript_refiner._post_ollama_chat("{}")
    _assert_sanitized(caught.value)


def test_ollama_oversized_response_is_rejected_without_echoing_body(monkeypatch):
    monkeypatch.setattr(transcript_refiner, "_MAX_RESPONSE_BYTES", 64)
    body = json.dumps({"message": {"content": _BODY_SENTINEL + "x" * 200}}).encode("utf-8")
    monkeypatch.setattr(
        transcript_refiner, "urlopen", lambda request, timeout: _FakeResponse(body)
    )

    with pytest.raises(TranscriptRefinementError, match="size limit") as caught:
        transcript_refiner._post_ollama_chat("{}")

    _assert_sanitized(caught.value)


def test_foundry_response_size_boundary(monkeypatch):
    body = _envelope(_GOOD_REPLY_TEXT)
    monkeypatch.setattr(transcript_refiner, "_MAX_RESPONSE_BYTES", len(body))
    monkeypatch.setattr(
        transcript_refiner, "_CLOUD_OPENER", _FakeOpener(body)
    )
    assert transcript_refiner._post_foundry_chat("{}")["groups"][0]["text"] == "Hello."

    monkeypatch.setattr(transcript_refiner, "_MAX_RESPONSE_BYTES", len(body) - 1)
    monkeypatch.setattr(
        transcript_refiner, "_CLOUD_OPENER", _FakeOpener(body)
    )
    with pytest.raises(TranscriptRefinementError, match="size limit") as caught:
        transcript_refiner._post_foundry_chat("{}")
    _assert_sanitized(caught.value)


def test_foundry_oversized_response_is_rejected_without_echoing_body(monkeypatch):
    monkeypatch.setattr(transcript_refiner, "_MAX_RESPONSE_BYTES", 64)
    monkeypatch.setattr(
        transcript_refiner,
        "_CLOUD_OPENER",
        _FakeOpener(_envelope(_BODY_SENTINEL + "x" * 200)),
    )

    with pytest.raises(TranscriptRefinementError, match="size limit") as caught:
        transcript_refiner._post_foundry_chat("{}")

    _assert_sanitized(caught.value)


def test_foundry_parser_failure_on_later_chunk_restarts_whole_transcript_locally(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_USE_CLOUD", True)
    monkeypatch.setattr(config, "TRANSCRIPT_REFINEMENT_CHUNK_SEGMENTS", 1)
    items = _segments(
        ("Alice", 0, 100, "One."),
        ("Alice", 200, 300, "Two."),
        ("Alice", 400, 500, "Three."),
    )
    cloud_chunks = []
    local_chunks = []

    def cloud_reply(request):
        user = json.loads(json.loads(request.data.decode("utf-8"))["messages"][0]["content"])
        ids = user["target_segment_ids"]
        cloud_chunks.append(ids)
        if ids == [2]:
            return _envelope("[" * 200000)
        return _envelope(json.dumps({
            "groups": [{"source_segment_ids": ids, "text": items[ids[0] - 1]["text"]}],
            "speaker_change_recommendations": [
                {"after_segment_id": 1, "reason": "Cloud-only recommendation"}
            ],
        }))

    def local(payload):
        targets = json.loads(payload)["target_segments"]
        local_chunks.append([item["segment_id"] for item in targets])
        return _reply(targets)

    monkeypatch.setattr(transcript_refiner, "_CLOUD_OPENER", _FakeOpener(cloud_reply))
    monkeypatch.setattr(transcript_refiner, "_post_ollama_chat", local)

    result = transcript_refiner.refine_transcript(items)

    assert cloud_chunks == [[1], [2]]
    assert local_chunks == [[1], [2], [3]]
    assert result.provider == "ollama"
    assert result.used_local_fallback is True
    assert result.fallback_reason == "Foundry returned an invalid structured response"
    assert result.speaker_change_recommendations == ()
    assert [group.source_segment_ids for group in result.groups] == [(1,), (2,), (3,)]


def test_cloud_redirects_are_rejected_and_never_followed():
    class RedirectingHTTPSHandler(urllib.request.HTTPSHandler):
        def __init__(self):
            super().__init__()
            self.requests = []

        def https_open(self, request):
            self.requests.append(request)
            headers = email.message.Message()
            headers["Location"] = "https://second-destination.invalid/steal"
            response = urllib.response.addinfourl(
                io.BytesIO(b""), headers, request.full_url, 302
            )
            response.msg = "Found"
            return response

    handler = RedirectingHTTPSHandler()
    opener = transcript_refiner._build_cloud_opener(
        handler, urllib.request.ProxyHandler({})
    )
    original = transcript_refiner._CLOUD_OPENER
    transcript_refiner._CLOUD_OPENER = opener
    try:
        with pytest.raises(TranscriptRefinementError) as caught:
            transcript_refiner._post_foundry_chat("{}")
    finally:
        transcript_refiner._CLOUD_OPENER = original

    assert [request.full_url for request in handler.requests] == [_FOUNDRY_URL]
    assert "second-destination" not in str(caught.value)
    _assert_sanitized(caught.value)


def test_default_cloud_opener_declines_redirects():
    handler = transcript_refiner._NoRedirectHandler()
    request = urllib.request.Request(_FOUNDRY_URL, data=b"{}", method="POST")

    assert handler.redirect_request(
        request, io.BytesIO(b""), 307, "Temporary Redirect", {}, "https://other.invalid/"
    ) is None


@pytest.mark.parametrize(
    "outcome",
    [
        urllib.error.URLError(f"{config.TRANSCRIPT_REFINEMENT_URL} {_ERROR_SENTINEL}"),
        _http_error(500),
        TimeoutError(_ERROR_SENTINEL),
        OSError(_ERROR_SENTINEL),
        b"not json " + _BODY_SENTINEL.encode(),
        json.dumps({"message": {"content": f"not json {_BODY_SENTINEL}"}}).encode(),
    ],
)
def test_local_ollama_errors_are_sanitized(monkeypatch, outcome):
    def fake_urlopen(_request, timeout):
        return _FakeOpener(outcome).open(_request, timeout)

    monkeypatch.setattr(transcript_refiner, "urlopen", fake_urlopen)

    with pytest.raises(TranscriptRefinementError) as caught:
        transcript_refiner._post_ollama_chat("{}")

    assert config.TRANSCRIPT_REFINEMENT_URL not in str(caught.value)
    assert "localhost" not in str(caught.value)
    _assert_sanitized(caught.value)


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