# OpenAI-Compatible Local Transcript Refinement Design

## Goal

Allow the local transcript refiner to use a `llama-server` (llama.cpp) instance
on another machine, such as `http://192.168.137.27:30000` serving
`unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL`, while keeping Ollama as the default.

## Current state

`transcript_refiner._post_ollama_chat` posts to Ollama's native
`{TRANSCRIPT_REFINEMENT_URL}/api/chat`. `llama-server` does not provide that
endpoint; it provides the OpenAI-compatible `/v1/chat/completions` endpoint.
The local provider ID is always `ollama`. That ID is saved in refinement
methods (`ollama:<model>`) and shown as "Local Ollama" in the UI.

## Configuration

Add `TRANSCRIPT_REFINEMENT_API` to `config/settings.py` and export it from
`config/__init__.py`:

- `ollama` (default): current behavior, unchanged.
- `openai`: OpenAI-compatible chat completions.

The value is trimmed and case-insensitive. Any other value raises `ValueError`
when settings load, naming the setting and the allowed values.

`TRANSCRIPT_REFINEMENT_URL`, `TRANSCRIPT_REFINEMENT_MODEL`, and
`TRANSCRIPT_REFINEMENT_TIMEOUT_SECONDS` keep their meanings for both APIs. The
URL is the server root without `/v1`.

## Request client

Add `_post_openai_compatible_chat(user_content: str) -> dict` next to
`_post_ollama_chat`.

- Send `POST {TRANSCRIPT_REFINEMENT_URL}/v1/chat/completions` with
  `Content-Type: application/json` and no authentication header.
- Use this payload:
  `model`, `temperature: 0`, `stream: false`, the existing system and user
  messages, and
  `response_format: {"type": "json_schema", "json_schema": {"name": "transcript_refinement", "schema": _OUTPUT_SCHEMA}}`.
- Read the response with `_read_bounded_body` and the configured timeout.
- Read the content from `choices[0].message.content`. Missing or non-string
  content is an invalid structured response.
- Remove a leading `<think>...</think>` block before calling the existing
  `_parse_structured_response`, which already handles fenced JSON.
- Handle errors the same way as Ollama. HTTP errors report only the status
  code. Transport errors use `_transport_failure_kind`. Decoding and shape
  errors report an invalid structured response. Messages use the provider label
  "Local OpenAI-compatible server" and do not include response bodies or
  transcript text.

## Provider routing and attribution

- Add a helper that returns the configured local provider ID: `ollama` or
  `openai`.
- `_refine_with_provider` routes `openai` to the new client with model
  `TRANSCRIPT_REFINEMENT_MODEL`. Unknown providers still raise `ValueError`.
- Per-chunk validation fallback, which keeps the source text for an invalid
  chunk, applies to both local providers instead of only `ollama`. The warning
  log names the actual provider.
- Local-only runs and cloud-to-local fallback both use the configured local
  provider. The combined fallback error names the local provider label.
- `transcription_core` accepts `openai` as a provider, so the saved method is
  `openai:<model>`.
- `app/pages/transcripts.py`:
  - `_provider_caption("openai:<model>")` returns
    `Local OpenAI-compatible: <model>`.
  - The refine button, spinner, and disabled-state hint use the configured
    local label ("local Ollama" or "local OpenAI-compatible server") instead of
    always saying Ollama.

Historical `ollama:<model>` methods remain unchanged and continue to show as
"Local Ollama".

## Privacy

The default remote example uses unencrypted HTTP on a LAN, so transcript text,
but not audio, crosses the network in clear text. The README's existing rule for
non-local endpoints applies. Documentation must restate that the endpoint must
be trusted.

## Testing

Add unit tests that mock `urlopen`:

1. The OpenAI client sends the expected URL, model, temperature, messages, and
   `response_format` schema.
2. It parses `choices[0].message.content`, including content with a leading
   `<think>` block and fenced JSON.
3. It reports HTTP errors with status only, reports transport failures, and
   rejects malformed envelopes without leaking response bodies.
4. With `TRANSCRIPT_REFINEMENT_API=openai`, local-only refinement returns
   `provider == "openai"` and the configured model.
5. Cloud failure falls back to the OpenAI-compatible local provider.
6. An invalid chunk from the OpenAI-compatible provider falls back to
   deterministic source text for that chunk.
7. `transcription_core` saves `openai:<model>`.
8. The UI caption shows `Local OpenAI-compatible: <model>`.
9. An invalid `TRANSCRIPT_REFINEMENT_API` value is rejected.

Existing Ollama tests remain unchanged and must pass.

## Documentation and local configuration

- README: document `TRANSCRIPT_REFINEMENT_API`, add a `llama-server` example,
  and update the behavior list so it describes local refinement through the
  configured local API.
- The user's `.env`, which is not committed:
  `TRANSCRIPT_REFINEMENT_API=openai`,
  `TRANSCRIPT_REFINEMENT_URL=http://192.168.137.27:30000`,
  `TRANSCRIPT_REFINEMENT_MODEL=unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL`,
  `TRANSCRIPT_REFINEMENT_TIMEOUT_SECONDS=300`.

## Out of scope

- API-key authentication for the local server.
- Automatically detecting the server type.
- Streaming responses.
- Reprocessing historical meetings.
