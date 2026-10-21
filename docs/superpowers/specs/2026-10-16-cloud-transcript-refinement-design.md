# Cloud Transcript Refinement With Local Fallback

## Approved Direction

Support the user's Azure AI Foundry deployment through the Anthropic Messages
API, alongside the existing local Ollama refiner. Select cloud or local with a
boolean configuration setting. If any cloud request or output validation fails,
discard all partial cloud output and refine the entire transcript locally.
Display the provider and configured model that actually produced the saved
readable transcript. Never weaken source-text or speaker-boundary validation.

## Scope And Non-Goals

- Extend existing configuration, transcript refinement, persistence, and transcript
  UI boundaries rather than introduce a general-purpose model framework.
- Preserve detailed Whisper text, corrected detailed text, timestamps, and speaker
  identities. Refinement formats text; it does not correct spoken words.
- Preserve deterministic readable output when refinement is disabled or all model
  attempts fail.
- Do not change recognition, attendance, enrollment, database schema, or tab layout.
- Do not submit real meeting transcripts to cloud services during implementation
  or tests. Do not run runtime migration against live data.
- Preserve pre-existing working-tree changes, including `.gitignore` and
  `app/main.py`. Never stage or commit `.env`.

## Configuration

Load the repository-root `.env` explicitly before environment-backed settings
are evaluated, independent of the terminal's working directory. Use
`python-dotenv` and declare it in both supported requirements files. Existing
process environment variables take precedence; missing `.env` is allowed.
Loading configuration must not initialize runtime paths, migrate data, or create
database connections.

| Setting | Meaning |
|---|---|
| `TRANSCRIPT_REFINEMENT_ENABLED` | Existing master switch; default false. |
| `TRANSCRIPT_REFINEMENT_USE_CLOUD` | New boolean; default false. True selects cloud first with automatic local fallback; false uses local only. |
| `ANTHROPIC_API_URL` | Full configured Foundry Messages endpoint ending in `/anthropic/v1/messages`; do not append a second Messages path. |
| `API_KEY` | Existing cloud API-key variable; required only for a cloud attempt. |
| `ANTHROPIC_VERSION` | Existing Anthropic API version passed with the cloud request. |
| `MODEL` | Existing cloud deployment/model identifier sent unchanged. Do not substitute or assume that the user-supplied `claude-haiku-5-5` is a public model identifier. |
| `TRANSCRIPT_REFINEMENT_URL` | Existing Ollama base URL; preserve its current default. |
| `TRANSCRIPT_REFINEMENT_MODEL` | Existing Ollama model; preserve its current default. |

Use the existing boolean parsing convention. Cloud mode explicitly authorizes
transmitting transcript text to the configured cloud endpoint. Local-only mode
must not contact cloud services or require cloud credentials. Missing or invalid
cloud configuration triggers local fallback rather than app startup failure.
Keep existing request timeout and chunk size settings for this change.

## Provider Requests

Keep provider-specific HTTP serialization and response parsing separate from
shared chunking and validation, within the existing refiner module where practical.
Both providers receive the same formatting rules and source references.

For Foundry, send a non-streaming Anthropic Messages request with the configured
model, API version, finite output-token limit, system instructions, and user
content. Use Foundry's documented API-key authentication. Confirm the precise
header contract against official Foundry documentation before implementation;
do not assume the endpoint is OpenAI-compatible. Require HTTPS for cloud
credentials and do not forward them through redirects to a different endpoint.

Request the existing groups/recommendations JSON structure in the instructions
without assuming that Foundry accepts Ollama's `format` schema parameter.
Accept a completed Anthropic response containing text blocks that parse as the
required JSON object; reject truncated, tool-use-only, malformed, or structurally
invalid responses. Retain the existing Ollama HTTP and schema contract.

No transcript bodies, credentials, authorization headers, or raw provider error
bodies may appear in logs, UI errors, or persisted fallback explanations. Errors
identify the provider and a sanitized category, such as configuration, timeout,
HTTP status, malformed output, or validation failure.

## Boundary-Safe Requests And Validation

Before applying existing segment-count and character limits, split targets into
runs at every speaker-label change, overlapping adjacent timestamp pair, or gap
greater than `TRANSCRIPT_MERGE_MAX_GAP_MS`. A gap of zero or exactly the configured
maximum is allowed. Retain bounded read-only neighboring context, clearly separate
from target IDs. Model output may reference only target IDs.

State these same timing and speaker rules in shared instructions. Keep all
existing validation: chronological consecutive references, exact target coverage,
unchanged word sequence, identical speaker labels within a group, no overlaps,
and bounded pauses. A provider can still violate these rules, and rejection
remains mandatory.

## Whole-Transcript Fallback

1. Validate and normalize source segments once. Invalid source data is not a
   provider failure and must not trigger duplicate network work.
2. With cloud mode enabled, attempt all chunks using Foundry.
3. On a cloud configuration, transport, response, or validation failure, discard
   every cloud group and recommendation, then restart all chunks with Ollama.
4. Local-only mode attempts Ollama directly, with no cloud requests.
5. If the local attempt fails, retain the existing deterministic readable output
   and record a sanitized failure summary.

Do not retry cloud automatically or combine cloud/local chunks. Empty transcripts
perform no network requests and must not claim successful model refinement.

## Attribution And UI

Extend the refinement result with actual provider, configured model, and whether
local success followed cloud failure. The transcription orchestration must use
this result rather than hard-code `ollama:<model>` from configuration.

Use existing method/status/error fields; no database migration is needed:

- Cloud success: method `foundry:<model>`, status `complete`.
- Direct local success: method `ollama:<model>`, status `complete`.
- Local success after cloud failure: method `ollama:<model>`, status
  `complete_local_fallback`, plus a sanitized cloud failure explanation.
- All attempted providers fail: method `deterministic`, status `fallback`.
- Refinement disabled or explicitly bypassed: retain deterministic behavior and
  existing disabled status.

The UI displays Cloud / Local Ollama / Deterministic, model where applicable,
and an explicit cloud-fallback indication. Successful local fallback must not
be described as deterministic failure. Existing saved `ollama:*` methods and
statuses remain readable. Changing configuration must not relabel historical
results: attribution comes from the saved run.

Replace local-only button, spinner, disabled-caption, and failure wording with
provider-appropriate wording. Keep the existing action and transcript views.
Document root `.env` setup, process-variable precedence, restart requirements,
cloud data disclosure, and fallback behavior in the README using placeholders
only. Do not edit the user's credential values.

## Verification

Use existing test modules and temporary fixtures with mocked HTTP:

- Root `.env` loads from both root and app working directories; process settings
  win; absent `.env` and disabled cloud credentials do not break imports.
- Cloud/local boolean routing, unchanged endpoint/model use, authentication and
  version headers, and appropriate provider response parsing.
- Cloud transport/configuration/output failures trigger a complete local rerun,
  including a later-chunk failure after earlier successful cloud chunks.
- No partial cloud groups or recommendations survive local fallback.
- Both providers retain all source and boundary-validation guarantees; boundary
  runs respect speaker changes, overlap, pause limits, and exact-limit cases.
- Local failure retains deterministic text and detailed source rows unchanged.
- Saved provider/model/fallback attribution matches actual output and remains
  correct when current configuration changes.
- Logs, UI messages, and stored failure explanations do not expose test secrets
  or raw response bodies.

Run focused config/refiner/transcription/UI checks followed by the packaged
`tests/unit tests/integration` suite. Live cloud connectivity and real model
quality remain unverified unless the user separately authorizes a smoke test.