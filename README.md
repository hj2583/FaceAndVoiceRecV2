# Face and Voice Recognition

A Streamlit application for face recognition, active-speaker estimation, and optional video transcription.

This project is built to run comfortably on CPU by default. If CUDA is available, the app can use it automatically. If not, it falls back to CPU instead of failing at startup.

## Overview

The system combines:

- UniFace SCRFD + ArcFace for face detection and embedding extraction
- UniFace FaceMesh for facial landmarks and mouth-motion measurements
- OpenCV for fallback detection and video handling
- PostgreSQL for relational records
- Qdrant for face and voice vector search
- Silero VAD for speech activity detection
- OpenAI Whisper for optional transcription
- SpeechBrain ECAPA-TDNN for voice embeddings and speaker clustering
- Streamlit for the web UI

## Models and processing flow

No single model handles face recognition, speaker attribution, transcription,
and attendance. The application combines these models and processing stages:

| Component | Model or tool | Purpose |
|---|---|---|
| Face detection | UniFace SCRFD (larger model for uploaded videos, lighter model for realtime) | Detects face locations in video frames. |
| Face recognition | UniFace ArcFace | Converts a face into a 512-dimensional embedding and compares it with enrolled face embeddings using cosine similarity and a confidence threshold. |
| Facial landmarks | UniFace FaceMesh | Locates facial features, including the mouth, to measure mouth opening and motion. |
| Speech detection | Silero VAD | Finds speech regions in audio; it does not generate transcript text. |
| Transcription | OpenAI Whisper (`small`) | Converts extracted audio into text and timestamps. FFmpeg extracts audio from uploaded video. Whisper does not identify speakers. |
| Voice embeddings | SpeechBrain ECAPA-TDNN | Represents voice samples numerically; the project uses embeddings to cluster speech and attempt matching against enrolled voice samples. |
| Active-speaker estimate | Application logic combining VAD timing and FaceMesh lip motion | Estimates which visible face is speaking only when lip movement is measurable and competitive; face-recognition similarity and temporal continuity are not treated as evidence of speaker identity. |
| Identity storage and matching | Relational database and Qdrant | Stores people and enrollment metadata relationally; searches face and voice embeddings in Qdrant. |
| Attendance | Application logic and PostgreSQL | Records a known person when the video or realtime pipeline recognizes their face with sufficient confidence. |
| Optional transcript formatting | Local Ollama model (disabled by default) | Formats and groups existing transcript fragments while preserving the spoken words and speaker labels. It does not replace Whisper or determine speaker identities. |

Face and voice enrollment are independent: enroll face samples for visual
identity recognition, and enroll clean voice samples to support voice matching.
Voice activity or an unknown face alone is not enough to create a known-person
attendance record. Person, enrollment, transcript, attendance, and log records
remain relational data; Qdrant stores vectors and non-sensitive lookup metadata.

### Production storage migration

The production target is PostgreSQL for relational records and transactions,
Qdrant for face and voice embeddings, and shared object storage for images,
audio, and video. Qdrant payloads contain vector metadata and object references,
not media files.

The runtime now reads known and unresolved face/voice vectors from Qdrant for
matching, and new face/voice vectors are indexed there as they are enrolled or
captured. Relational records remain authoritative for person IDs, enrollment
quality, and unresolved/resolved status. Local `.npy` embeddings are retained
for recovery and backfill; images, audio, and video remain files and are not
stored in Qdrant. Source scripts belong in version control, not a database.

PostgreSQL is required for application runtime; SQLite is retained only as the
legacy import source and for isolated tests. Alembic creates the PostgreSQL
schema. Logs, transcripts, attendance, and other structured records remain
relational data; shared object storage for media is not yet implemented. Back
up the SQLite database and its media before importing, then verify the imported
row counts before starting normal application traffic.

Qdrant collections are initialized from environment settings. The report-first
`scripts/backfill_qdrant_embeddings.py` migrates existing known and unresolved
face/voice vectors with deterministic IDs. Run it from the repository root,
review the report, then pass `--apply`. Re-running it safely upserts the same
points. It does not migrate relational records or media files.

Set `DATABASE_URL` in the root `.env` before launching the app. Use a PostgreSQL
URL with the `postgresql+psycopg` driver, for example:

```dotenv
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/face_voice_rec
```

For a new empty database, create its schema with
`python -m alembic upgrade head`. To import the current SQLite records, first
run `python scripts/migrate_sqlite_to_postgres.py` for a report, then
`python scripts/migrate_sqlite_to_postgres.py --apply`. The importer creates the
schema and refuses to write into a non-empty target. It copies relational rows,
not media files. After confirming the imported records, backfill Qdrant vectors
with `python scripts/backfill_qdrant_embeddings.py --apply`. Keep database and
Qdrant credentials out of source control.

Voice identity matching compares ECAPA cosine similarity with quality-filtered
enrollment samples, deduplicates identical vectors, and checks both the best
profile score and the margin to the next person. Defaults require cosine
similarity of at least `VOICE_MATCH_THRESHOLD` (0.50), a lead of at least
`VOICE_AMBIGUITY_MARGIN` (0.05), and two supporting enrollment samples when
two or more independent samples are available. Automatic voice identity is
withheld for clusters shorter than `VOICE_MIN_IDENTITY_DURATION_SECONDS`
(3 seconds). If a visual active-speaker candidate conflicts with a supported
voice match, the result is left unresolved rather than forcing either label.
Cosine similarity and the lip-motion activity score are not calibrated
probabilities; the UI labels them as scores, not confidence percentages.
Optional candidate diagnostics can be enabled with
`VOICE_DIAGNOSTICS_ENABLED=1`; reports contain sample timing and identity-score
summaries but not audio content or full media paths.

### Multilingual transcription

Whisper uses `transcribe` mode so it retains the spoken language rather than
translating to English. The default language policy remains fixed English for
the project's English-dominant collection. For another dominant meeting
language, set `WHISPER_LANGUAGE_MODE=auto` before launching the app; the
pipeline samples multiple VAD speech regions and uses the detected meeting
language. `WHISPER_LANGUAGE` can set a fixed Whisper language code when the
mode is `fixed`. Automatic detection chooses one language for the meeting, so
heavily code-switched recordings may still need transcript review.

### Optional readable transcript refinement

Transcription always retains the detailed Whisper segments. A separate readable
version is built from those segments and can be formatted by an optional LLM
refiner. Refinement is disabled by default; the deterministic readable view
still works without any LLM service.

In the UI, the readable view also presents continuous nearby speech as a
paragraph, even if its detailed source rows have different speaker labels. Such
a paragraph is marked **Mixed/uncertain attribution** and lists every source
speaker label; it does not reassign any segment. Use the Detailed view to
inspect the original labels and timestamps.

The app loads repository-root `.env` values, but existing process environment
variables always take precedence. After changing refinement settings, restart
Streamlit so the updated values are loaded.

Use placeholder values (example only):

```dotenv
TRANSCRIPT_REFINEMENT_ENABLED=true
TRANSCRIPT_REFINEMENT_API=ollama
TRANSCRIPT_REFINEMENT_USE_CLOUD=true
ANTHROPIC_API_URL=https://example.services.ai.azure.com/anthropic/v1/messages
API_KEY=your-foundry-key
ANTHROPIC_VERSION=2023-06-01
MODEL=your-deployment-name
TRANSCRIPT_REFINEMENT_URL=http://localhost:11434
TRANSCRIPT_REFINEMENT_MODEL=qwen2.5:7b
TRANSCRIPT_REFINEMENT_TIMEOUT_SECONDS=90
```

Behavior:

- `TRANSCRIPT_REFINEMENT_ENABLED=false` (default): deterministic readable output only.
- `TRANSCRIPT_REFINEMENT_ENABLED=true` and `TRANSCRIPT_REFINEMENT_USE_CLOUD=false` (default): local refinement only, through the configured local API.
- `TRANSCRIPT_REFINEMENT_ENABLED=true` and `TRANSCRIPT_REFINEMENT_USE_CLOUD=true`: cloud Foundry first, then automatic local fallback through the configured local API, then deterministic fallback if both fail.

If cloud mode is enabled, transcript text (not audio) is sent to the configured
cloud endpoint. Audio files are never uploaded by this feature. Validation is
strict: changed/omitted/reordered/invented words are rejected, and grouping
across speaker changes, overlaps, or long pauses is rejected.

The same privacy rule applies if `TRANSCRIPT_REFINEMENT_URL` points to a
non-local Ollama service: transcript text is sent to that endpoint. Use only an
endpoint you trust, and protect the connection because Ollama's default URL uses
unencrypted HTTP. Keep API keys in `.env` or the process environment; do not
commit or share them. `API_KEY` and `MODEL` are generic environment-variable
names, so existing process values override `.env` and must identify the intended
Foundry key and deployment. Foundry requires an HTTPS endpoint ending in
`/anthropic/v1/messages` and the deployed model's exact deployment name.

The app persists real refinement attribution for each run (provider and model)
so historical labels remain accurate even after configuration changes.
Changing settings does not automatically reprocess old meetings; use the
refinement action in Meeting Transcripts to regenerate a meeting's readable
view.

To enable local refinement, install and start Ollama, download a model, then
start Streamlit from the repository root or from `app/`:

```powershell
ollama pull qwen2.5:7b
$env:TRANSCRIPT_REFINEMENT_ENABLED = "1"
$env:TRANSCRIPT_REFINEMENT_URL = "http://localhost:11434"
$env:TRANSCRIPT_REFINEMENT_MODEL = "qwen2.5:7b"
streamlit run app/main.py
# or from the app directory:
# streamlit run main.py
```

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

Speaker-change suggestions are advisory and never change attribution. Failures
leave the detailed Whisper transcript intact and show a deterministic readable
fallback. The app stores the original Whisper text, manually corrected detailed
text, and readable output separately so they can be compared.

### Participant self-introductions

After video transcription, the app looks for explicit self-introduction
phrases in English, Malay, Spanish, and French, and saves the extracted name,
optional role, timestamps, source segment IDs, diarized voice label, and
time-overlapping active face track in the `meeting_introductions` table. A
clear introduction on a stable unknown voice track can create/reuse the named
person and enroll that track's existing voice samples. If the face is not
already recognized, a uniquely aligned active face track is enrolled from its
saved face samples and linked to that person. Face tracks are assigned a
per-processing-run label so samples from different videos are not accidentally
treated as the same face. If there is no unique saved face track, the
introduction remains pending for review. Conflicting audio/face identities and
multiple introductions in one segment are not auto-assigned; pending candidates
can be confirmed in Meeting Transcripts. Introduction/parser and lip-activity
scores are heuristic scores, not calibrated probabilities. No face identity is
inferred just because a face is visible.

## Features

- Realtime webcam recognition
- Uploaded-video processing
- Active-speaker estimation from voice + lip movement
- Participant name and role extraction from explicit meeting introductions
- Unknown-face tracking and review
- Face database management
- Transcription support for video/audio content

## Requirements

- Python 3.10+ 64-bit
- Python 3.14 is the current CI-tested interpreter; the declared minimum remains Python 3.10.
- FFmpeg installed and available on PATH
- A webcam for realtime mode
- A microphone for live audio/VAD analysis

## CPU setup

1. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
```

2. Install the CPU requirements:

```powershell
pip install -r requirements.txt
```

3. Start Qdrant locally (Docker Desktop required) and verify the service:

```powershell
docker run -d --name face-voice-qdrant -p 6333:6333 -v face_voice_qdrant:/qdrant/storage qdrant/qdrant
Invoke-RestMethod http://localhost:6333/
```

For another Qdrant server, set `QDRANT_URL` and, if required, `QDRANT_API_KEY`
in the root `.env` or process environment. The default URL is
`http://localhost:6333`; the application does not start Qdrant itself.

4. Start PostgreSQL locally and configure the root `.env`:

```powershell
docker run -d --name face-voice-postgres `
    -e POSTGRES_USER=face_voice_rec `
    -e POSTGRES_PASSWORD=change-this-password `
    -e POSTGRES_DB=face_voice_rec `
    -p 5432:5432 `
    -v face_voice_postgres:/var/lib/postgresql/data postgres:17
```

Put the matching connection URL in `.env` (use a private password):

```dotenv
DATABASE_URL=postgresql+psycopg://face_voice_rec:change-this-password@localhost:5432/face_voice_rec
```

5. Migrate relational data. If importing the existing SQLite database, run a
report first, then apply it to the empty PostgreSQL database:

```powershell
$env:PYTHONPATH = 'src'
python scripts/migrate_sqlite_to_postgres.py
python scripts/migrate_sqlite_to_postgres.py --apply
```

For a new project with no SQLite data to import, create only the schema:

```powershell
python -m alembic upgrade head
```

6. Backfill existing face and voice vectors into Qdrant. Report first, then apply:

```powershell
python scripts/backfill_qdrant_embeddings.py
python scripts/backfill_qdrant_embeddings.py --apply
```

7. Verify FFmpeg is installed:

```powershell
ffmpeg -version
```

8. Launch the app:

```powershell
streamlit run app/main.py
# or from app/:
# streamlit run main.py
```

## Optional GPU setup

If your machine has NVIDIA CUDA support and you want to use the GPU-accelerated path, install the GPU variant instead:

```powershell
pip install -r requirements-gpu.txt
```

The project prefers CUDA automatically when it is available. If CUDA is missing or unavailable, it falls back to CPU mode. GPU acceleration covers both face recognition (ONNX Runtime) and the torch-based audio models (Whisper, Silero VAD, SpeechBrain voice embeddings) — `requirements-gpu.txt` pulls a CUDA-enabled build of `torch` from PyTorch's own package index for this reason. If you already have a CPU-only `torch` installed from `requirements.txt`, re-run `pip install -r requirements-gpu.txt` to replace it with the CUDA build.

## Project structure

```text
FaceAndVoiceRecV2/
├── app/
│   ├── __init__.py
│   ├── main.py                  # streamlit app entry
│   ├── pages/
│   │   ├── realtime.py
│   │   ├── video.py
│   │   ├── transcripts.py
│   │   └── voice_enrollment.py
│   └── ui/
│       ├── attendance_view.py
│       ├── renderers.py
│       └── status.py
│
├── src/
│   └── face_voice_rec/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── settings.py
│       │   ├── paths.py
│       │   └── constants.py
│       │
│       ├── domain/
│       │   ├── face/
│       │   │   ├── __init__.py
│       │   │   ├── face_core.py
│       │   │   ├── face_backend.py
│       │   │   ├── detection_core.py
│       │   │   └── tracking.py
│       │   │
│       │   ├── voice/
│       │   │   ├── __init__.py
│       │   │   ├── voice_core.py
│       │   │   ├── audio_core.py
│       │   │   └── speaker_attribution.py
│       │   │
│       │   ├── transcription/
│       │   │   ├── __init__.py
│       │   │   ├── transcription_core.py
│       │   │   ├── transcript_refiner.py
│       │   │   ├── subtitle_renderer.py
│       │   │   └── speaker_introductions.py
│       │   │
│       │   └── attendance/
│       │       ├── __init__.py
│       │       └── attendance_service.py
│       │
│       ├── pipelines/
│       │   ├── __init__.py
│       │   ├── realtime/
│       │   │   ├── __init__.py
│       │   │   ├── realtime.py
│       │   │   ├── realtime_launcher.py
│       │   │   └── realtime_runtime.py
│       │   ├── video/
│       │   │   ├── __init__.py
│       │   │   ├── video_workflow.py
│       │   │   ├── video_processor.py
│       │   │   └── ffmpeg_utils.py
│       │   └── workflow/
│       │       └── pipeline_orchestration.py
│       │
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── database.py
│       │   ├── sqlite_utils.py
│       │   └── repositories.py
│       │
│       ├── services/
│       │   ├── __init__.py
│       │   ├── identity_service.py
│       │   ├── recognition_service.py
│       │   ├── speaker_service.py
│       │   └── meeting_service.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── time_utils.py
│           ├── text_processing.py
│           └── file_utils.py
│
├── data/
│   ├── db/
│   │   └── faces.db
│   ├── artifacts/
│   │   ├── face_index.npz
│   │   ├── known_faces/
│   │   ├── unknown_faces/
│   │   ├── unknown_voices/
│   │   ├── trackedVideo/
│   │   ├── initialVideo/
│   │   ├── live/
│   │   └── logs/
│   ├── transcripts/
│   └── temp/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── requirements.txt
├── requirements-gpu.txt
├── pyproject.toml
├── README.md
└── .gitignore
```

## Typical workflow

1. Start the Streamlit app.
2. Manage persons in the database.
3. Run realtime recognition or process a video.
4. Review unknown faces and assign them to a known identity.
5. Open Voice Enrollment to listen to enrolled samples and unresolved speakers,
   edit the person assigned to an existing saved voice sample, or confirm an
   assignment/create a person for an unidentified speaker.
6. Continue recognition using vectors stored in Qdrant.

## Important data folders

- `known_faces/`: label/reference face images
- `known_voices/`: enrolled voice embeddings and their saved audio samples
- `unknown_faces/`: unresolved face samples
- `unknown_voices/`: unresolved voice embeddings and representative audio clips
- `transcripts/`: generated transcription outputs
- `trackedVideo/`: processed output videos
- `logs/`: runtime logs
- `faces.db`: legacy SQLite import source; runtime records are stored in PostgreSQL
- `face_index.npz`: legacy cache path handled during migration; runtime matching uses Qdrant

## Development and testing

Run the Python test suite:

```powershell
pytest -q
```

Check dependency health:

```powershell
python -m pip check
```

## Notes

- Active-speaker detection is a visual speech estimate, not a biometric voice ID system.
- Whisper transcription is optional and depends on FFmpeg for audio extraction.
- If microphone capture fails, realtime voice analysis may be disabled while face recognition continues.
- If ONNX Runtime does not detect CUDA, the app will continue in CPU mode.

## Troubleshooting

- FFmpeg is taken from PATH when available; otherwise the `imageio-ffmpeg` dependency supplies a bundled executable.
- If realtime startup fails, inspect the latest log file under `logs/`.
- If CUDA is unavailable, do not worry; CPU mode is supported.
