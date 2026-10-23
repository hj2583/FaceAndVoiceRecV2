# Docker Compose Deployment Design

## Goal

Build and run the FaceAndVoiceRec Streamlit app in Linux containers with NVIDIA CUDA support, while preserving the existing PostgreSQL, Qdrant, media, and model data.

## Current state

- The app starts at `app/main.py` and expects PostgreSQL (`DATABASE_URL`).
- Face and voice vectors are stored in Qdrant (`QDRANT_URL`). The app does not start PostgreSQL or Qdrant itself.
- Root `.env` contains local runtime configuration, including the external OpenAI-compatible transcript model endpoint. It must not be copied into the image or committed.
- Runtime paths, including uploads, tracked videos, transcripts, and app logs, are under `data/`.
- CPU requirements are in `requirements.txt`; the NVIDIA CUDA variant is in `requirements-gpu.txt`.
- The README documents existing Docker volume names `face_voice_postgres` and `face_voice_qdrant`.
- The app supports video/audio file uploads. Windows Docker Desktop does not directly expose Windows webcam/microphone devices to a Linux container.

## Architecture

Create a Linux NVIDIA/CUDA application image and a Docker Compose stack with separate services for the app, PostgreSQL, and Qdrant. PostgreSQL and Qdrant use the existing named volumes, declared as external Compose volumes; they must not be recreated or deleted. Mount `./data` into the app container for persistent application data and logs, and use a separate named volume for downloaded ML model caches. The transcript LLM server remains external and is addressed by the URL in `.env`.

The image uses Python 3.14, installs `requirements-gpu.txt`, system libraries required by OpenCV/audio, and FFmpeg. Compose maps the app's database and Qdrant URLs to the internal service names (`postgres` and `qdrant`) rather than `localhost`. The app listens on container port 8501 and maps to host port 8501. Health checks gate app startup on PostgreSQL and Qdrant readiness.

## Docker files

- `Dockerfile`: reproducible NVIDIA/CUDA-capable application image, installs runtime libraries, FFmpeg, Python GPU requirements, and the project package; starts Streamlit on `0.0.0.0:8501`.
- `docker-compose.yml`: app, PostgreSQL and Qdrant services; GPU reservation for the app; health checks; persistent data/model-cache mounts; external declarations for the existing PostgreSQL and Qdrant volumes; service-address overrides.
- `.dockerignore`: exclude `.env`, `.git`, virtual environments, local caches, `data/`, test outputs, and other local artifacts from the image build context.
- Optional `docker-compose.linux-devices.yml`: Linux-host-only webcam/microphone device mappings. Not used on Windows Docker Desktop.
- `README.md`: backup, environment, build/start, verification, stop/restart and troubleshooting instructions.

## Configuration and secret handling

Compose reads the root `.env` locally with `env_file` and passes runtime values to the app. The image build must not receive the contents of `.env`. Compose overrides `DATABASE_URL` to use `postgres:5432` and `QDRANT_URL` to use `http://qdrant:6333`. The PostgreSQL service must be configured with the same user, database, and password that initialized the existing data volume. The app continues to use the external model-server URL configured in `.env`.

Before Compose starts, inspect the existing containers and verify their named volume mounts and PostgreSQL database credentials. Create a PostgreSQL logical backup and Qdrant collection snapshots before stopping either service. Never run `docker compose down -v`, `docker volume rm`, or any command that deletes existing volumes. Compose must declare the existing volume names as `external: true`.

## Data and hardware behavior

`./data` is bind-mounted into the app container so uploaded source media, tracked outputs, transcripts, databases that happen to use local paths, and application logs remain on the host. Model caches persist in a named volume to avoid re-downloading Whisper/SpeechBrain model files after each container recreation.

NVIDIA GPU acceleration is supported only when the host NVIDIA driver and Docker GPU passthrough are working. Windows Docker Desktop supports file uploads and prerecorded video/audio processing but cannot pass Windows webcam/microphone devices directly into a Linux container. Realtime webcam/microphone remains a host-run workflow on this Windows machine. The Linux device override is only for hosts that expose `/dev/video0` and `/dev/snd`.

## Startup and verification

1. Verify Docker GPU access and the model endpoint before disrupting existing services.
2. Back up PostgreSQL and Qdrant and verify the external named volumes.
3. Stop the existing PostgreSQL and Qdrant containers without removing their volumes.
4. Build the application image and start the Compose stack.
5. Verify app health at `http://localhost:8501`, PostgreSQL health, Qdrant health, GPU visibility in the app container, the external model server endpoint, and the presence of existing data.
6. Run one prerecorded video/audio workflow and verify that generated media, transcripts, and logs are persisted under `./data`.

A failed verification must not trigger deletion or recreation of the existing data volumes. If the existing volume names, credentials, or GPU configuration differ from the assumptions above, stop before service handoff and revise the Compose configuration with the user.

## Out of scope

- Bundling PostgreSQL, Qdrant, the local LLM server, or model weights into the application image.
- Migrating to fresh databases or volumes.
- Windows webcam/microphone passthrough into the Linux container.
- Changing application runtime behavior unrelated to container support.
