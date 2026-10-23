# Docker Compose Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the Streamlit app in an NVIDIA/CUDA-enabled Docker image with PostgreSQL and Qdrant Compose services that reuse the existing data volumes.

**Architecture:** Keep the app, PostgreSQL and Qdrant in separate Linux containers. The app image contains source and Python dependencies only; Compose supplies root `.env`, routes the app to internal `postgres` and `qdrant` service names, mounts `./data`, and persists model downloads separately. The transcript LLM server remains external.

**Tech Stack:** Dockerfile, Docker Compose v2.30+, Python 3.14 slim, `requirements-gpu.txt` (CUDA 12.8 PyTorch build), PostgreSQL 17, Qdrant v1.20.0, Streamlit, FFmpeg, NVIDIA Container Toolkit/Docker Desktop GPU support.

**Design:** `docs/superpowers/specs/2026-10-23-docker-compose-deployment-design.md`

## Global Constraints

- Run commands from `D:\Git\FaceAndVoiceRecV2` in PowerShell.
- Keep existing Docker volumes `face_voice_postgres` and `face_voice_qdrant`; declare them `external: true`.
- Never run `docker compose down -v`, `docker volume rm`, or any command that deletes/recreates the existing data volumes.
- Before stopping current services, verify mounts and create a PostgreSQL dump plus Qdrant snapshots in a timestamped directory under `data/backups/`.
- Stop only `face-voice-postgres` and `face-voice-qdrant`, and only after backups and the app image build pass.
- `.env` must never enter the Docker build context, image, logs, or Git. Never print secret values.
- The user must add `POSTGRES_PASSWORD` locally to the ignored root `.env`, matching the password that initialized the current PostgreSQL volume; do not guess or replace credentials.
- Inside the app container, `DATABASE_URL` uses host `postgres` and `QDRANT_URL` uses `http://qdrant:6333`, not `localhost`.
- Keep the configured OpenAI-compatible model server external and use its existing `.env` URL.
- Persist application data with `./data:/app/data` and model caches in a separate named volume mounted at `/models`.
- Streamlit listens on `0.0.0.0:8501`; host access is `http://localhost:8501`.
- GPU access must be demonstrated in a container before DB/Qdrant handoff. If the Python/CUDA combination fails to build or report a CUDA device, stop and report before stopping services.
- Windows Docker Desktop supports uploaded/prerecorded media. Webcam/microphone mapping is Linux-host-only; do not imply it works on Windows.
- Stop the stack with `docker compose down` only; never append `-v`.

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `Dockerfile` | Create | CUDA Python runtime, FFmpeg/OS libs, GPU requirements, Qdrant readiness wait, Streamlit entrypoint |
| `.dockerignore` | Create | Exclude `.env`, `data`, VCS, virtualenvs and caches from build context |
| `docker-compose.yml` | Create | App/PostgreSQL/Qdrant services, PostgreSQL health check, GPU request and persistent/external volumes |
| `docker-compose.linux-devices.yml` | Create | Optional Linux-only webcam and sound device mappings |
| `.gitignore` | Modify | Ignore `data/backups/` |
| `README.md` | Modify | Credentials, backups, safe handoff, build/run/stop, GPU and Windows limitations |

---

### Task 1: Preflight Docker, GPU, current volumes and credentials

**Files:** none; read-only inspection only.

**Interfaces:**
- Consumes: containers `face-voice-postgres` and `face-voice-qdrant`; volumes `face_voice_postgres` and `face_voice_qdrant`; root `.env`.
- Produces: verified Compose/GPU availability and redacted DB identity, volume mounts, Qdrant version/collections. Never print passwords.

- [ ] **Step 1: Confirm Docker Engine, Compose and volumes**

Run:

```powershell
docker version --format '{{.Server.Version}}'
docker compose version
docker volume inspect face_voice_postgres face_voice_qdrant --format '{{.Name}}'
```

Expected: Engine and Compose v2.30+ respond; both volumes exist. Otherwise stop and ask the user to start/configure Docker Desktop. Do not create replacements.

- [ ] **Step 2: Verify the existing container mounts and versions**

Run:

```powershell
docker inspect face-voice-postgres face-voice-qdrant --format '{{.Name}} {{.State.Status}} {{.Config.Image}} {{range .Mounts}}{{.Name}}:{{.Destination}} {{end}}'
Invoke-RestMethod http://localhost:6333/ | Select-Object version
```

Expected: Postgres uses `face_voice_postgres:/var/lib/postgresql/data`; Qdrant uses `face_voice_qdrant:/qdrant/storage`; running Qdrant reports v1.20.0 and Postgres major version 17. If a mount or version differs, stop and update Compose before handoff.

- [ ] **Step 3: Verify GPU passthrough without touching project services**

Run:

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

Expected: `nvidia-smi` sees the host GPU. If this diagnostic image is unavailable, use the same known-good CUDA image already tested by the user. If no GPU is visible, stop; do not silently substitute a CPU image.

- [ ] **Step 4: Verify DB identity and current Qdrant collections without exposing secrets**

Run:

```powershell
docker exec face-voice-postgres psql -U face_voice_rec -d face_voice_rec -Atc 'SELECT current_database(), current_user'
(Invoke-RestMethod http://localhost:6333/collections).result.collections | Select-Object name
```

Expected: `face_voice_rec|face_voice_rec` and the existing collection list. Record collection names/counts without printing keys or media. Check only whether `.env` contains `POSTGRES_PASSWORD`:

```powershell
$hasPostgresPassword = Select-String -Path .env -Pattern '^POSTGRES_PASSWORD=' -Quiet
Write-Output "POSTGRES_PASSWORD present: $hasPostgresPassword"
```

If absent, ask the user to add it locally with the current DB password; never print the value.

- [ ] **Step 5: Confirm the external model endpoint is reachable**

Run:

```powershell
@'
import json
from urllib.request import urlopen
from face_voice_rec import config

with urlopen(config.TRANSCRIPT_REFINEMENT_URL.rstrip('/') + '/v1/models', timeout=15) as response:
  payload = json.loads(response.read())
  print(f"HTTP {response.status}")
  print([model.get("id") for model in payload.get("data", [])])
'@ | .\directmlvenv\Scripts\python.exe -
```

Expected: HTTP 200. Stop before service handoff if unreachable.

---

### Task 2: Create and build the app image safely

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

**Interfaces:**
- Consumes: project source except `.env`, runtime `data/`, `.git`, virtual environments and caches.
- Produces: image `face-voice-rec:gpu`; source stays at `/app`, and `PYTHONPATH=/app/src` ensures `config.paths.PROJECT_ROOT` resolves to `/app`.

- [ ] **Step 1: Create `.dockerignore`**

Use:

```dockerignore
.env
.env.*
.git
.gitignore
.worktrees/
.venv/
venv/
directmlvenv/
3.12venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
*.egg-info/
data/
docker-backups/
*.log
```

Do not ignore `app/`, `src/`, `migrations/`, `alembic.ini`, `pyproject.toml`, `requirements-gpu.txt`, `sitecustomize.py`, or README.

- [ ] **Step 2: Create `Dockerfile`**

Use Python 3.14 slim, FFmpeg, OpenCV/audio runtime libraries, GPU requirements, `/app/src` on the Python path, and persistent model caches. Poll Qdrant for at most 180 seconds before starting Streamlit; fail clearly if `/readyz` never succeeds:

```dockerfile
FROM python:3.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    XDG_CACHE_HOME=/models/cache \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libgomp1 libportaudio2 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-gpu.txt ./requirements-gpu.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-gpu.txt

COPY app ./app
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini pyproject.toml sitecustomize.py ./

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["sh", "-c", "for attempt in $(seq 1 90); do if curl -fsS \"${QDRANT_URL:-http://localhost:6333}/readyz\" >/dev/null; then exec python -m streamlit run app/main.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true; fi; sleep 2; done; echo 'Qdrant did not become ready within 180 seconds' >&2; exit 1"]
```

This deliberately does not `pip install .`: keeping `/app/src` on `PYTHONPATH` makes `config.paths.PROJECT_ROOT` resolve to `/app`, so `./data` resolves to `/app/data`. The startup command polls Qdrant readiness because the current official Qdrant image has no shell or HTTP client for its own Compose health check. Do not use `COPY . .` or pass `.env` during build.

If Python 3.14 plus `requirements-gpu.txt` does not build or cannot see CUDA, stop before modifying service containers and report the exact incompatibility for approval.

- [ ] **Step 3: Build the app image**

Run:

```powershell
docker build --pull -t face-voice-rec:gpu .
```

Expected: exit code 0.

- [ ] **Step 4: Verify CUDA and exclusion of local data/secrets**

Run:

```powershell
docker run --rm --gpus all --entrypoint python face-voice-rec:gpu -c "from pathlib import Path; import torch; print('cuda_available=', torch.cuda.is_available()); print('env_file_in_image=', Path('/app/.env').exists()); print('data_dir_in_image=', Path('/app/data').exists())"
```

Expected: CUDA is available, `/app/.env` and `/app/data` do not exist in the image. This isolated command must not attach project volumes.

---

### Task 3: Add Compose services and Linux-only device override

**Files:**
- Create: `docker-compose.yml`
- Create: `docker-compose.linux-devices.yml`

**Interfaces:**
- `postgres`: PostgreSQL 17, external `face_voice_postgres` volume, Compose health check.
- `qdrant`: pinned `qdrant/qdrant:v1.20.0`, external `face_voice_qdrant` volume; app startup polls `/readyz`.
- `app`: root `.env` at runtime, internal DB/Qdrant URLs, `./data` and model-cache mounts, GPU, port 8501, app health check.

- [ ] **Step 1: Create `docker-compose.yml`**

Use this service shape; `POSTGRES_PASSWORD` must match the credential that initialized the existing database volume:

```yaml
services:
  postgres:
    image: postgres:17
    environment:
      POSTGRES_USER: face_voice_rec
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?Add the current DB password as POSTGRES_PASSWORD in root .env}
      POSTGRES_DB: face_voice_rec
    ports:
      - "5432:5432"
    volumes:
      - face_voice_postgres:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $${POSTGRES_USER} -d $${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 10
      start_period: 20s
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:v1.20.0
    ports:
      - "6333:6333"
    volumes:
      - face_voice_qdrant:/qdrant/storage
    restart: unless-stopped

  app:
    image: face-voice-rec:gpu
    build:
      context: .
      dockerfile: Dockerfile
    env_file:
      - .env
    environment:
      DATABASE_URL: postgresql+psycopg://face_voice_rec:${POSTGRES_PASSWORD}@postgres:5432/face_voice_rec
      QDRANT_URL: http://qdrant:6333
      QDRANT_API_KEY: ""
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - model-cache:/models
    gpus: all
    depends_on:
      postgres:
        condition: service_healthy
      qdrant:
        condition: service_started
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 5s
      retries: 5
      start_period: 120s
    restart: unless-stopped

volumes:
  face_voice_postgres:
    external: true
    name: face_voice_postgres
  face_voice_qdrant:
    external: true
    name: face_voice_qdrant
  model-cache:
```

Compose loads root `.env`; the app receives remaining settings via `env_file`. Compose overrides `DATABASE_URL` to use `postgres` and `QDRANT_URL` to use `qdrant`. Keep the model URL external. Do not set `container_name`.

- [ ] **Step 2: Add the optional Linux-only device override**

Create `docker-compose.linux-devices.yml`:

```yaml
services:
  app:
    devices:
      - /dev/video0:/dev/video0
    volumes:
      - /dev/snd:/dev/snd
```

Document not to use it with Windows Docker Desktop.

- [ ] **Step 3: Validate Compose without printing resolved secrets**

Run:

```powershell
docker compose config --quiet
```

Expected: exit code 0. If `POSTGRES_PASSWORD` is absent, stop and tell the user to add the existing password locally; never print it. Do not start services yet.

---

### Task 4: Document credentials, backups, handoff and run commands

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`

- [ ] **Step 1: Ignore backup artifacts**

Add `data/backups/` to `.gitignore` so local DB and Qdrant backups cannot be committed.

- [ ] **Step 2: Document local Compose configuration**

Document adding `POSTGRES_PASSWORD` to root `.env` with the same existing database credential. Do not include a credential value in README or source control. Explain that the app service uses `postgres:5432` and `qdrant:6333`, and transcript refinement still uses the external URL in `.env`.

- [ ] **Step 3: Document backup directory creation**

Use this before backup commands:

```powershell
$backupRoot = Join-Path $PWD ("data\backups\docker-handoff-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null
```

- [ ] **Step 4: Document PostgreSQL backup**

```powershell
docker exec face-voice-postgres pg_dump -U face_voice_rec -d face_voice_rec -Fc -f /tmp/face_voice_rec.dump
if ($LASTEXITCODE -ne 0) { throw "pg_dump failed; do not stop current containers" }
docker cp face-voice-postgres:/tmp/face_voice_rec.dump "$backupRoot\face_voice_rec.dump"
if ($LASTEXITCODE -ne 0) { throw "docker cp failed; do not stop current containers" }
if ((Get-Item "$backupRoot\face_voice_rec.dump").Length -le 0) { throw "Postgres backup is empty" }
docker exec face-voice-postgres rm -f /tmp/face_voice_rec.dump
```

- [ ] **Step 5: Document Qdrant snapshot backup**

```powershell
$collections = (Invoke-RestMethod http://localhost:6333/collections).result.collections
foreach ($collection in $collections) {
    $name = [uri]::EscapeDataString($collection.name)
    $snapshot = Invoke-RestMethod -Method Post -Uri "http://localhost:6333/collections/$name/snapshots"
    $snapshotName = [uri]::EscapeDataString($snapshot.result.name)
    $path = Join-Path $backupRoot "$($collection.name).snapshot"
    Invoke-WebRequest -Uri "http://localhost:6333/collections/$name/snapshots/$snapshotName" -OutFile $path -UseBasicParsing
    if ((Get-Item $path).Length -le 0) { throw "Qdrant snapshot is empty: $($collection.name)" }
}
```

Only after both backups and image build pass, document stopping the old services with `docker stop face-voice-postgres face-voice-qdrant` and starting Compose with `docker compose up -d --build`. Warn against `docker compose down -v`, `docker volume rm`, and `docker rm` for old data containers. Include rollback with `docker compose down` (without `-v`) and starting the old named containers. Restoring backups requires explicit confirmation.

- [ ] **Step 6: Document health checks, GPU/model checks, and limitations**

Document checks for `docker compose ps`, `http://localhost:8501/_stcore/health`, `http://localhost:6333/readyz`, Postgres `pg_isready`, CUDA visibility, model endpoint access from the app container, preserved Postgres/Qdrant data, and one prerecorded media workflow. Explain `./data` and model-cache persistence, plus the Windows webcam/microphone limitation.

- [ ] **Step 7: Check README diff and secret hygiene**

Run:

```powershell
git diff --check
```

Inspect the Docker README section and confirm it contains no actual credentials or `.env` contents.

---

### Task 5: Back up data and hand off existing services

**Files:** none; runtime actions are deliberate and authorized by the user.

- [ ] **Step 1: Recheck preconditions immediately before handoff**

Repeat container mount inspection, `docker compose config --quiet`, app image build, GPU visibility and model endpoint health. Verify locally that `POSTGRES_PASSWORD` is set to the existing DB credential without printing it. If any precondition differs, leave old services running and stop.

- [ ] **Step 2: Run and verify backups**

Run the README backup procedure. Verify the Postgres dump and one non-empty snapshot per current Qdrant collection exist under `$backupRoot`. Record pre-handoff counts for Postgres tables `persons`, `face_embeddings`, `voice_embeddings`, `meetings`, plus Qdrant collection names/point counts. Do not output transcript text, images, vectors or secrets.

- [ ] **Step 3: Stop only current PostgreSQL and Qdrant containers**

Run:

```powershell
docker stop face-voice-postgres face-voice-qdrant
```

Verify both are stopped and both named volumes still exist. Do not remove the containers or volumes.

---

### Task 6: Start and verify the Compose stack

**Files:** none; runtime validation only.

- [ ] **Step 1: Start services**

Run:

```powershell
docker compose up -d --build
docker compose ps
```

Expected: app, postgres and qdrant start; app waits for both Postgres health and Qdrant `/readyz`. If startup fails, preserve logs, run `docker compose down` without `-v`, restart the original containers, and keep all volumes intact.

- [ ] **Step 2: Verify app and dependency health**

Run:

```powershell
Invoke-RestMethod http://localhost:8501/_stcore/health
Invoke-RestMethod http://localhost:6333/readyz
docker compose exec -T postgres pg_isready -U face_voice_rec -d face_voice_rec
```

Expected: Streamlit and Qdrant return success; Postgres reports accepting connections.

- [ ] **Step 3: Verify CUDA and model-server access inside app**

Run:

```powershell
docker compose exec -T app python -c "import os, json, urllib.request, torch, onnxruntime; assert torch.cuda.is_available(); print('cuda_available=True'); print('onnx_providers=', onnxruntime.get_available_providers()); u=os.environ['TRANSCRIPT_REFINEMENT_URL'].rstrip('/')+'/v1/models'; d=json.load(urllib.request.urlopen(u, timeout=30)); print('model_ids=', [x.get('id') for x in d.get('data', [])])"
```

Expected: CUDA is available, ONNX providers include CUDA, and model API responds. Do not print a full URL or secret.

- [ ] **Step 4: Compare existing data after handoff**

Repeat Task 5 Postgres counts and Qdrant collection/point counts through Compose. Expected: existing records, collection names and point counts remain visible. If not, stop Compose without `-v`, restart old containers, and investigate before writes.

- [ ] **Step 5: Smoke-test uploaded media and persistent paths**

Open `http://localhost:8501`, upload a small prerecorded MP4 or MP3, run its workflow, and verify source media under `data/artifacts/initialVideo` and generated outputs/transcripts/logs under `data/`. Do not test Windows webcam/microphone in this Linux container.

- [ ] **Step 6: Verify restart persistence**

Run `docker compose restart app`, recheck app health, and verify the uploaded file, transcript, Postgres records and Qdrant collections remain. Do not remove volumes.

- [ ] **Step 7: Commit only after verification passes**

```powershell
git add Dockerfile .dockerignore docker-compose.yml docker-compose.linux-devices.yml .gitignore README.md
git commit -m "feat: run app and data services with Docker Compose"
```

Do not add `.env`, backups, model caches, or `data/` runtime outputs to Git.
