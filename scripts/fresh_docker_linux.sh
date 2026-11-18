#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
cd "$project_root"
export APP_ENV_FILE="$project_root/.env"
DATA_ROOT="${DATA_ROOT:-$project_root/data}"
export DATA_ROOT

fail() {
    printf 'Error: %s\n' "$1" >&2
    exit 1
}

for command_name in docker openssl python3; do
    command -v "$command_name" >/dev/null 2>&1 || fail "Required command not found: $command_name"
done

[[ -f docker-compose.yml ]] || fail "Run this script from a checkout containing docker-compose.yml."
docker info >/dev/null 2>&1 || fail "Docker Engine is unavailable or the current user cannot access it."
docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is required."

if [[ -L .env ]]; then
    fail ".env is a symlink; refusing to replace it."
fi

if [[ -e .env ]]; then
    env_backup="$(mktemp ".env.backup.$(date +%Y%m%d%H%M%S).XXXXXX")"
    cp .env "$env_backup"
    chmod 600 "$env_backup"
    env_recreate="$(mktemp .env.recreate.XXXXXX)"
    install -m 600 "$env_backup" "$env_recreate"
    mv -f "$env_recreate" .env
    printf 'Backed up the existing .env to %s and recreated it with its settings preserved.\n' \
        "$env_backup"
fi

postgres_volume_exists=0
if docker volume inspect face_voice_postgres >/dev/null 2>&1; then
    postgres_volume_exists=1
fi

qdrant_volume_exists=0
if docker volume inspect face_voice_qdrant >/dev/null 2>&1; then
    qdrant_volume_exists=1
fi

if [[ ! -f .env ]]; then
    if (( postgres_volume_exists )); then
        printf 'Existing PostgreSQL volume found. Enter its DATABASE_URL (input hidden): ' >&2
        IFS= read -r -s database_url || fail "Could not read DATABASE_URL."
        printf '\n' >&2
        [[ -n "$database_url" ]] || fail "DATABASE_URL cannot be empty."
        printf 'DATABASE_URL=%s\n' "$database_url" > .env
        unset database_url
    else
        database_password="$(openssl rand -hex 32)"
        printf 'DATABASE_URL=postgresql+psycopg://face_voice_rec:%s@localhost:5432/face_voice_rec\n' \
            "$database_password" > .env
        unset database_password
    fi
    chmod 600 .env
fi

database_url="$(python3 - <<'PY'
from pathlib import Path

for line in Path(".env").read_text().splitlines():
    if line.startswith("DATABASE_URL="):
        value = line.partition("=")[2].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        print(value)
        break
PY
)"
[[ -n "$database_url" ]] || fail ".env must contain DATABASE_URL."

database_password="$(DATABASE_URL_VALUE="$database_url" python3 - <<'PY'
from urllib.parse import unquote, urlsplit
import os

url = urlsplit(os.environ["DATABASE_URL_VALUE"])
if url.scheme not in {"postgresql", "postgresql+psycopg"}:
    raise SystemExit("DATABASE_URL must use PostgreSQL.")
if unquote(url.username or "") != "face_voice_rec":
    raise SystemExit("DATABASE_URL must use the face_voice_rec user.")
if unquote(url.path.lstrip("/")) != "face_voice_rec":
    raise SystemExit("DATABASE_URL must target the face_voice_rec database.")
password = unquote(url.password or "")
if not password or "\n" in password or "\r" in password:
    raise SystemExit("DATABASE_URL must contain a usable password.")
print(password)
PY
)" || fail "Could not read a valid PostgreSQL password from DATABASE_URL."

init_env_file="$(mktemp "${TMPDIR:-/tmp}/face-voice-postgres-init.XXXXXX")"
container_started=0

cleanup() {
    exit_status=$?
    trap - EXIT
    if (( container_started )); then
        if (( exit_status != 0 )); then
            docker logs "$init_container" >&2 2>/dev/null || true
        fi
        docker stop --time 30 "$init_container" >/dev/null 2>&1 || true
        docker rm "$init_container" >/dev/null 2>&1 || true
    fi
    rm -f "$init_env_file"
    exit "$exit_status"
}
trap cleanup EXIT

docker compose config --quiet || fail "Compose configuration validation failed."

if (( ! postgres_volume_exists )); then
    init_container="face-voice-postgres-init"
    if docker container inspect "$init_container" >/dev/null 2>&1; then
        fail "Container '$init_container' already exists; inspect it before continuing."
    fi

    printf 'POSTGRES_USER=face_voice_rec\nPOSTGRES_DB=face_voice_rec\nPOSTGRES_PASSWORD=%s\n' \
        "$database_password" > "$init_env_file"
    docker volume create face_voice_postgres >/dev/null

    docker run -d \
        --name "$init_container" \
        --env-file "$init_env_file" \
        --volume face_voice_postgres:/var/lib/postgresql/data \
        postgres:17 >/dev/null
    container_started=1

    printf 'Waiting for the new PostgreSQL database to initialize...\n'
    database_ready=0
    for ((attempt = 0; attempt < 60; attempt++)); do
        if docker exec "$init_container" \
            psql -U face_voice_rec -d face_voice_rec -Atqc 'SELECT 1' >/dev/null 2>&1; then
            database_ready=1
            break
        fi
        if [[ "$(docker inspect --format '{{.State.Running}}' "$init_container" 2>/dev/null || true)" != true ]]; then
            docker logs "$init_container" >&2 2>/dev/null || true
            fail "The PostgreSQL initialization container stopped unexpectedly."
        fi
        sleep 2
    done

    (( database_ready )) || fail "PostgreSQL did not initialize within 120 seconds."

    docker stop --time 30 "$init_container" >/dev/null
    docker rm "$init_container" >/dev/null
    container_started=0
else
    printf 'Reusing existing PostgreSQL volume without changing its contents or credentials.\n'
fi

if (( ! qdrant_volume_exists )); then
    docker volume create face_voice_qdrant >/dev/null
else
    printf 'Reusing existing Qdrant volume without changing its contents.\n'
fi

rm -f "$init_env_file"
init_env_file=""
trap - EXIT

printf 'Starting the Docker Compose stack...\n'
docker compose up -d --build
docker compose ps
printf '\nApp health: http://localhost:8502/_stcore/health\n'
printf 'The .env file is mode 600. Keep it private.\n'