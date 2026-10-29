#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
cd "$project_root"

fail() {
    printf 'Error: %s\n' "$1" >&2
    exit 1
}

for command_name in docker openssl; do
    command -v "$command_name" >/dev/null 2>&1 || fail "Required command not found: $command_name"
done

[[ -f docker-compose.yml ]] || fail "Run this script from a checkout containing docker-compose.yml."
docker info >/dev/null 2>&1 || fail "Docker Engine is unavailable or the current user cannot access it."
docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is required."

if [[ -e .env || -L .env ]]; then
    fail ".env already exists; refusing to overwrite it. Review it and start the stack manually with docker compose up -d --build."
fi

for volume_name in face_voice_postgres face_voice_qdrant; do
    if docker volume inspect "$volume_name" >/dev/null 2>&1; then
        fail "Volume '$volume_name' already exists; refusing to reuse or reset it for a fresh install."
    fi
done

init_container="face-voice-postgres-init"
if docker container inspect "$init_container" >/dev/null 2>&1; then
    fail "Container '$init_container' already exists; inspect it before continuing."
fi

database_password="$(openssl rand -hex 32)"
printf 'DATABASE_URL=postgresql+psycopg://face_voice_rec:%s@localhost:5432/face_voice_rec\n' \
    "$database_password" > .env
unset database_password

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

database_password="$(sed -n 's|^DATABASE_URL=postgresql+psycopg://face_voice_rec:\([^@]*\)@localhost:5432/face_voice_rec$|\1|p' .env)"
[[ -n "$database_password" ]] || fail "Could not prepare the generated database credentials."
printf 'POSTGRES_USER=face_voice_rec\nPOSTGRES_DB=face_voice_rec\nPOSTGRES_PASSWORD=%s\n' \
    "$database_password" > "$init_env_file"
unset database_password

docker compose config --quiet || fail "Compose configuration validation failed."

docker volume create face_voice_postgres >/dev/null
docker volume create face_voice_qdrant >/dev/null

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
rm -f "$init_env_file"
init_env_file=""
trap - EXIT

printf 'PostgreSQL is initialized. Building and starting the fresh Docker Compose stack...\n'
docker compose up -d --build
docker compose ps
printf '\nApp health: http://localhost:8502/_stcore/health\n'
printf 'Generated .env is mode 600 and was not printed. Keep it private.\n'