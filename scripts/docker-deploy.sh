#!/usr/bin/env bash
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$ROOT_DIR"

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker is required." >&2
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "Error: Docker Compose v2 (the 'docker compose' plugin) is required." >&2
    exit 1
fi
COMPOSE=(docker compose)

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example. Edit OPENAI_API_KEY and provider settings if needed."
fi

port=${MYCLAW_PORT:-}
if [ -z "$port" ] && [ -f .env ]; then
    port=$(sed -n 's/^MYCLAW_PORT=//p' .env | head -n 1)
fi
port=${port:-8765}

if ! "${COMPOSE[@]}" up -d --build --wait --wait-timeout 60; then
    "${COMPOSE[@]}" ps || true
    "${COMPOSE[@]}" logs --tail=100 myclaw || true
    exit 1
fi

echo "MyClaw is running at http://127.0.0.1:${port}"
