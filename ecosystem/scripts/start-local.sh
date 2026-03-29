#!/bin/bash
# Levanta servicios con override de desarrollo local (16 GB RAM)
set -e

cd "$(dirname "$0")/.."

echo "==> Starting RAG ecosystem (LOCAL DEV mode)..."
echo "    Using: docker-compose.yml + docker-compose.override.yml"
echo ""

docker compose -f docker-compose.yml -f docker-compose.override.yml up -d

echo ""
echo "==> Services starting. Check status with:"
echo "    docker compose ps"
echo "    ./scripts/wait_for_services.sh"
