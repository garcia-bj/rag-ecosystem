#!/bin/bash
# Levanta servicios con configuración de producción (Hetzner 128 GB RAM)
set -e

cd "$(dirname "$0")/.."

echo "==> Starting RAG ecosystem (PRODUCTION mode)..."
echo "    Using: docker-compose.yml only (no override)"
echo ""

# Pre-check: vm.max_map_count for Elasticsearch
MAPCOUNT=$(sysctl -n vm.max_map_count 2>/dev/null || echo "0")
if [ "$MAPCOUNT" -lt 262144 ]; then
    echo "WARNING: vm.max_map_count=$MAPCOUNT (need >= 262144 for Elasticsearch)"
    echo "Run: sudo sysctl -w vm.max_map_count=262144"
    exit 1
fi

docker compose -f docker-compose.yml up -d

echo ""
echo "==> Services starting. Check status with:"
echo "    docker compose ps"
echo "    ./scripts/wait_for_services.sh"
