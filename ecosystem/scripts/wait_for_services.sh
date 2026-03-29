#!/bin/bash
# Espera hasta que todos los servicios Docker estén healthy
set -e

cd "$(dirname "$0")/.."

MAX_WAIT=300  # 5 minutos máximo
INTERVAL=5
ELAPSED=0

echo "==> Waiting for all services to become healthy (max ${MAX_WAIT}s)..."

while [ $ELAPSED -lt $MAX_WAIT ]; do
    UNHEALTHY=$(docker compose ps --format json 2>/dev/null | \
        python3 -c "
import sys, json
unhealthy = []
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    svc = json.loads(line)
    status = svc.get('Health', svc.get('Status', ''))
    if 'healthy' not in status.lower():
        unhealthy.append(svc.get('Service', svc.get('Name', 'unknown')))
print(','.join(unhealthy))
" 2>/dev/null || echo "error")

    if [ -z "$UNHEALTHY" ]; then
        echo ""
        echo "==> All services are healthy!"
        docker compose ps
        exit 0
    fi

    echo "    [${ELAPSED}s] Waiting on: $UNHEALTHY"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo ""
echo "==> TIMEOUT: Some services did not become healthy in ${MAX_WAIT}s"
docker compose ps
exit 1
