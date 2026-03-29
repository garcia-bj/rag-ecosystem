#!/bin/bash
# =============================================================
# dev.sh — Arranque diario del ecosistema RAG
# Levanta Docker + API + 2 workers Celery en una sola terminal
# Uso: cd ecosystem && ./scripts/dev.sh
# Salir: Ctrl+C (mata todo automáticamente)
# =============================================================
set -e

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[!!]${NC} $1"; }
die()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
step() { echo -e "\n${YELLOW}==>${NC} $1"; }

# Activar venv
if [ -f "$ROOT/venv/bin/activate" ]; then
    source "$ROOT/venv/bin/activate"
elif [ -f "$ROOT/venv/Scripts/activate" ]; then
    source "$ROOT/venv/Scripts/activate"
else
    die "venv no encontrado — ejecuta primero: ./scripts/setup.sh"
fi

# Verificar .env
[ -f "$ROOT/.env" ] || die ".env no encontrado — ejecuta primero: ./scripts/setup.sh"

# Guardar PIDs para cleanup
PIDS=()

cleanup() {
    echo ""
    warn "Apagando servicios..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    ok "Todo apagado. Docker sigue corriendo (usa 'docker compose stop' para pararlo)"
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "============================================="
echo "  RAG Ecosystem — Arranque dev"
echo "============================================="

# ── Docker services ───────────────────────────────────────────
step "1/4 — Servicios Docker"
RUNNING=$(docker compose ps --services --filter status=running 2>/dev/null | wc -l | tr -d ' ')
if [ "$RUNNING" -lt 5 ]; then
    echo "    Levantando contenedores..."
    docker compose -f "$ROOT/docker-compose.yml" -f "$ROOT/docker-compose.override.yml" up -d
    "$ROOT/scripts/wait_for_services.sh"
else
    ok "Docker ya corriendo ($RUNNING servicios activos)"
fi

# ── FastAPI ───────────────────────────────────────────────────
step "2/4 — FastAPI (puerto 8000)"
LOG_API="$ROOT/logs/api.log"
mkdir -p "$ROOT/logs"
PYTHONPATH="$ROOT" python3 -m uvicorn api.main:app \
    --host 0.0.0.0 --port 8000 --reload \
    > "$LOG_API" 2>&1 &
PIDS+=($!)
ok "API arrancando → http://localhost:8000/docs   (logs: logs/api.log)"

# Esperar que la API responda
echo "    Esperando que la API esté lista..."
for i in $(seq 1 20); do
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        ok "API lista"
        break
    fi
    sleep 1
done

# ── Celery worker ingesta ─────────────────────────────────────
step "3/4 — Celery worker: ingesta"
LOG_INGEST="$ROOT/logs/celery_ingest.log"
PYTHONPATH="$ROOT" celery -A ingestion.workers.celery_app worker \
    --queues=ingest \
    --concurrency=2 \
    --loglevel=info \
    --include=ingestion.workers.ingest_task \
    > "$LOG_INGEST" 2>&1 &
PIDS+=($!)
ok "Worker ingesta arrancado   (logs: logs/celery_ingest.log)"

# ── Celery worker embeddings ──────────────────────────────────
step "4/4 — Celery worker: embeddings"
LOG_EMBED="$ROOT/logs/celery_embed.log"
PYTHONPATH="$ROOT" celery -A ingestion.workers.celery_app worker \
    --queues=embed,index \
    --concurrency=2 \
    --loglevel=info \
    --include=embeddings.workers.index_task \
    > "$LOG_EMBED" 2>&1 &
PIDS+=($!)
ok "Worker embeddings arrancado (logs: logs/celery_embed.log)"

echo ""
echo "============================================="
ok "Todo el ecosistema está corriendo"
echo ""
echo "  API:        http://localhost:8000/docs"
echo "  Langfuse:   http://localhost:3000"
echo "  RabbitMQ:   http://localhost:15672"
echo "  Neo4j:      http://localhost:7474"
echo ""
echo "  Logs en: $ROOT/logs/"
echo "  Ctrl+C para apagar todo"
echo "============================================="

# Esperar hasta Ctrl+C
wait
