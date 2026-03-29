#!/bin/bash
# =============================================================
# setup.sh — Primera instalación del ecosistema RAG
# Uso: cd ecosystem && ./scripts/setup.sh
# =============================================================
set -e

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[!!]${NC} $1"; }
die()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
step() { echo -e "\n${YELLOW}==>${NC} $1"; }

echo "============================================="
echo "  RAG Ecosystem — Setup inicial"
echo "============================================="

# ── PASO 1: .env ──────────────────────────────────────────────
step "PASO 1/5 — Configuración de entorno"
if [ ! -f "$ROOT/.env" ]; then
    cp "$ROOT/.env.example" "$ROOT/.env"
    warn ".env creado desde .env.example"
    warn "IMPORTANTE: Edita $ROOT/.env con tus credenciales reales antes de continuar"
    echo ""
    read -p "¿Quieres editar .env ahora? [y/N]: " EDIT_ENV
    if [[ "$EDIT_ENV" =~ ^[Yy]$ ]]; then
        ${EDITOR:-nano} "$ROOT/.env"
    else
        warn "Recuerda editar .env antes de levantar los servicios"
    fi
else
    ok ".env ya existe — omitiendo"
fi

# Verificar que .env no tenga CHANGE_ME sin reemplazar
UNSET=$(grep -c "CHANGE_ME" "$ROOT/.env" 2>/dev/null || true)
if [ "$UNSET" -gt 0 ]; then
    warn "$UNSET variable(s) aún tienen CHANGE_ME en .env — revísalas"
fi

# ── PASO 2: Python venv ───────────────────────────────────────
step "PASO 2/5 — Entorno Python"
if [ ! -d "$ROOT/venv" ]; then
    python3 -m venv "$ROOT/venv"
    ok "venv creado"
else
    ok "venv ya existe — omitiendo"
fi

source "$ROOT/venv/bin/activate" 2>/dev/null || source "$ROOT/venv/Scripts/activate"
pip install --quiet --upgrade pip
pip install --quiet -r "$ROOT/requirements.txt"
ok "Dependencias instaladas"

# ── PASO 3: Docker services ───────────────────────────────────
step "PASO 3/5 — Servicios Docker (locales)"
if ! command -v docker &>/dev/null; then
    die "Docker no está instalado"
fi

echo "    Levantando: Elasticsearch, Neo4j, RabbitMQ, ClickHouse, Langfuse, Traefik"
docker compose -f "$ROOT/docker-compose.yml" -f "$ROOT/docker-compose.override.yml" up -d

echo "    Esperando que estén healthy (puede tardar 2-3 min)..."
"$ROOT/scripts/wait_for_services.sh"

# ── PASO 4: Bootstrap PostgreSQL schema ───────────────────────
step "PASO 4/5 — Inicializar schema PostgreSQL"
PYTHONPATH="$ROOT" python3 - <<'PYEOF'
import asyncio, sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('.env')
from core.database import bootstrap_schema
async def run():
    await bootstrap_schema()
    print("Schema PostgreSQL inicializado")
asyncio.run(run())
PYEOF
ok "Schema PostgreSQL listo"

# ── PASO 5: Verificar conexiones ──────────────────────────────
step "PASO 5/5 — Verificando conexiones a todos los servicios"
PYTHONPATH="$ROOT" python3 "$ROOT/scripts/verify_connections.py" || warn "Algunos servicios no responden aún — reintenta en unos minutos"

echo ""
echo "============================================="
ok "Setup completado"
echo "  Siguiente paso: ./scripts/dev.sh"
echo "============================================="
