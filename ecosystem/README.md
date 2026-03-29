# RAG Ecosystem

Sistema de Recuperación Aumentada con Generación (RAG) multi-tenant, multi-modal y production-ready. Permite ingestar documentos de múltiples formatos, indexarlos en bases de datos vectoriales y responder preguntas sobre ellos usando LLMs.

---

## Índice

- [Arquitectura general](#arquitectura-general)
- [Tipos de documentos soportados](#tipos-de-documentos-soportados)
- [OCR automático](#ocr-automático)
- [Almacenamiento de documentos (RustFS)](#almacenamiento-de-documentos-rustfs)
- [Stack tecnológico](#stack-tecnológico)
- [Sistema de usuarios y planes](#sistema-de-usuarios-y-planes)
- [Pipeline de ingesta](#pipeline-de-ingesta)
- [Pipeline de retrieval](#pipeline-de-retrieval)
- [LLM Router](#llm-router)
- [API REST](#api-rest)
- [Observabilidad](#observabilidad)
- [Evaluación con RAGAS](#evaluación-con-ragas)
- [MCP Server](#mcp-server)
- [Instalación y arranque](#instalación-y-arranque)
- [Variables de entorno](#variables-de-entorno)
- [Producción y escalado](#producción-y-escalado)
- [Limitaciones conocidas](#limitaciones-conocidas)

---

## Arquitectura general

```
┌──────────────────────────────────────────────────────────────┐
│                        Clientes                              │
│   rag_ui.html · Swagger /docs · n8n · Claude Desktop (MCP)  │
└───────────────────────────┬──────────────────────────────────┘
                            │ HTTP / SSE
┌───────────────────────────▼──────────────────────────────────┐
│                     FastAPI  (:8000)                         │
│   /auth  /ingest  /query  /admin  /health                    │
│   JWT auth · API-Key auth · Rate limiting · CORS             │
└──────┬────────────────────┬─────────────────────────────────┘
       │ Celery task        │ Query
       ▼                    ▼
┌─────────────┐    ┌────────────────────────────────────────┐
│  RabbitMQ   │    │         Retrieval Pipeline             │
│  (broker)   │    │  Cache → Expand → Embed → Hybrid       │
└──┬──────────┘    │  Retrieval → Rerank → Compress → LLM  │
   │               └──────────────┬─────────────────────────┘
   ▼                              │
┌─────────────────┐    ┌──────────▼──────────┐
│  Ingest Worker  │    │     LLM Router       │
│  parse → chunk  │    │  Haiku/GPT-4o-mini  │
│  → embed → idx  │    │  GPT-4o / fallback  │
└────────┬────────┘    └─────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│         Almacenamiento               │
│  RustFS (S3)   Qdrant   ES   Neo4j   │
│  tenant/user/  vector   BM25  grafo  │
└──────────────────────────────────────┘
```

---

## Tipos de documentos soportados

### Texto (modality: `text`)
| Formato | Extensión | Motor |
|---------|-----------|-------|
| PDF (texto embebido) | `.pdf` | Unstructured |
| PDF (escaneado) | `.pdf` | Unstructured → **Mistral OCR** (fallback automático) |
| Word | `.docx`, `.doc` | Unstructured |
| Markdown | `.md` | Unstructured |
| Texto plano | `.txt` | Unstructured |
| HTML | `.html`, `.htm` | Unstructured |
| reStructuredText | `.rst` | Unstructured |

### Imágenes (modality: `image`)
| Formato | Extensión | Motor OCR |
|---------|-----------|-----------|
| JPEG | `.jpg`, `.jpeg` | **Mistral OCR** (texto literal) → fallback Gemini Vision |
| PNG | `.png` | **Mistral OCR** → fallback Gemini Vision |
| WebP | `.webp` | **Mistral OCR** → fallback Gemini Vision |
| TIFF | `.tiff`, `.tif` | **Mistral OCR** → fallback Gemini Vision |
| GIF | `.gif` | **Mistral OCR** → fallback Gemini Vision |
| BMP | `.bmp` | **Mistral OCR** → fallback Gemini Vision |

### Audio (modality: `audio`)
| Formato | Extensión | Motor |
|---------|-----------|-------|
| MP3 | `.mp3` | faster-whisper (local) |
| WAV | `.wav` | faster-whisper (local) |
| M4A | `.m4a` | faster-whisper (local) |
| FLAC | `.flac` | faster-whisper (local) |
| OGG | `.ogg` | faster-whisper (local) |
| Opus | `.opus` | faster-whisper (local) |
| WebM | `.webm` | faster-whisper (local) |

La transcripción es 100% local usando Whisper (sin enviar audio a APIs externas). Modelo por defecto: `base`.

### Datos estructurados (modality: `structured`)
| Formato | Extensión | Motor |
|---------|-----------|-------|
| CSV | `.csv` | Polars / Pandas |
| TSV | `.tsv` | Polars / Pandas |
| Excel | `.xlsx`, `.xls` | openpyxl + Polars |
| JSON | `.json` | stdlib |
| JSONL | `.jsonl`, `.ndjson` | stdlib |
| Parquet | `.parquet` | Polars |

### URLs web (modality: `web`)
Cualquier URL `http://` o `https://`. Se descarga el HTML, se extrae el texto limpio y se procesa como texto.

### Límite de tamaño
**50 MB por archivo** vía API.

---

## OCR automático

El sistema aplica OCR de forma **automática y transparente** sin configuración adicional.

### Selección de backend

| Condición | Backend activo |
|-----------|---------------|
| `MISTRAL_API_KEY` configurada | **Mistral OCR** (predeterminado) |
| Solo `GOOGLE_API_KEY` | **Gemini Vision** |
| Ninguna key | Solo metadatos EXIF (imágenes) |

Forzar un backend específico: `OCR_BACKEND=mistral | gemini | none`

### Para imágenes

```
imagen recibida
  └─ Mistral OCR → extrae texto literal (facturas, capturas, formularios)
       └─ si retorna vacío → Gemini Vision (descripción visual)
            └─ si no hay keys → solo metadatos EXIF
```

El resultado final siempre incluye los metadatos EXIF (cámara, GPS, fecha) si están presentes.

### Para PDFs escaneados

Los PDFs pasan primero por Unstructured. Si el texto extraído es inferior a `OCR_MIN_CHARS_PER_PAGE` (default: 50 caracteres), se detecta como PDF escaneado y se activa Mistral OCR automáticamente como fallback. El resultado son chunks por página en Markdown.

### El OCR es opcional

El sistema funciona sin OCR. Solo impacta en la calidad de extracción de:
- Imágenes con texto (facturas, capturas, formularios)
- PDFs escaneados (sin texto embebido)

PDFs normales, DOCX, HTML, audio y datos estructurados no usan OCR.

### Variables de entorno OCR

| Variable | Default | Descripción |
|----------|---------|-------------|
| `OCR_BACKEND` | auto | `mistral` / `gemini` / `none` |
| `MISTRAL_API_KEY` | — | Activa Mistral OCR |
| `MISTRAL_OCR_MODEL` | `mistral-ocr-latest` | Modelo Mistral a usar |
| `OCR_MIN_CHARS_PER_PAGE` | `50` | Umbral para detectar PDF escaneado |

---

## Almacenamiento de documentos (RustFS)

Los archivos originales se almacenan en **RustFS** (S3-compatible) en un bucket separado del bucket de Langfuse, bajo una estructura de paths que aísla completamente los datos por tenant y por usuario.

### Estructura de paths

```
s3://rag-documents/
  └── {tenant_id[:8]}/
        └── {user_id[:8]}/
              └── {uuid_filename}
```

Ejemplo: `s3://rag-documents/a1b2c3d4/e5f6a7b8/invoice_2024.pdf`

Esto garantiza que:
- Un tenant nunca accede a archivos de otro tenant
- Dentro de un tenant, los archivos de cada usuario están separados
- Los metadatos en PostgreSQL siempre referencian el `storage_key` completo

### Buckets

| Bucket | Variable | Uso |
|--------|----------|-----|
| `rag-documents` | `S3_RAG_BUCKET` | Documentos RAG subidos por usuarios |
| `langfuse` | `S3_BUCKET` | Eventos de observabilidad de Langfuse |

### Operaciones disponibles

| Operación | Descripción |
|-----------|-------------|
| `upload_bytes` / `upload_file` | Sube un documento a RustFS |
| `download_file` | Descarga a un path local |
| `delete_file` | Elimina el archivo (al borrar un documento) |
| `presigned_url` | Genera URL de descarga temporal (1h por defecto) |

### Sin RustFS configurado

Si `S3_ENDPOINT` no está configurado, el almacenamiento de archivos originales se omite silenciosamente. El pipeline de ingesta igual procesa y indexa el documento — solo no persiste el archivo original.

---

## Stack tecnológico

| Componente | Tecnología | Propósito |
|------------|-----------|-----------|
| API | FastAPI + uvicorn | REST API y autenticación |
| Cola de tareas | Celery + RabbitMQ | Ingesta asíncrona |
| Base vectorial | Qdrant (externo, Dokploy) | Búsqueda semántica densa |
| Búsqueda BM25 | Elasticsearch (local, Docker) | Búsqueda por palabras clave |
| Grafo de conocimiento | Neo4j (local, Docker) | Relaciones entre entidades |
| Caché semántica | Redis (externo, Dokploy) | Reutilización de queries similares |
| Base de datos | PostgreSQL (externo, Dokploy) | Metadatos, usuarios, tenants |
| Almacenamiento objetos | RustFS / S3 (externo, Dokploy) | Archivos originales por tenant/usuario |
| Embeddings | Google Gemini text-embedding-004 (768d) | Vectorización |
| OCR imágenes | Mistral OCR (primario) / Gemini Vision (fallback) | Extracción de texto en imágenes |
| OCR PDFs | Mistral OCR (fallback automático para escaneados) | PDFs sin texto embebido |
| LLM principal | Claude Haiku 4.5 / GPT-4o-mini / GPT-4o | Generación de respuestas |
| LLM local (PII) | Claude Haiku (configurable Ollama) | Queries con datos sensibles |
| Observabilidad | Langfuse (local, Docker) | Trazas de queries y costos |
| Evaluación | RAGAS | Métricas de calidad |
| Protocolo AI | MCP (Model Context Protocol) | Integración con Claude Desktop |

---

## Sistema de usuarios y planes

### Roles

| Rol | Permisos |
|-----|---------|
| `viewer` | Solo consultas y lectura de documentos |
| `editor` | Subir y eliminar documentos + consultas |
| `admin` | Todo + gestión de usuarios, API keys y estadísticas |

### Planes de uso (por tenant)

| Plan | Queries/día | Documentos/mes |
|------|------------|----------------|
| `free` | 100 | 10 |
| `starter` | 500 | 50 |
| `pro` | 1.000 | 100 |
| `enterprise` | Ilimitado | Ilimitado |

### Multi-tenancy

Cada tenant tiene datos completamente aislados:
- Colección Qdrant propia: `rag_{tenant_id[:8]}`
- Índice Elasticsearch propio: `rag_chunks_{tenant_id[:8]}`
- Filtro por `tenant_id` en Neo4j y PostgreSQL
- Prefijo en RustFS: `{tenant_id[:8]}/{user_id[:8]}/`

Un usuario de un tenant **nunca puede ver** documentos de otro tenant.

### Autenticación

**JWT** (para usuarios interactivos):
```http
POST /auth/login
{"email": "...", "password": "..."}
→ {"access_token": "...", "expires_in": 86400}
```

**API Key** (para integraciones y scripts):
```http
X-API-Key: <key>
```
Se genera desde `POST /admin/api-keys`.

---

## Pipeline de ingesta

```
Archivo / URL
    │
    ▼
1. Detección de modalidad (por extensión o URL)
    │
    ▼
2. Almacenamiento en RustFS  →  s3://rag-documents/{tenant}/{user}/{file}
    │
    ▼
3. Parser específico (Document / Image / Audio / Structured / Web)
    │  Document: Unstructured → Mistral OCR si PDF escaneado
    │  Image:    Mistral OCR → fallback Gemini Vision → + EXIF
    │  Audio:    faster-whisper (local)
    │  → Genera ParsedChunks con texto + metadatos básicos
    │
    ▼
4. Chunker semántico
    │  → min: 256 tokens · max: 1500 tokens · overlap: 20%
    │  → Respeta límites de párrafos y secciones
    │
    ▼
5. Extractor de metadatos (LLM)
    │  → Título, resumen, temas, idioma, tipo de doc, sentimiento
    │  → Entidades NER (personas, organizaciones, fechas)
    │
    ▼
6. Embed Worker (Gemini text-embedding-004, 768 dimensiones)
    │
    ▼
7. Indexación paralela:
    ├── Qdrant  (búsqueda vectorial densa)
    ├── Elasticsearch (búsqueda BM25 por palabras clave)
    └── Neo4j   (grafo de entidades y relaciones)
```

---

## Pipeline de retrieval

```
Query del usuario
    │
    ▼
1. Caché semántica (Redis)
    │  → Si existe query similar (cosine ≥ 0.87) → respuesta instantánea
    │
    ▼
2. Query Expansion
    │  → Genera 2-3 variaciones de la pregunta para mejor cobertura
    │
    ▼
3. Embedding de la query (Gemini 768d)
    │
    ▼
4. Hybrid Retrieval (en paralelo)
    ├── Qdrant: top-K por similitud vectorial
    ├── Elasticsearch: BM25 por palabras clave
    └── Neo4j: búsqueda en grafo de entidades
    │
    ▼
5. RRF Fusion (Reciprocal Rank Fusion)
    │  → Combina y re-ordena resultados de las 3 fuentes
    │
    ▼
6. Reranker (cross-encoder ms-marco-MiniLM-L-6-v2)
    │  → Reordena por relevancia semántica real
    │
    ▼
7. Context Compressor
    │  → Elimina fragmentos irrelevantes dentro de cada chunk
    │
    ▼
8. LLM Router → Respuesta final
```

---

## LLM Router

| Condición | Modelo | Costo estimado |
|-----------|--------|---------------|
| Query corta (< 30 tokens) | Claude Haiku 4.5 | $0.001 |
| Query media (30-100 tokens) | GPT-4o-mini | $0.006 |
| Query compleja o keywords avanzadas | GPT-4o | $0.030 |
| Query con PII detectado | Claude Haiku 4.5 | $0.001 |

**Circuit Breaker**: si un modelo falla 5 veces consecutivas, se desactiva 60 segundos y el router hace fallback al siguiente nivel.

**Hallucination Guard**: compara la respuesta con los chunks fuente usando cosine similarity. Si la similitud es < 0.65, reintenta la generación una vez.

**Detección de PII**: usa spaCy NER + regex (emails, teléfonos). Si detecta datos sensibles, fuerza el modelo más privado.

---

## API REST

```
POST   /auth/login              Obtener token JWT
POST   /auth/register           Registrar nuevo usuario

POST   /ingest/file             Subir archivo (multipart/form-data)
POST   /ingest/url              Ingestar URL pública
GET    /ingest/status/{job_id}  Estado de un job de ingesta
GET    /ingest/documents        Listar documentos del tenant
DELETE /ingest/{doc_id}         Eliminar documento y sus chunks

POST   /query                   Consulta RAG (respuesta completa)
POST   /query/stream            Consulta RAG con streaming SSE
GET    /query/history           Historial de queries del tenant

GET    /admin/stats             Estadísticas globales del tenant
GET    /admin/users             Listar usuarios
POST   /admin/users             Crear usuario
DELETE /admin/users/{id}        Eliminar usuario
GET    /admin/api-keys          Listar API keys
POST   /admin/api-keys          Generar API key
GET    /admin/costs             Costos LLM desglosados por día

GET    /health                  Estado de la API
GET    /docs                    Swagger UI interactivo
```

---

## Observabilidad

Langfuse registra automáticamente cada operación:

- **trace_query**: query completa con respuesta, costo, latencia y si fue cache hit
- **trace_llm_call**: modelo usado, tokens in/out, costo, latencia, si hubo retry
- **trace_retrieval**: chunks recuperados, score RRF máximo, flags de expand/rerank/compress
- **trace_ingest**: documento procesado, chunks generados, modalidad, costo, backend OCR usado

Acceso en `http://localhost:3000`.

---

## Evaluación con RAGAS

```bash
# Evaluación manual con dataset de prueba
cd ecosystem && python scripts/run_evaluation.py

# Benchmark completo (mide P50/P95 latencia)
cd ecosystem && python scripts/benchmark.py --dry-run
cd ecosystem && python scripts/benchmark.py --tenant-id <uuid>
```

Métricas evaluadas: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`.

El **quality monitor** corre cada 6 horas via Celery Beat y envía alerta si `faithfulness_avg < 0.70`.

---

## MCP Server

Permite que **Claude Desktop** se conecte directamente al ecosistema:

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "rag-ecosystem": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "B:/ecosistema_RAG/ecosystem"
    }
  }
}
```

**Herramientas disponibles:**

| Tool | Descripción |
|------|-------------|
| `search_knowledge` | Busca en la base de conocimiento |
| `ingest_document` | Sube e indexa un documento |
| `get_ingest_status` | Consulta el estado de un job |
| `get_tenant_stats` | Estadísticas del tenant (solo admin) |

---

## Instalación y arranque

### Requisitos previos
- Python 3.11+
- Docker + Docker Compose
- Git Bash (Windows) o terminal Unix

### Primera vez

```bash
cd ecosystem
./scripts/setup.sh
```

Este script hace todo automáticamente:
1. Crea `.env` desde `.env.example` y ofrece editarlo
2. Crea el venv e instala todas las dependencias Python
3. Levanta los servicios Docker (Elasticsearch, Neo4j, RabbitMQ, ClickHouse, Langfuse, Traefik)
4. Inicializa el schema de PostgreSQL
5. Verifica conexiones a todos los servicios

### Arranque diario

```bash
cd ecosystem
./scripts/dev.sh
```

Levanta en una sola terminal: Docker (si no está corriendo) + API FastAPI + Worker ingesta + Worker embeddings. `Ctrl+C` apaga todo limpiamente.

### Logs en tiempo real

```
ecosystem/logs/api.log
ecosystem/logs/celery_ingest.log
ecosystem/logs/celery_embed.log
```

### URLs locales

| Servicio | URL |
|----------|-----|
| API / Swagger | http://localhost:8000/docs |
| Langfuse UI | http://localhost:3000 |
| RabbitMQ UI | http://localhost:15672 |
| Neo4j UI | http://localhost:7474 |
| Traefik UI | http://localhost:8080 |

---

## Variables de entorno

### Servicios externos (Dokploy)

| Variable | Descripción | Requerida |
|----------|-------------|-----------|
| `POSTGRES_HOST` / `POSTGRES_PORT` | Host y puerto PostgreSQL | Sí |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | Credenciales PG | Sí |
| `QDRANT_URL` / `QDRANT_API_KEY` | URL y key de Qdrant | Sí |
| `REDIS_URL` / `REDIS_PASSWORD` | URL completa de Redis | Sí |
| `S3_ENDPOINT` / `S3_ACCESS_KEY` / `S3_SECRET_KEY` | RustFS para documentos RAG | Sí |
| `S3_RAG_BUCKET` | Bucket para documentos (default: `rag-documents`) | No |

### Servicios locales (Docker)

| Variable | Descripción |
|----------|-------------|
| `NEO4J_PASSWORD` | Contraseña Neo4j |
| `RABBITMQ_USER` / `RABBITMQ_PASSWORD` / `RABBITMQ_VHOST` | RabbitMQ |
| `CLICKHOUSE_PASSWORD` | ClickHouse |
| `ES_JAVA_OPTS` | Memoria Elasticsearch (dev: `-Xms512m -Xmx512m`) |

### LLM y OCR

| Variable | Descripción | Requerida |
|----------|-------------|-----------|
| `OPENAI_API_KEY` | Para GPT-4o y GPT-4o-mini | Sí |
| `ANTHROPIC_API_KEY` | Para Claude Haiku | Sí |
| `GOOGLE_API_KEY` | Para Gemini embeddings + Vision | Sí |
| `MISTRAL_API_KEY` | Para Mistral OCR (imágenes y PDFs escaneados) | No |
| `OCR_BACKEND` | `mistral` / `gemini` / `none` (auto-detectado) | No |
| `MISTRAL_OCR_MODEL` | Modelo OCR (default: `mistral-ocr-latest`) | No |
| `OCR_MIN_CHARS_PER_PAGE` | Umbral para detectar PDF escaneado (default: `50`) | No |

### Aplicación

| Variable | Descripción |
|----------|-------------|
| `APP_ENV` | `development` / `production` |
| `APP_DEBUG` | `true` / `false` |
| `APP_SECRET_KEY` | Clave para firmar JWTs |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Observabilidad |

---

## Producción y escalado

### Cambios obligatorios para producción

```bash
APP_ENV=production
APP_DEBUG=false
APP_SECRET_KEY=<random 32 bytes>
# Reemplazar todos los CHANGE_ME en .env
```

### Escalar para más requests

| Ajuste | Dónde |
|--------|-------|
| Más workers API | `uvicorn --workers 4` (1 por CPU core) |
| Más workers ingesta | `celery --concurrency=4` en worker ingest |
| Más workers embeddings | `celery --concurrency=4` en worker embed |
| Más cache hits | Bajar `SIMILARITY_THRESHOLD` a `0.85` en `cache/semantic_cache.py` |
| Menos costo LLM | Subir TTLs en `cache/ttl_classifier.py` |

### Límites de planes

Se configuran directamente en `api/middleware/rate_limit.py`:

```python
"free":       {"queries_day": 100,   "docs_month": 10}
"starter":    {"queries_day": 500,   "docs_month": 50}
"pro":        {"queries_day": 1_000, "docs_month": 100}
"enterprise": {"queries_day": 999_999, "docs_month": 999_999}
```

---

## Limitaciones conocidas

| Limitación | Detalle |
|------------|---------|
| Audio local only | Whisper corre localmente. Archivos > 1h requieren ajustar configuración |
| Sin OCR → imágenes degradan | Sin `MISTRAL_API_KEY` ni `GOOGLE_API_KEY`, las imágenes solo indexan metadatos EXIF |
| Sin RustFS | Si `S3_ENDPOINT` no está configurado, los archivos originales no se persisten (el índice sí se crea) |
| Primer query lento | El reranker (BERT) y hallucination guard (MiniLM) se cargan en memoria la primera vez. Tarda 20-40 segundos |
| Sin super admin global | El rol `admin` es por tenant. No hay un rol que vea todos los tenants |
| Workers en Windows | Celery en Windows requiere `--pool=solo`. En Linux usar `--pool=prefork` para mayor rendimiento |
| Tamaño máximo de archivo | 50 MB vía API |
| Rate limits en dev | Para resetear contadores: eliminar claves `rate:{tenant_id}:*` de Redis |
