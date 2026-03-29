"""
Phase 1 — Verificación de conexiones a todos los servicios.

Uso:
    pip install psycopg2-binary redis qdrant-client neo4j pika httpx python-dotenv
    python scripts/verify_connections.py
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv


@dataclass
class CheckResult:
    service: str
    healthy: bool
    detail: str


async def check_postgres(env: dict) -> CheckResult:
    """Verifica conexión a PostgreSQL externo."""
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=env.get("POSTGRES_HOST", "localhost"),
            port=int(env.get("POSTGRES_PORT", "5432")),
            user=env.get("POSTGRES_USER", "rag"),
            password=env.get("POSTGRES_PASSWORD", ""),
            dbname=env.get("POSTGRES_DB", "rag_metadata"),
            connect_timeout=5,
        )
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        cur.close()
        conn.close()
        return CheckResult("PostgreSQL", True, version[:60])
    except Exception as e:
        return CheckResult("PostgreSQL", False, str(e)[:80])


async def check_qdrant(env: dict) -> CheckResult:
    """Verifica conexión a Qdrant externo."""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=env.get("QDRANT_URL", "http://localhost:6333"),
            api_key=env.get("QDRANT_API_KEY"),
            timeout=5,
        )
        collections = client.get_collections()
        return CheckResult(
            "Qdrant", True, f"{len(collections.collections)} collections"
        )
    except Exception as e:
        return CheckResult("Qdrant", False, str(e)[:80])


async def check_redis(env: dict) -> CheckResult:
    """Verifica conexión a Redis externo."""
    try:
        import redis as redis_lib

        url = env.get("REDIS_URL", "redis://localhost:6379")
        client = redis_lib.from_url(url, socket_timeout=5)
        info = client.info("server")
        client.close()
        return CheckResult("Redis", True, f"v{info['redis_version']}")
    except Exception as e:
        return CheckResult("Redis", False, str(e)[:80])


async def check_neo4j(env: dict) -> CheckResult:
    """Verifica conexión a Neo4j local."""
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=(env.get("NEO4J_USER", "neo4j"), env.get("NEO4J_PASSWORD", "")),
        )
        with driver.session() as session:
            result = session.run("RETURN 1 AS n")
            result.single()
        driver.close()
        return CheckResult("Neo4j", True, "bolt://localhost:7687")
    except Exception as e:
        return CheckResult("Neo4j", False, str(e)[:80])


async def check_elasticsearch(env: dict) -> CheckResult:
    """Verifica conexión a Elasticsearch local."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("http://localhost:9200/_cluster/health")
            data = resp.json()
            status = data.get("status", "unknown")
            return CheckResult("Elasticsearch", status in ("green", "yellow"), f"cluster status: {status}")
    except Exception as e:
        return CheckResult("Elasticsearch", False, str(e)[:80])


async def check_rabbitmq(env: dict) -> CheckResult:
    """Verifica conexión a RabbitMQ local."""
    try:
        import pika

        credentials = pika.PlainCredentials(
            env.get("RABBITMQ_USER", "rag_rabbit"),
            env.get("RABBITMQ_PASSWORD", ""),
        )
        params = pika.ConnectionParameters(
            host="localhost",
            port=5672,
            virtual_host=env.get("RABBITMQ_VHOST", "rag"),
            credentials=credentials,
            connection_attempts=1,
            socket_timeout=5,
        )
        conn = pika.BlockingConnection(params)
        conn.close()
        return CheckResult("RabbitMQ", True, "amqp://localhost:5672")
    except Exception as e:
        return CheckResult("RabbitMQ", False, str(e)[:80])


async def check_clickhouse(env: dict) -> CheckResult:
    """Verifica conexión a ClickHouse local."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("http://localhost:8123/ping")
            return CheckResult("ClickHouse", resp.status_code == 200, "http://localhost:8123")
    except Exception as e:
        return CheckResult("ClickHouse", False, str(e)[:80])


async def check_s3(env: dict) -> CheckResult:
    """Verifica conexión a RustFS/S3 externo."""
    try:
        endpoint = env.get("S3_ENDPOINT", "http://localhost:9000")
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{endpoint}/minio/health/live")
            return CheckResult("RustFS/S3", resp.status_code == 200, endpoint)
    except Exception as e:
        return CheckResult("RustFS/S3", False, str(e)[:80])


async def check_langfuse(env: dict) -> CheckResult:
    """Verifica conexión a Langfuse local."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("http://localhost:3000/api/public/health")
            return CheckResult("Langfuse", resp.status_code == 200, "http://localhost:3000")
    except Exception as e:
        return CheckResult("Langfuse", False, str(e)[:80])


async def check_traefik(env: dict) -> CheckResult:
    """Verifica conexión a Traefik local."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("http://localhost:8080/api/overview")
            return CheckResult("Traefik", resp.status_code == 200, "http://localhost:8080")
    except Exception as e:
        return CheckResult("Traefik", False, str(e)[:80])


async def main() -> int:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    env = dict(os.environ)

    checks = [
        check_postgres,
        check_qdrant,
        check_redis,
        check_neo4j,
        check_elasticsearch,
        check_rabbitmq,
        check_clickhouse,
        check_s3,
        check_langfuse,
        check_traefik,
    ]

    results = await asyncio.gather(
        *[check(env) for check in checks], return_exceptions=True
    )

    print("\n" + "=" * 65)
    print(f"  {'Service':<18} {'Status':<10} {'Detail'}")
    print("=" * 65)

    all_ok = True
    for r in results:
        if isinstance(r, Exception):
            print(f"  {'???':<18} {'ERROR':<10} {r}")
            all_ok = False
        else:
            icon = "OK" if r.healthy else "FAIL"
            print(f"  {r.service:<18} {icon:<10} {r.detail}")
            if not r.healthy:
                all_ok = False

    print("=" * 65)
    print(f"  Result: {'ALL SERVICES OK' if all_ok else 'SOME SERVICES FAILED'}")
    print("=" * 65 + "\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
