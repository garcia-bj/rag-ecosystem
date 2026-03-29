"""Celery application — RabbitMQ broker, Redis result backend."""

from __future__ import annotations

import os
from celery import Celery
from kombu import Exchange, Queue

# ── Connection URLs ────────────────────────────────────────────────────────────

def _broker_url() -> str:
    explicit = os.getenv("CELERY_BROKER_URL")
    if explicit:
        return explicit
    user = os.getenv("RABBITMQ_USER", "rag_rabbit")
    password = os.getenv("RABBITMQ_PASSWORD", "")
    host = os.getenv("RABBITMQ_HOST", "localhost")
    port = os.getenv("RABBITMQ_PORT", "5672")
    vhost = os.getenv("RABBITMQ_VHOST", "rag")
    return f"amqp://{user}:{password}@{host}:{port}/{vhost}"


def _backend_url() -> str:
    explicit = os.getenv("CELERY_RESULT_BACKEND")
    if explicit:
        return explicit
    return os.getenv("REDIS_URL", "redis://localhost:6379/0")


# ── Queues ─────────────────────────────────────────────────────────────────────

_default_exchange = Exchange("rag", type="direct")

QUEUES = (
    Queue("ingest",    _default_exchange, routing_key="ingest"),
    Queue("embed",     _default_exchange, routing_key="embed"),
    Queue("index",     _default_exchange, routing_key="index"),
    Queue("default",   _default_exchange, routing_key="default"),
)

# ── App ────────────────────────────────────────────────────────────────────────

app = Celery("rag_ingestion")

app.conf.update(
    broker_url=_broker_url(),
    result_backend=_backend_url(),

    # Serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Reliability
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,

    # Queues
    task_queues=QUEUES,
    task_default_queue="default",
    task_default_exchange="rag",
    task_default_routing_key="default",

    task_routes={
        "ingestion.workers.ingest_task.ingest_document": {
            "queue": "ingest",
            "routing_key": "ingest",
        },
    },

    # Results
    result_expires=3600,

    # Timezone
    timezone="UTC",
    enable_utc=True,
)

# Auto-discover tasks from ingestion.workers
app.autodiscover_tasks(["ingestion.workers"], related_name="ingest_task")
