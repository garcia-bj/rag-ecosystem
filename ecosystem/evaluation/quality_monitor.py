"""Quality degradation monitor — Celery beat task, runs every 6 hours.

Reads the most recent evaluation scores from PostgreSQL.  If the
faithfulness average drops below 0.70 over the last 100 evaluations,
a warning is logged and (optionally) an email alert is sent.

Standalone usage::

    import asyncio
    from evaluation.quality_monitor import check_quality_degradation
    result = asyncio.run(check_quality_degradation())
    print(result)

Register Celery beat task at worker startup::

    from evaluation.quality_monitor import register_celery_task
    register_celery_task()
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

FAITHFULNESS_THRESHOLD = 0.70
MONITOR_SAMPLE_SIZE    = 100


# ── Public ────────────────────────────────────────────────────────────────────

async def check_quality_degradation(tenant_id: str | None = None) -> dict:
    """Check whether faithfulness has degraded below the threshold.

    Args:
        tenant_id: Scope check to one tenant. ``None`` → all tenants.

    Returns:
        {
            "ok":               bool,
            "faithfulness_avg": float | None,
            "sample_count":     int,
            "alert_sent":       bool,
        }
    """
    scores = await _get_recent_scores(tenant_id, limit=MONITOR_SAMPLE_SIZE)

    if not scores:
        return {"ok": True, "faithfulness_avg": None, "sample_count": 0, "alert_sent": False}

    avg = sum(s.get("faithfulness", 1.0) for s in scores) / len(scores)
    ok  = avg >= FAITHFULNESS_THRESHOLD
    alert_sent = False

    if not ok:
        tenant_label = f" for tenant {tenant_id}" if tenant_id else ""
        msg = (
            f"[RAG Quality Alert] Faithfulness degraded to {avg:.3f} "
            f"(threshold={FAITHFULNESS_THRESHOLD}) "
            f"over last {len(scores)} evaluations{tenant_label}"
        )
        logger.warning(msg)
        alert_sent = await _send_alert(msg)

    return {
        "ok":               ok,
        "faithfulness_avg": round(avg, 4),
        "sample_count":     len(scores),
        "alert_sent":       alert_sent,
    }


# ── Internal ──────────────────────────────────────────────────────────────────

async def _get_recent_scores(tenant_id: str | None, limit: int) -> list[dict]:
    try:
        from core.database import get_connection
        conn = await get_connection()
        try:
            if tenant_id:
                rows = await conn.fetch(
                    """
                    SELECT scores FROM evaluations
                    WHERE tenant_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    tenant_id, limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT scores FROM evaluations
                    ORDER BY created_at DESC
                    LIMIT $1
                    """,
                    limit,
                )
            return [json.loads(r["scores"]) for r in rows]
        finally:
            await conn.close()
    except Exception as exc:
        logger.debug("_get_recent_scores failed: %s", exc)
        return []


async def _send_alert(message: str) -> bool:
    """Log critical alert; send email if SMTP is configured."""
    logger.critical("QUALITY ALERT: %s", message)

    smtp_host  = os.getenv("SMTP_HOST", "")
    alert_to   = os.getenv("ALERT_EMAIL", "")
    if not smtp_host or not alert_to:
        return False

    try:
        import smtplib
        from email.mime.text import MIMEText

        msg           = MIMEText(message)
        msg["Subject"] = "[RAG] Quality Degradation Alert"
        msg["From"]    = os.getenv("SMTP_FROM", "rag-alerts@localhost")
        msg["To"]      = alert_to

        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.sendmail(msg["From"], [alert_to], msg.as_string())
        logger.info("Alert email sent to %s", alert_to)
        return True
    except Exception as exc:
        logger.debug("Email alert failed (non-critical): %s", exc)
        return False


# ── Celery periodic task ──────────────────────────────────────────────────────

def register_celery_task():
    """Register ``run_quality_check`` as a Celery beat task (every 6 hours).

    Call this once at worker startup.  Safe to call multiple times.
    """
    try:
        import asyncio
        from ingestion.workers.celery_app import app as celery_app

        @celery_app.task(
            name="evaluation.quality_monitor.run_quality_check",
            queue="default",
            bind=False,
        )
        def run_quality_check():
            result = asyncio.run(check_quality_degradation())
            logger.info("Periodic quality check: %s", result)
            return result

        # Beat schedule — 6-hour interval
        beat_schedule = getattr(celery_app.conf, "beat_schedule", {}) or {}
        beat_schedule["rag-quality-monitor-6h"] = {
            "task":     "evaluation.quality_monitor.run_quality_check",
            "schedule": 6 * 3600,
        }
        celery_app.conf.beat_schedule = beat_schedule

        logger.info("Quality monitor Celery task registered (every 6 h)")
        return run_quality_check
    except Exception as exc:
        logger.debug("Could not register Celery beat task: %s", exc)
        return None
