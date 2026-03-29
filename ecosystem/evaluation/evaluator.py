"""RAGAS evaluation — evaluate_batch() and DB persistence.

Runs faithfulness, answer_relevancy, context_precision, and (optionally)
context_recall over a set of RAG query/response/context triples.

Results are persisted to the ``evaluations`` table and can be read by
the quality monitor.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


async def evaluate_batch(
    queries: List[str],
    responses: List[str],
    contexts: List[List[str]],
    ground_truths: Optional[List[str]] = None,
    tenant_id: str = "",
    save_to_db: bool = True,
) -> dict:
    """Run RAGAS evaluation on a batch of query/response pairs.

    Args:
        queries:       User questions.
        responses:     RAG-generated answers (one per question).
        contexts:      Retrieved context chunks (list-of-lists, one per question).
        ground_truths: Expected answers — enables context_recall when provided.
        tenant_id:     Used for DB persistence (skipped when empty).
        save_to_db:    Persist scores to PostgreSQL.

    Returns:
        dict with keys: faithfulness, answer_relevancy, context_precision,
        [context_recall].
    """
    # Normalise: contexts must be List[List[str]]
    normalised_contexts = [
        [c] if isinstance(c, str) else list(c) for c in contexts
    ]

    try:
        scores = await _run_ragas(queries, responses, normalised_contexts, ground_truths)
    except Exception as exc:
        logger.warning("evaluate_batch: _run_ragas raised unexpectedly: %s", exc)
        scores = {
            "faithfulness":      0.0,
            "answer_relevancy":  0.0,
            "context_precision": 0.0,
            "error":             str(exc),
        }

    if save_to_db and tenant_id:
        await _save_evaluation(tenant_id, scores, len(queries))

    return scores


async def _run_ragas(
    queries: List[str],
    responses: List[str],
    contexts: List[List[str]],
    ground_truths: Optional[List[str]],
) -> dict:
    """Execute RAGAS evaluation; handles both 0.1.x and 0.2.x APIs."""
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
        )

        data: dict = {
            "question": queries,
            "answer":   responses,
            "contexts": contexts,
        }
        metrics = [faithfulness, answer_relevancy, context_precision]

        if ground_truths is not None:
            try:
                from ragas.metrics import context_recall
                data["ground_truth"] = ground_truths
                metrics.append(context_recall)
            except ImportError:
                logger.debug("context_recall not available in this ragas version")

        dataset = Dataset.from_dict(data)
        result  = ragas_evaluate(dataset, metrics=metrics)

        scores: dict = {
            "faithfulness":      _safe_float(result, "faithfulness"),
            "answer_relevancy":  _safe_float(result, "answer_relevancy"),
            "context_precision": _safe_float(result, "context_precision"),
        }
        if ground_truths is not None and "context_recall" in result:
            scores["context_recall"] = _safe_float(result, "context_recall")

        return scores

    except Exception as exc:
        logger.warning("RAGAS evaluation failed: %s", exc)
        # Return placeholder so callers don't crash
        return {
            "faithfulness":      0.0,
            "answer_relevancy":  0.0,
            "context_precision": 0.0,
            "error":             str(exc),
        }


def _safe_float(result: object, key: str) -> float:
    try:
        val = result[key]  # type: ignore[index]
        return float(val) if val is not None else 0.0
    except Exception:
        return 0.0


async def _save_evaluation(tenant_id: str, scores: dict, sample_size: int) -> None:
    try:
        from core.database import get_connection
        conn = await get_connection()
        try:
            await conn.execute(
                """
                INSERT INTO evaluations (tenant_id, sample_size, scores)
                VALUES ($1, $2, $3::jsonb)
                """,
                tenant_id,
                sample_size,
                json.dumps(scores),
            )
            logger.info(
                "Saved evaluation: tenant=%s n=%d faithfulness=%.3f",
                tenant_id[:8], sample_size, scores.get("faithfulness", 0),
            )
        finally:
            await conn.close()
    except Exception as exc:
        logger.warning("Failed to persist evaluation: %s", exc)
