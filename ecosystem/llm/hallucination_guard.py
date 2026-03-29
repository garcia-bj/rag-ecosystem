"""Hallucination guard — semantic faithfulness check, retry on low score."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List

logger = logging.getLogger(__name__)

FAITHFULNESS_THRESHOLD = 0.65
_MAX_RETRIES = 1  # keep cost bounded

_RESTRICTIVE_SUFFIX = (
    "\n\nIMPORTANT: Base your answer STRICTLY on the provided context. "
    "Do not add any information not explicitly stated in the context. "
    "If the context does not contain the answer, say so."
)


@dataclass
class GuardResult:
    score: float
    passed: bool
    retried: bool = False
    final_response: str = ""


class HallucinationGuard:
    """Verify that an LLM response is grounded in the retrieved chunks.

    Uses cosine similarity between the response embedding and the concatenated
    context embedding (via sentence-transformers all-MiniLM-L6-v2).
    Score < threshold triggers one retry with a more restrictive prompt.

    Usage::

        guard = HallucinationGuard()
        result = await guard.check(
            query, response, chunks, router_call_fn, prompt
        )
    """

    def __init__(self, threshold: float = FAITHFULNESS_THRESHOLD) -> None:
        self._threshold = threshold
        self._model: Any = None  # lazy — SentenceTransformer

    # ── Public ────────────────────────────────────────────────────────────────

    def score(self, response: str, context: str) -> float:
        """Return faithfulness score (0-1) between response and context."""
        try:
            model = self._get_model()
            if model is None:
                return 1.0  # fail open — no model available
            from sentence_transformers import util
            import asyncio
            loop = asyncio.get_event_loop()
            embs = model.encode([response, context], convert_to_tensor=True)
            sim = float(util.cos_sim(embs[0], embs[1]))
            # Convert from [-1,1] to [0,1]
            return (sim + 1) / 2
        except Exception as exc:
            logger.warning("HallucinationGuard score failed: %s", exc)
            return 1.0  # fail open

    async def check(
        self,
        response: str,
        chunks: List[Any],
        router_fn,
        query: str,
        prompt: str,
    ) -> GuardResult:
        """Check faithfulness; retry once with stricter prompt if below threshold.

        Args:
            response:  The initial LLM response text.
            chunks:    Retrieved chunks used as context.
            router_fn: Callable ``async (prompt) → str`` to generate a new response.
            query:     Original user query (for logging).
            prompt:    The prompt used to generate the response.
        """
        context = "\n\n".join(getattr(c, "text", str(c)) for c in chunks)
        sc = self.score(response, context)

        if sc >= self._threshold:
            logger.debug("HallucinationGuard PASS score=%.3f query=%r", sc, query[:60])
            return GuardResult(score=sc, passed=True, final_response=response)

        logger.info(
            "HallucinationGuard FAIL score=%.3f — retrying with restrictive prompt. query=%r",
            sc, query[:60],
        )

        # One retry with a stricter prompt
        try:
            strict_prompt = prompt + _RESTRICTIVE_SUFFIX
            retry_response = await router_fn(strict_prompt)
            retry_score = self.score(retry_response, context)
            logger.info("HallucinationGuard retry score=%.3f", retry_score)
            return GuardResult(
                score=retry_score,
                passed=retry_score >= self._threshold,
                retried=True,
                final_response=retry_response,
            )
        except Exception as exc:
            logger.warning("HallucinationGuard retry failed: %s", exc)
            return GuardResult(score=sc, passed=False, retried=True, final_response=response)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as exc:
                logger.warning("SentenceTransformer unavailable: %s", exc)
                self._model = False
        return self._model if self._model is not False else None
