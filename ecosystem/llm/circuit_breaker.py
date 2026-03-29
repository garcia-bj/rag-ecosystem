"""Circuit breaker per LLM provider — CLOSED → OPEN → HALF_OPEN."""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """Raised when the circuit breaker is in OPEN state."""


class CircuitBreaker:
    """Per-provider circuit breaker.

    States::

        CLOSED    Normal operation; failures increment counter.
        OPEN      All calls rejected immediately; started by counter ≥ threshold.
        HALF_OPEN One probe call allowed; success → CLOSED, failure → OPEN.

    Usage::

        cb = CircuitBreaker("anthropic")
        result = await cb.call(some_async_fn())
    """

    def __init__(
        self,
        provider: str,
        threshold_failures: int = 5,
        recovery_time: float = 60.0,
    ) -> None:
        self.provider = provider
        self._threshold = threshold_failures
        self._recovery_time = recovery_time
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._last_failure: float = 0.0
        self._lock = asyncio.Lock()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def failures(self) -> int:
        return self._failures

    # ── Public ────────────────────────────────────────────────────────────────

    async def call(self, coro) -> Any:
        """Execute *coro* through the circuit breaker."""
        await self._check_open()

        try:
            result = await coro
            await self._on_success()
            return result
        except CircuitBreakerOpen:
            raise
        except Exception as exc:
            await self._on_failure(exc)
            raise

    def reset(self) -> None:
        """Manually reset to CLOSED (for testing / ops recovery)."""
        self._state    = CircuitState.CLOSED
        self._failures = 0
        self._last_failure = 0.0

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _check_open(self) -> None:
        async with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._last_failure
                if elapsed >= self._recovery_time:
                    self._state = CircuitState.HALF_OPEN
                    logger.info("CircuitBreaker[%s]: OPEN → HALF_OPEN", self.provider)
                else:
                    raise CircuitBreakerOpen(
                        f"{self.provider} circuit OPEN — "
                        f"retry in {self._recovery_time - elapsed:.0f}s"
                    )

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state    = CircuitState.CLOSED
                self._failures = 0
                logger.info("CircuitBreaker[%s]: HALF_OPEN → CLOSED", self.provider)

    async def _on_failure(self, exc: Exception) -> None:
        async with self._lock:
            self._failures += 1
            self._last_failure = time.monotonic()
            if self._state == CircuitState.HALF_OPEN or self._failures >= self._threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    "CircuitBreaker[%s]: → OPEN after %d failures. exc=%s",
                    self.provider, self._failures, exc,
                )
