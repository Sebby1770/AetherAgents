"""Circuit breaker wrapper for any :class:`LLMProvider`.

After ``failure_threshold`` consecutive failures the breaker opens and
subsequent calls raise :class:`CircuitOpenError` until ``reset_after``
seconds have elapsed. A successful call closes the breaker.

Inject ``clock`` (a zero-arg callable returning seconds) to make the
cooldown deterministic in tests. No extra dependencies.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ..core.errors import CircuitOpenError
from ..core.messages import Message
from .base import LLMProvider, LLMResponse


class CircuitBreakerProvider(LLMProvider):
    """Fail fast after a streak of provider errors, then cool down."""

    def __init__(
        self,
        inner: LLMProvider,
        *,
        failure_threshold: int = 3,
        reset_after: float = 30.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if reset_after < 0:
            raise ValueError("reset_after must be >= 0")
        self.inner = inner
        self.failure_threshold = failure_threshold
        self.reset_after = reset_after
        self._clock = clock or time.monotonic
        self.name = f"breaker({inner.name})"
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        """True when the breaker is currently rejecting calls."""
        if self._opened_at is None:
            return False
        return (self._clock() - self._opened_at) < self.reset_after

    def _raise_if_open(self) -> None:
        if self._opened_at is None:
            return
        elapsed = self._clock() - self._opened_at
        if elapsed >= self.reset_after:
            # Half-open: allow one probe through to the inner provider.
            return
        remaining = self.reset_after - elapsed
        raise CircuitOpenError(
            f"Circuit breaker is open for '{self.inner.name}' "
            f"(threshold={self.failure_threshold}); "
            f"retry after {remaining:.1f}s",
            provider=self.inner.name,
            reset_after=self.reset_after,
        )

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._opened_at = self._clock()

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self._raise_if_open()
        try:
            response = await self.inner.complete(
                messages, tools=tools, model=model, temperature=temperature, **kwargs
            )
        except CircuitOpenError:
            raise
        except Exception:
            self._record_failure()
            raise
        self._record_success()
        return response
