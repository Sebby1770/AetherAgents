"""Circuit breaker wrapper for any :class:`LLMProvider`.

After ``failure_threshold`` consecutive failures the breaker opens and
subsequent calls raise :class:`CircuitOpenError` until ``reset_after``
seconds have elapsed. A successful call closes the breaker.

State transitions (failure increment, open, reset) and the open-check are
protected by a ``threading.Lock`` so concurrent ``complete`` calls cannot
lose counts or tear the open/closed flag. The lock is **not** held during
the inner provider call.

Inject ``clock`` (a zero-arg callable returning seconds) to make the
cooldown deterministic in tests. No extra dependencies.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from ..core.errors import CircuitOpenError
from ..core.messages import Message
from .base import LLMProvider, LLMResponse


class CircuitBreakerProvider(LLMProvider):
    """Fail fast after a streak of provider errors, then cool down.

    Counter increments, open, and reset run under ``_lock`` so two threads
    flipping failures cannot drop updates. Half-open probes after
    ``reset_after`` are not serialized through the inner provider.
    """

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
        self._lock = threading.Lock()

    def _is_open_unlocked(self) -> bool:
        if self._opened_at is None:
            return False
        return (self._clock() - self._opened_at) < self.reset_after

    @property
    def is_open(self) -> bool:
        """True when the breaker is currently rejecting calls."""
        with self._lock:
            return self._is_open_unlocked()

    def _raise_if_open(self) -> None:
        with self._lock:
            if self._opened_at is None:
                return
            elapsed = self._clock() - self._opened_at
            if elapsed >= self.reset_after:
                # Half-open: allow a probe through to the inner provider.
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
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._opened_at = self._clock()

    def _record_success(self) -> None:
        with self._lock:
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
