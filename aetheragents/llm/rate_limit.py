"""Minimum-interval wrapper for any :class:`LLMProvider`.

``RateLimitedProvider`` sleeps between ``complete`` calls so an inner
provider is not hammered faster than ``min_interval_s``. Inject ``clock``
and ``sleep`` to make waits deterministic in tests. No extra dependencies.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any

from ..core.messages import Message
from .base import LLMProvider, LLMResponse

SleepFn = Callable[[float], Awaitable[None]]


class RateLimitedProvider(LLMProvider):
    """Sleep between calls so ``inner`` is invoked at most once per interval.

    ``clock`` is a zero-arg callable returning seconds (defaults to
    ``time.monotonic``). ``sleep`` is an awaitable delay (defaults to
    ``asyncio.sleep``). A fake clock whose ``sleep`` both records waits and
    advances time makes tests fully offline.
    """

    def __init__(
        self,
        inner: LLMProvider,
        min_interval_s: float = 0.0,
        *,
        clock: Callable[[], float] | None = None,
        sleep: SleepFn | None = None,
    ) -> None:
        if min_interval_s < 0:
            raise ValueError("min_interval_s must be >= 0")
        self.inner = inner
        self.min_interval_s = min_interval_s
        self._clock = clock or time.monotonic
        self._sleep: SleepFn = sleep or asyncio.sleep
        self.name = f"rate_limit({inner.name})"
        self._last_call_at: float | None = None
        self._lock = threading.Lock()

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        wait_for = 0.0
        with self._lock:
            now = self._clock()
            if self._last_call_at is not None and self.min_interval_s > 0:
                remaining = self.min_interval_s - (now - self._last_call_at)
                if remaining > 0:
                    wait_for = remaining
            # Reserve this slot so concurrent callers space out.
            self._last_call_at = now + wait_for
        if wait_for > 0:
            await self._sleep(wait_for)
        return await self.inner.complete(
            messages, tools=tools, model=model, temperature=temperature, **kwargs
        )
