"""Automatic retries with exponential backoff for any provider.

Wrap a provider to make transient failures (rate limits, network blips)
self-healing::

    provider = RetryingProvider(LiteLLMProvider("gpt-4o"), max_retries=3)

Only :meth:`complete` retries directly. :meth:`stream` intentionally uses the
base-class fallback (which calls ``complete`` and therefore inherits retries) -
resuming a partially-consumed native stream after a mid-stream failure would
silently duplicate output.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..core.errors import ProviderError
from ..core.messages import Message
from .base import LLMProvider, LLMResponse


class RetryingProvider(LLMProvider):
    """Wraps another provider and retries failed completions."""

    def __init__(
        self,
        inner: LLMProvider,
        *,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 8.0,
        retry_on: tuple[type[Exception], ...] = (ProviderError,),
    ) -> None:
        self.inner = inner
        self.max_retries = max(0, max_retries)
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retry_on = retry_on
        self.name = f"retry({inner.name})"
        #: Number of retry attempts performed over this provider's lifetime.
        self.retries_used = 0

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self.inner.complete(
                    messages, tools=tools, model=model, temperature=temperature, **kwargs
                )
            except self.retry_on as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                self.retries_used += 1
                delay = min(self.base_delay * (2**attempt), self.max_delay)
                await asyncio.sleep(delay)
        raise ProviderError(
            f"{self.inner.name} failed after {self.max_retries + 1} attempts: {last_exc}"
        ) from last_exc
