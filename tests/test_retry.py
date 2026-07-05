import asyncio
from typing import Any

import pytest

from aetheragents import (
    LLMProvider,
    LLMResponse,
    Message,
    MockProvider,
    ProviderError,
    RetryingProvider,
)


class FlakyProvider(LLMProvider):
    """Fails ``failures`` times, then succeeds."""

    name = "flaky"

    def __init__(self, failures: int, exc: Exception | None = None):
        self.failures = failures
        self.attempts = 0
        self.exc = exc or ProviderError("boom")

    async def complete(self, messages, **kwargs: Any) -> LLMResponse:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise self.exc
        return LLMResponse(content="recovered", finish_reason="stop")


def test_retries_until_success():
    flaky = FlakyProvider(failures=2)
    provider = RetryingProvider(flaky, max_retries=3, base_delay=0.001)
    resp = asyncio.run(provider.complete([Message.user("hi")]))
    assert resp.content == "recovered"
    assert flaky.attempts == 3
    assert provider.retries_used == 2


def test_retries_exhausted_raises_provider_error():
    flaky = FlakyProvider(failures=10)
    provider = RetryingProvider(flaky, max_retries=2, base_delay=0.001)
    with pytest.raises(ProviderError, match="after 3 attempts"):
        asyncio.run(provider.complete([Message.user("hi")]))
    assert flaky.attempts == 3


def test_non_retryable_exception_passes_through():
    flaky = FlakyProvider(failures=5, exc=ValueError("not transient"))
    provider = RetryingProvider(flaky, max_retries=3, base_delay=0.001)
    with pytest.raises(ValueError):
        asyncio.run(provider.complete([Message.user("hi")]))
    assert flaky.attempts == 1  # no retries for unknown exception types


def test_stream_fallback_inherits_retries():
    flaky = FlakyProvider(failures=1)
    provider = RetryingProvider(flaky, max_retries=2, base_delay=0.001)

    async def run():
        return [e async for e in provider.stream([Message.user("hi")])]

    events = asyncio.run(run())
    assert events[-1].response.content == "recovered"
    assert flaky.attempts == 2


def test_wraps_mock_provider_transparently():
    provider = RetryingProvider(MockProvider(["fine"]), base_delay=0.001)
    resp = asyncio.run(provider.complete([Message.user("hi")]))
    assert resp.content == "fine"
    assert provider.name == "retry(mock)"
