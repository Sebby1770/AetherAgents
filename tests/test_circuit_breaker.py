"""CircuitBreakerProvider trips after consecutive failures."""

import asyncio
import threading
from typing import Any

import pytest

from aetheragents import (
    CircuitBreakerProvider,
    CircuitOpenError,
    LLMProvider,
    LLMResponse,
    Message,
    MockProvider,
    ProviderError,
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


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _complete(provider):
    return asyncio.run(provider.complete([Message.user("hi")]))


def test_opens_after_threshold_then_blocks():
    clock = FakeClock()
    inner = FlakyProvider(failures=10)
    breaker = CircuitBreakerProvider(
        inner, failure_threshold=3, reset_after=30.0, clock=clock
    )
    for _ in range(3):
        with pytest.raises(ProviderError):
            _complete(breaker)
    assert inner.attempts == 3
    assert breaker.is_open

    with pytest.raises(CircuitOpenError, match="Circuit breaker is open"):
        _complete(breaker)
    assert inner.attempts == 3  # inner not called while open


def test_resets_after_cooldown():
    clock = FakeClock()
    inner = FlakyProvider(failures=3)
    breaker = CircuitBreakerProvider(
        inner, failure_threshold=3, reset_after=10.0, clock=clock
    )
    for _ in range(3):
        with pytest.raises(ProviderError):
            _complete(breaker)

    clock.now = 10.0
    assert breaker.is_open is False
    resp = _complete(breaker)
    assert resp.content == "recovered"
    assert breaker.is_open is False


def test_success_resets_consecutive_failures():
    clock = FakeClock()
    inner = FlakyProvider(failures=1)
    breaker = CircuitBreakerProvider(
        inner, failure_threshold=3, reset_after=30.0, clock=clock
    )
    with pytest.raises(ProviderError):
        _complete(breaker)
    assert _complete(breaker).content == "recovered"

    # One isolated failure must not trip the breaker.
    inner.failures = inner.attempts + 1
    with pytest.raises(ProviderError):
        _complete(breaker)
    assert breaker.is_open is False


def test_wraps_mock_provider():
    breaker = CircuitBreakerProvider(MockProvider(["fine"]))
    assert _complete(breaker).content == "fine"
    assert breaker.name == "breaker(mock)"


def test_concurrent_failure_increments_are_not_lost():
    """Lock around increment/open so two threads cannot drop failure counts."""
    inner = FlakyProvider(failures=1000)
    breaker = CircuitBreakerProvider(
        inner, failure_threshold=1000, reset_after=60.0
    )
    per_thread = 10
    n_threads = 4

    def worker() -> None:
        for _ in range(per_thread):
            with pytest.raises(ProviderError):
                _complete(breaker)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert breaker._consecutive_failures == n_threads * per_thread
    assert breaker.is_open is False
    assert isinstance(breaker._lock, threading.Lock)


def test_record_failure_lock_serializes_open():
    breaker = CircuitBreakerProvider(
        MockProvider(["x"]), failure_threshold=8, reset_after=30.0
    )

    def worker() -> None:
        for _ in range(10):
            breaker._record_failure()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert breaker._consecutive_failures == 40
    assert breaker.is_open is True


def test_stream_fallback_respects_open_breaker():
    clock = FakeClock()
    inner = FlakyProvider(failures=10)
    breaker = CircuitBreakerProvider(
        inner, failure_threshold=1, reset_after=30.0, clock=clock
    )
    with pytest.raises(ProviderError):
        _complete(breaker)

    async def run():
        return [e async for e in breaker.stream([Message.user("hi")])]

    with pytest.raises(CircuitOpenError):
        asyncio.run(run())
