"""RateLimitedProvider spaces calls with an injectable clock."""

import asyncio

import pytest

from aetheragents import Message, MockProvider, RateLimitedProvider


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now
        self.waits: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds


def _complete(provider):
    return asyncio.run(provider.complete([Message.user("hi")]))


def test_first_call_does_not_wait():
    clock = FakeClock()
    inner = MockProvider(["a", "b"])
    provider = RateLimitedProvider(
        inner, min_interval_s=0.5, clock=clock, sleep=clock.sleep
    )
    assert _complete(provider).content == "a"
    assert clock.waits == []


def test_second_call_waits_remaining_interval():
    clock = FakeClock()
    inner = MockProvider(["a", "b"])
    provider = RateLimitedProvider(
        inner, min_interval_s=0.5, clock=clock, sleep=clock.sleep
    )
    _complete(provider)
    assert _complete(provider).content == "b"
    assert clock.waits == [0.5]
    assert clock.now == 0.5


def test_skips_wait_when_interval_already_elapsed():
    clock = FakeClock()
    inner = MockProvider(["a", "b"])
    provider = RateLimitedProvider(
        inner, min_interval_s=0.25, clock=clock, sleep=clock.sleep
    )
    _complete(provider)
    clock.now = 10.0
    _complete(provider)
    assert clock.waits == []


def test_zero_interval_never_sleeps():
    clock = FakeClock()
    inner = MockProvider(["a", "b"])
    provider = RateLimitedProvider(
        inner, min_interval_s=0.0, clock=clock, sleep=clock.sleep
    )
    _complete(provider)
    _complete(provider)
    assert clock.waits == []
    assert provider.name == "rate_limit(mock)"


def test_rejects_negative_interval():
    with pytest.raises(ValueError):
        RateLimitedProvider(MockProvider(["x"]), min_interval_s=-0.1)
