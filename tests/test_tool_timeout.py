"""Agent tool_timeout_s wraps sync and async tools."""

import asyncio
import time

from aetheragents import Agent, MockProvider, tool


@tool()
def slow_sync() -> str:
    "A slow synchronous tool."
    time.sleep(1.0)
    return "sync-done"


@tool()
async def slow_async() -> str:
    "A slow async tool."
    await asyncio.sleep(1.0)
    return "async-done"


@tool()
def quick() -> str:
    "A fast tool."
    return "quick-ok"


def test_sync_tool_timeout_returns_error():
    provider = MockProvider([[("slow_sync", {})], "continued after timeout"])
    agent = Agent(
        "a",
        provider,
        tools=[slow_sync],
        tool_timeout_s=0.05,
    )
    start = time.perf_counter()
    result = agent.run("go")
    elapsed = time.perf_counter() - start
    assert elapsed < 0.8
    timed = [s for s in result.steps if s.type == "tool_result"]
    assert timed[0].ok is False
    assert "timed out" in (timed[0].content or "")
    assert result.output == "continued after timeout"


def test_async_tool_timeout_returns_error():
    provider = MockProvider([[("slow_async", {})], "continued after timeout"])
    agent = Agent(
        "a",
        provider,
        tools=[slow_async],
        tool_timeout_s=0.05,
    )
    result = agent.run("go")
    timed = [s for s in result.steps if s.type == "tool_result"]
    assert timed[0].ok is False
    assert "timed out" in (timed[0].content or "")
    assert result.output == "continued after timeout"


def test_fast_tool_unaffected_by_timeout():
    provider = MockProvider([[("quick", {})], "done"])
    agent = Agent("a", provider, tools=[quick], tool_timeout_s=1.0)
    result = agent.run("go")
    results = [s for s in result.steps if s.type == "tool_result"]
    assert results[0].ok is True
    assert results[0].content == "quick-ok"
    assert result.output == "done"


def test_no_timeout_by_default():
    provider = MockProvider([[("quick", {})], "done"])
    agent = Agent("a", provider, tools=[quick])
    assert agent.tool_timeout_s is None
    result = agent.run("go")
    assert result.output == "done"
