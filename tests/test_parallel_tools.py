"""Parallel tool execution within a single agent step."""

import asyncio
import time

from aetheragents import Agent, MockProvider, tool


@tool()
async def slow_a() -> str:
    "Slow tool A."
    await asyncio.sleep(0.05)
    return "A"


@tool()
async def slow_b() -> str:
    "Slow tool B."
    await asyncio.sleep(0.05)
    return "B"


def test_parallel_tools_run_concurrently():
    provider = MockProvider(
        [[("slow_a", {}), ("slow_b", {})], "both done"]
    )
    agent = Agent(
        "p",
        provider,
        tools=[slow_a, slow_b],
        parallel_tools=True,
    )
    start = time.perf_counter()
    result = agent.run("go")
    elapsed = time.perf_counter() - start
    # Serial would be ~0.10s; parallel should finish closer to 0.05s.
    assert elapsed < 0.12
    results = [s for s in result.steps if s.type == "tool_result"]
    assert {s.content for s in results} == {"A", "B"}
    assert result.output == "both done"


def test_serial_tools_default():
    provider = MockProvider(
        [[("slow_a", {}), ("slow_b", {})], "serial done"]
    )
    agent = Agent("s", provider, tools=[slow_a, slow_b], parallel_tools=False)
    result = agent.run("go")
    results = [s for s in result.steps if s.type == "tool_result"]
    assert [s.content for s in results] == ["A", "B"]
    assert result.output == "serial done"
