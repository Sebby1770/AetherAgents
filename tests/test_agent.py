import asyncio

import pytest

from aetheragents import (
    Agent,
    MaxStepsExceeded,
    MemoryManager,
    MockProvider,
    ToolRegistry,
    tool,
)


@tool()
def add(a: int, b: int) -> int:
    "Add two numbers."
    return a + b


def test_simple_answer():
    agent = Agent("a", MockProvider(["42"]))
    res = agent.run("what is the answer?")
    assert res.output == "42"
    assert res.steps[-1].type == "final"


def test_tool_calling_loop():
    provider = MockProvider([[("add", {"a": 2, "b": 3})], "The sum is 5."])
    agent = Agent("calc", provider, tools=[add])
    res = agent.run("add 2 and 3")
    assert res.output == "The sum is 5."
    results = [s for s in res.steps if s.type == "tool_result"]
    assert results[0].content == "5"
    assert results[0].ok


def test_max_steps_exceeded():
    provider = MockProvider(handler=lambda msgs: [("add", {"a": 1, "b": 1})])
    agent = Agent("loop", provider, tools=[add], max_steps=3)
    with pytest.raises(MaxStepsExceeded):
        agent.run("go forever")


def test_memory_is_recorded():
    mem = MemoryManager("a")
    agent = Agent("a", MockProvider(["done"]), memory=mem)
    agent.run("please remember this")
    roles = [e["role"] for e in mem.short_term]
    assert "user" in roles
    assert "assistant" in roles


def test_agent_as_tool_delegation():
    inner = Agent("translator", MockProvider(["bonjour"]))
    reg = ToolRegistry([inner.as_tool(description="translate to french")])
    res = asyncio.run(reg.execute("translator", {"input": "hello"}))
    assert res.content == "bonjour"


def test_usage_is_accumulated():
    agent = Agent("a", MockProvider(["a longer answer that costs some tokens"]))
    res = agent.run("hi")
    assert res.usage.total_tokens > 0


def test_run_inside_event_loop_raises():
    async def main():
        agent = Agent("a", MockProvider(["x"]))
        agent.run("hi")  # sync call inside a running loop -> error

    with pytest.raises(RuntimeError):
        asyncio.run(main())
