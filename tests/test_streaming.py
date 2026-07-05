import asyncio
from typing import Any

from aetheragents import Agent, LLMProvider, LLMResponse, Message, MockProvider


def collect(agen):
    async def _run():
        return [e async for e in agen]

    return asyncio.run(_run())


def test_mock_stream_deltas_reconstruct_content():
    p = MockProvider(["hello streaming world"])
    events = collect(p.stream([Message.user("hi")]))
    deltas = [e.delta for e in events if e.type == "delta"]
    assert "".join(deltas) == "hello streaming world"
    assert events[-1].type == "done"
    assert events[-1].response.content == "hello streaming world"


def test_base_class_fallback_stream():
    class PlainProvider(LLMProvider):
        async def complete(self, messages, **kwargs: Any) -> LLMResponse:
            return LLMResponse(content="whole answer", finish_reason="stop")

    events = collect(PlainProvider().stream([Message.user("x")]))
    assert [e.type for e in events] == ["delta", "done"]
    assert events[0].delta == "whole answer"


def test_agent_astream_yields_deltas_steps_and_result():
    def add(a: int, b: int) -> int:
        "Add."
        return a + b

    provider = MockProvider([[("add", {"a": 2, "b": 3})], "The sum is 5."])
    agent = Agent("calc", provider, tools=[add])
    events = collect(agent.astream("add 2 and 3"))

    types = [e.type for e in events]
    assert types.count("result") == 1 and types[-1] == "result"
    step_types = [e.step.type for e in events if e.type == "step"]
    assert step_types == ["tool_call", "tool_result", "final"]
    deltas = "".join(e.delta for e in events if e.type == "delta")
    assert deltas == "The sum is 5."
    assert events[-1].result.output == "The sum is 5."


def test_arun_still_works_via_stream():
    agent = Agent("a", MockProvider(["plain answer"]))
    assert agent.run("hi").output == "plain answer"
