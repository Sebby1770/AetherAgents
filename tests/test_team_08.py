"""0.8 team features: fallback, blackboard, tool cache, consensus, workflow."""

import asyncio

import pytest

from aetheragents import (
    Agent,
    Blackboard,
    BlackboardError,
    FallbackProvider,
    MockProvider,
    OrchestrationError,
    Orchestrator,
    ProviderError,
    ToolCache,
    tool,
)
from aetheragents.llm.base import LLMProvider, LLMResponse


class _Boom(LLMProvider):
    name = "boom"

    async def complete(self, messages, *, tools=None, model=None, temperature=None, **kwargs):
        raise ProviderError("down")


class _Ok(LLMProvider):
    name = "ok"

    async def complete(self, messages, *, tools=None, model=None, temperature=None, **kwargs):
        return LLMResponse(content="served", model="ok-1")


class _Bug(LLMProvider):
    name = "bug"

    async def complete(self, messages, *, tools=None, model=None, temperature=None, **kwargs):
        raise RuntimeError("programming error")


def test_fallback_uses_the_next_provider():
    provider = FallbackProvider([_Boom(), _Ok()])
    response = asyncio.run(provider.complete([]))
    assert response.content == "served"
    assert provider.last_provider == "ok"
    assert provider.attempts == ["boom", "ok"]


def test_fallback_reports_every_failure():
    provider = FallbackProvider([_Boom(), _Boom()])
    with pytest.raises(ProviderError, match="all fallback providers failed"):
        asyncio.run(provider.complete([]))
    assert provider.attempts == ["boom", "boom"]


def test_fallback_does_not_swallow_other_errors():
    provider = FallbackProvider([_Bug(), _Ok()])
    with pytest.raises(RuntimeError, match="programming error"):
        asyncio.run(provider.complete([]))


def test_fallback_rejects_an_empty_list():
    with pytest.raises(ValueError):
        FallbackProvider([])


def test_blackboard_round_trip_and_isolation():
    board = Blackboard()
    board.put("plan", {"steps": ["a", "b"]})
    stored = board.get("plan")
    stored["steps"].append("c")
    assert board.get("plan") == {"steps": ["a", "b"]}
    assert board.get("missing", "nope") == "nope"
    assert "plan" in board.snapshot()


def test_blackboard_rejects_bad_keys_and_values():
    board = Blackboard()
    with pytest.raises(BlackboardError):
        board.put("has space", "x")
    with pytest.raises(BlackboardError):
        board.put("ok", float("nan"))
    with pytest.raises(BlackboardError):
        board.put("ok", object())


def test_blackboard_tools_on_an_agent():
    board = Blackboard()
    agent = Agent(
        "scribe",
        MockProvider(
            [
                [("blackboard_write", {"key": "note", "value": "hello"})],
                [("blackboard_read", {"key": "note"})],
                "done",
            ]
        ),
    )
    board.bind(agent)
    result = asyncio.run(agent.arun("remember hello"))
    assert result.output == "done"
    assert board.get("note") == "hello"


def test_tool_cache_skips_repeat_calls():
    calls = {"n": 0}

    @tool()
    def add(a: int, b: int) -> int:
        "Add two numbers."
        calls["n"] += 1
        return a + b

    cache = ToolCache()
    cached = cache.wrap(add)
    assert cached.func(a=2, b=2) == 4
    assert cached.func(a=2, b=2) == 4
    assert cached.func(a=1, b=2) == 3
    assert calls["n"] == 2
    assert cache.hits == 1
    assert cache.misses == 2


def test_consensus_majority_and_tie_break():
    orch = Orchestrator(
        [
            Agent("a", MockProvider(["yes"])),
            Agent("b", MockProvider(["yes"])),
            Agent("c", MockProvider(["no"])),
        ]
    )
    out = asyncio.run(orch.consensus("ship it?"))
    assert out.winner == "a"
    assert out.output == "yes"
    assert out.judged is False
    assert out.agreement == pytest.approx(2 / 3)
    assert out.votes == {"a": "yes", "b": "yes", "c": "no"}

    tied = Orchestrator(
        [
            Agent("red", MockProvider(["red"])),
            Agent("blue", MockProvider(["blue"])),
        ]
    )
    tie = asyncio.run(tied.consensus("color?"))
    assert tie.winner == "red"
    assert tie.agreement == pytest.approx(0.5)


def test_consensus_judge_can_override_and_bad_verdict_falls_back():
    orch = Orchestrator(
        [
            Agent("a", MockProvider(["same"])),
            Agent("b", MockProvider(["same"])),
            Agent("c", MockProvider(["other"])),
            Agent("judge", MockProvider(["PICK: c\nmore careful"])),
        ]
    )
    judged = asyncio.run(orch.consensus("q", names=["a", "b", "c"], judge="judge"))
    assert judged.judged is True
    assert judged.winner == "c"
    assert judged.output == "other"
    assert judged.agreement == pytest.approx(1 / 3)

    fallback_team = Orchestrator(
        [
            Agent("a", MockProvider(["same"])),
            Agent("b", MockProvider(["same"])),
            Agent("c", MockProvider(["other"])),
            Agent("judge", MockProvider(["not a verdict"])),
        ]
    )
    fallback = asyncio.run(fallback_team.consensus("q", names=["a", "b", "c"], judge="judge"))
    assert fallback.judged is False
    assert fallback.winner == "a"


def test_consensus_requires_agents():
    with pytest.raises(OrchestrationError):
        asyncio.run(Orchestrator().consensus("q"))


def test_workflow_feeds_output_and_saves_to_blackboard():
    board = Blackboard()
    orch = Orchestrator(
        [
            Agent("draft", MockProvider(["first pass"])),
            Agent("edit", MockProvider(["polished"])),
        ]
    )
    out = asyncio.run(
        orch.workflow(
            "write a note",
            [
                {"agent": "draft", "save_as": "draft"},
                {"agent": "edit", "save_as": "final"},
            ],
            blackboard=board,
        )
    )
    assert out.output == "polished"
    assert out.saved == {"draft": "first pass", "final": "polished"}
    assert board.get("final") == "polished"
    edit_messages = orch["edit"].provider.calls[0]
    assert any("first pass" in (message.content or "") for message in edit_messages)


def test_workflow_rejects_empty_and_bad_keys():
    orch = Orchestrator([Agent("draft", MockProvider(["x"]))])
    with pytest.raises(OrchestrationError):
        asyncio.run(orch.workflow("q", []))
    with pytest.raises(OrchestrationError, match="cannot save"):
        asyncio.run(orch.workflow("q", [{"agent": "draft", "save_as": "bad key"}], blackboard=Blackboard()))


def test_workflow_step_model_rejects_unknown_shape():
    orch = Orchestrator([Agent("draft", MockProvider(["x"]))])
    with pytest.raises(OrchestrationError, match="invalid workflow step"):
        asyncio.run(orch.workflow("q", ["draft"]))
