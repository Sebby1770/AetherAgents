"""0.9: conditional workflows, team budgets, quorum, and result diffs."""

import asyncio

import pytest

from aetheragents import (
    Agent,
    Blackboard,
    MockProvider,
    OrchestrationError,
    Orchestrator,
    diff_results,
    register_model_cost,
    tool,
)


def test_workflow_skips_until_the_blackboard_matches():
    board = Blackboard()
    board.put("status", "hold")
    orch = Orchestrator(
        [
            Agent("draft", MockProvider(["note"])),
            Agent("ship", MockProvider(["sent"])),
        ]
    )
    held = asyncio.run(
        orch.workflow(
            "go",
            [
                {"agent": "draft", "save_as": "note"},
                {"agent": "ship", "when": "status", "equals": "go", "prompt": "Ship {note}"},
            ],
            blackboard=board,
        )
    )
    assert held.skipped == ["ship"]
    assert held.output == "note"
    assert len(orch["ship"].provider.calls) == 0

    board.put("status", "go")
    orch["draft"].provider._queue.append("note")
    sent = asyncio.run(
        orch.workflow(
            "go",
            [
                {"agent": "draft", "save_as": "note"},
                {"agent": "ship", "when": "status", "equals": "go", "prompt": "Ship {note}"},
            ],
            blackboard=board,
        )
    )
    assert sent.skipped == []
    assert sent.output == "sent"
    blob = " ".join(message.content or "" for message in orch["ship"].provider.calls[0])
    assert "Ship note" in blob


def test_workflow_stops_at_the_team_budget():
    register_model_cost("budget-test-model", 0.0, 1_000_000.0)
    orch = Orchestrator(
        [
            Agent("first", MockProvider(["hello"], model="budget-test-model")),
            Agent("second", MockProvider(["later"], model="budget-test-model")),
        ]
    )
    out = asyncio.run(
        orch.workflow("q", [{"agent": "first"}, {"agent": "second"}], max_cost_usd=0.5)
    )
    assert out.stopped == "budget"
    assert [step.agent for step in out.steps] == ["first"]
    assert out.spent_usd >= 0.5
    assert len(orch["second"].provider.calls) == 0


def test_workflow_rejects_a_negative_budget():
    orch = Orchestrator([Agent("draft", MockProvider(["x"]))])
    with pytest.raises(OrchestrationError):
        asyncio.run(orch.workflow("q", [{"agent": "draft"}], max_cost_usd=-1))


def test_consensus_quorum_threshold():
    orch = Orchestrator(
        [
            Agent("a", MockProvider(["yes"])),
            Agent("b", MockProvider(["yes"])),
            Agent("c", MockProvider(["no"])),
        ]
    )
    low = asyncio.run(orch.consensus("q", min_agreement=0.9))
    assert low.winner == "a"
    assert low.quorum is False

    orch["a"].provider._queue.append("yes")
    orch["b"].provider._queue.append("yes")
    orch["c"].provider._queue.append("no")
    enough = asyncio.run(orch.consensus("q", min_agreement=0.5))
    assert enough.quorum is True

    with pytest.raises(OrchestrationError):
        asyncio.run(orch.consensus("q", min_agreement=1.5))


def test_diff_results_reports_output_tools_and_cost():
    @tool()
    def ping() -> str:
        "Ping."
        return "pong"

    left_agent = Agent("left", MockProvider(["same"]), tools=[ping])
    right_agent = Agent(
        "right",
        MockProvider([[("ping", {})], "different"]),
        tools=[ping],
    )
    left = asyncio.run(left_agent.arun("q"))
    right = asyncio.run(right_agent.arun("q"))
    delta = diff_results(left, right)
    assert delta["same_output"] is False
    assert delta["tools_added"] == ["ping"]
    assert delta["tools_removed"] == []
    assert delta["left_agent"] == "left"
    assert delta["cost_delta_usd"] is None
