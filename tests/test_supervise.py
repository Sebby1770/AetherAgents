"""Orchestrator.supervise — worker/critic revision loop."""

import asyncio

import pytest

from aetheragents import Agent, MockProvider, OrchestrationError, Orchestrator


def test_supervise_accepts_on_round_2():
    orch = Orchestrator(
        [
            Agent("worker", MockProvider(["draft v1", "draft v2"])),
            Agent("critic", MockProvider(["REVISE: add numbers", "ACCEPT"])),
        ]
    )
    out = asyncio.run(
        orch.supervise("Write a summary", worker="worker", critic="critic", max_rounds=2)
    )
    assert out.accepted is True
    assert out.rounds == 2
    assert out.final.output == "draft v2"
    assert out.output == "draft v2"

    second = orch["worker"].provider.calls[1]
    blob = " ".join(m.content or "" for m in second)
    assert "add numbers" in blob
    assert "draft v1" in blob
    assert "Write a summary" in blob


def test_supervise_accepts_first_round():
    orch = Orchestrator(
        [
            Agent("worker", MockProvider(["good enough"])),
            Agent("critic", MockProvider(["ACCEPT"])),
        ]
    )
    out = asyncio.run(orch.supervise("q", "worker", "critic", max_rounds=2))
    assert out.accepted is True
    assert out.rounds == 1
    assert out.final.output == "good enough"
    assert len(orch["worker"].provider.calls) == 1


def test_supervise_never_accepts():
    orch = Orchestrator(
        [
            Agent("worker", MockProvider(["v1", "v2"])),
            Agent("critic", MockProvider(["REVISE: more", "REVISE: still more"])),
        ]
    )
    out = asyncio.run(orch.supervise("q", "worker", "critic", max_rounds=2))
    assert out.accepted is False
    assert out.rounds == 2
    assert out.final.output == "v2"


def test_supervise_accepts_agent_instances():
    worker = Agent("w", MockProvider(["ok"]))
    critic = Agent("c", MockProvider(["ACCEPT"]))
    orch = Orchestrator()
    out = asyncio.run(orch.supervise("q", worker, critic, max_rounds=1))
    assert out.accepted is True
    assert out.final.agent == "w"


def test_supervise_invalid_rounds():
    orch = Orchestrator(
        [
            Agent("w", MockProvider(["x"])),
            Agent("c", MockProvider(["ACCEPT"])),
        ]
    )
    with pytest.raises(OrchestrationError):
        asyncio.run(orch.supervise("q", "w", "c", max_rounds=0))


def test_supervise_unknown_agent():
    orch = Orchestrator([Agent("w", MockProvider(["x"]))])
    with pytest.raises(OrchestrationError, match="critic"):
        asyncio.run(orch.supervise("q", "w", "ghost"))
