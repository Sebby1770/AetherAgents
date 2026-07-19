"""Debate and map-reduce orchestration patterns."""

import asyncio

import pytest

from aetheragents import Agent, MockProvider, OrchestrationError, Orchestrator


def test_debate_two_agents_two_rounds():
    orch = Orchestrator(
        [
            Agent("alice", MockProvider(["alice-r1", "alice-r2"])),
            Agent("bob", MockProvider(["bob-r1", "bob-r2"])),
        ]
    )
    out = asyncio.run(orch.debate("Is the sky blue?", agents=["alice", "bob"], rounds=2))
    assert out["rounds"] == 2
    assert len(out["transcript"]) == 4  # 2 agents * 2 rounds
    assert out["transcript"][0]["agent"] == "alice"
    assert out["transcript"][0]["output"] == "alice-r1"
    assert out["transcript"][1]["agent"] == "bob"
    assert out["synthesis"] is None
    assert out["output"] == "bob-r2"  # last speaker


def test_debate_with_synthesizer():
    orch = Orchestrator(
        [
            Agent("pro", MockProvider(["pro says yes"])),
            Agent("con", MockProvider(["con says no"])),
            Agent("judge", MockProvider(["final: yes, carefully"])),
        ]
    )
    out = asyncio.run(
        orch.debate(
            "Should we ship?",
            agents=["pro", "con"],
            rounds=1,
            synthesizer="judge",
        )
    )
    assert out["synthesis"] is not None
    assert out["synthesis"].output == "final: yes, carefully"
    assert out["output"] == "final: yes, carefully"
    assert len(out["results"]) == 3  # pro + con + judge


def test_debate_invalid_rounds():
    orch = Orchestrator([Agent("a", MockProvider(["x"]))])
    with pytest.raises(OrchestrationError):
        asyncio.run(orch.debate("q", rounds=0))


def test_map_reduce_workers_then_reducer():
    orch = Orchestrator(
        [
            Agent("w1", MockProvider(["facts from w1"])),
            Agent("w2", MockProvider(["facts from w2"])),
            Agent("reducer", MockProvider(["combined answer"])),
        ]
    )
    out = asyncio.run(
        orch.map_reduce("research X", worker_names=["w1", "w2"], reducer_name="reducer")
    )
    assert out["workers"]["w1"].output == "facts from w1"
    assert out["workers"]["w2"].output == "facts from w2"
    assert out["reducer"].output == "combined answer"
    assert out["output"] == "combined answer"

    # Reducer must have seen both worker outputs in its prompt.
    reducer_msgs = orch["reducer"].provider.calls[0]
    contents = " ".join(m.content or "" for m in reducer_msgs)
    assert "facts from w1" in contents
    assert "facts from w2" in contents
    assert "research X" in contents


def test_map_reduce_empty_workers():
    orch = Orchestrator([Agent("r", MockProvider(["x"]))])
    with pytest.raises(OrchestrationError):
        asyncio.run(orch.map_reduce("q", worker_names=[], reducer_name="r"))


def test_map_reduce_unknown_reducer():
    orch = Orchestrator([Agent("w1", MockProvider(["x"]))])
    with pytest.raises(OrchestrationError):
        asyncio.run(orch.map_reduce("q", worker_names=["w1"], reducer_name="ghost"))
