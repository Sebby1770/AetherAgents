"""Orchestrator.handoff — transfer work from one agent to another."""

import asyncio

import pytest

from aetheragents import Agent, MockProvider, OrchestrationError, Orchestrator


def test_handoff_feeds_output_to_next_agent():
    orch = Orchestrator(
        [
            Agent("researcher", MockProvider(["research notes"])),
            Agent("writer", MockProvider(["article from notes"])),
        ]
    )
    result = asyncio.run(
        orch.handoff("Write about X", from_name="researcher", to_name="writer")
    )
    assert result.from_name == "researcher"
    assert result.to_name == "writer"
    assert result.from_result.output == "research notes"
    assert result.to_result.output == "article from notes"
    assert result.output == "article from notes"
    assert result.result is result.to_result

    writer_msgs = orch["writer"].provider.calls[0]
    blob = " ".join(m.content or "" for m in writer_msgs)
    assert "handoff" in blob.lower()
    assert "research notes" in blob
    assert "Write about X" in blob
    roles = [m.role.value for m in writer_msgs]
    assert "system" in roles


def test_handoff_unknown_agent():
    orch = Orchestrator([Agent("writer", MockProvider(["x"]))])
    with pytest.raises(OrchestrationError):
        asyncio.run(orch.handoff("x", from_name="ghost", to_name="writer"))
    with pytest.raises(OrchestrationError):
        asyncio.run(orch.handoff("x", from_name="writer", to_name="ghost"))
