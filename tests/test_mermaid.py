"""Orchestrator.to_mermaid diagram export."""

from aetheragents import Agent, MockProvider, Orchestrator


def test_mermaid_empty_team():
    orch = Orchestrator()
    diagram = orch.to_mermaid()
    assert diagram.startswith("flowchart")
    assert "No agents" in diagram


def test_mermaid_lists_agents():
    orch = Orchestrator(
        [
            Agent("researcher", MockProvider(["x"]), instructions="Find facts"),
            Agent("writer", MockProvider(["y"]), instructions="Write prose"),
        ]
    )
    diagram = orch.to_mermaid()
    assert "flowchart LR" in diagram
    assert "researcher" in diagram
    assert "writer" in diagram
    assert "Find facts" in diagram
    assert "---" in diagram  # edge between agents


def test_mermaid_sanitises_names():
    orch = Orchestrator([Agent("data-scientist!", MockProvider(["x"]))])
    diagram = orch.to_mermaid()
    # Node id must be alphanumeric/underscore; label still has the real name.
    assert "data_scientist_" in diagram or "data-scientist!" in diagram
    assert "data-scientist!" in diagram
