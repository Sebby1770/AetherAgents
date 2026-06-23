import asyncio

import pytest

from aetheragents import (
    Agent,
    MockProvider,
    OrchestrationError,
    Orchestrator,
    keyword_router,
)


def test_sequential_pipeline():
    orch = Orchestrator(
        [
            Agent("a", MockProvider(["step-a-out"])),
            Agent("b", MockProvider(["step-b-out"])),
        ]
    )
    results = asyncio.run(orch.sequential("start"))
    assert [r.output for r in results] == ["step-a-out", "step-b-out"]


def test_parallel_fanout():
    orch = Orchestrator(
        [
            Agent("a", MockProvider(["A"])),
            Agent("b", MockProvider(["B"])),
        ]
    )
    out = asyncio.run(orch.parallel("x"))
    assert out["a"].output == "A"
    assert out["b"].output == "B"


def test_route_with_keyword_router():
    orch = Orchestrator(
        [
            Agent("math", MockProvider(handler=lambda m: "mathy")),
            Agent("prose", MockProvider(handler=lambda m: "prosey")),
        ]
    )
    router = keyword_router({"calculate": "math"}, default="prose")
    assert asyncio.run(orch.route("please calculate 2+2", selector=router)).output == "mathy"
    assert asyncio.run(orch.route("write me a poem", selector=router)).output == "prosey"


def test_duplicate_agent_name():
    orch = Orchestrator([Agent("a", MockProvider(["x"]))])
    with pytest.raises(OrchestrationError):
        orch.add(Agent("a", MockProvider(["y"])))


def test_unknown_agent_in_order():
    orch = Orchestrator([Agent("a", MockProvider(["x"]))])
    with pytest.raises(OrchestrationError):
        asyncio.run(orch.sequential("go", order=["a", "ghost"]))
