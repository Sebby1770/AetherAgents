"""Multi-agent orchestration: a small research team.

Demonstrates the three orchestration patterns plus agent-as-tool delegation,
all offline with MockProvider::

    python examples/research_team.py
"""

import asyncio

from aetheragents import (
    Agent,
    MockProvider,
    Orchestrator,
    ToolRegistry,
    keyword_router,
)


def build_team() -> Orchestrator:
    researcher = Agent(
        "researcher",
        MockProvider(
            handler=lambda msgs: "Findings: AetherAgents is tiny, typed and provider-agnostic."
        ),
        instructions="Gather concise facts about the topic.",
    )
    writer = Agent(
        "writer",
        MockProvider(
            handler=lambda msgs: "# AetherAgents\nA tiny, typed, provider-agnostic agent framework."
        ),
        instructions="Turn findings into a short article.",
    )
    support = Agent(
        "support",
        MockProvider(handler=lambda msgs: "Have you tried turning it off and on again?"),
        instructions="Answer support questions.",
    )
    return Orchestrator([researcher, writer, support])


async def main() -> None:
    team = build_team()

    print("== Sequential pipeline (researcher -> writer) ==")
    results = await team.sequential("Write about AetherAgents", order=["researcher", "writer"])
    print(results[-1].output, "\n")

    print("== Parallel fan-out ==")
    out = await team.parallel("Summarise the project", names=["researcher", "writer"])
    for name, res in out.items():
        print(f"[{name}] {res.output}")
    print()

    print("== Routing ==")
    router = keyword_router({"bug": "support", "write": "writer"}, default="researcher")
    routed = await team.route("I found a bug, please help", selector=router)
    print(f"routed to support -> {routed.output}\n")

    print("== Agent-as-tool delegation ==")
    manager = Agent(
        "manager",
        MockProvider(
            [
                [("support", {"input": "my app crashes"})],
                "I asked support and they suggested a restart.",
            ]
        ),
        tools=ToolRegistry([team["support"].as_tool(description="Escalate to support")]),
    )
    print((await manager.arun("A user reports a crash")).output)


if __name__ == "__main__":
    asyncio.run(main())
