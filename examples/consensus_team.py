"""Offline consensus team: three agents vote, a judge can overrule.

Run: python examples/consensus_team.py
"""

import asyncio

from aetheragents import Agent, Blackboard, MockProvider, Orchestrator, ToolCache, tool


@tool()
def word_count(text: str) -> int:
    "Count words in text."
    return len(text.split())


def main() -> None:
    cache = ToolCache()
    board = Blackboard()
    orch = Orchestrator(
        [
            Agent("alpha", MockProvider(["Ship the patch."])),
            Agent("beta", MockProvider(["Ship the patch."])),
            Agent("gamma", MockProvider(["Hold for tests."])),
            Agent("judge", MockProvider(["PICK: gamma"])),
            Agent(
                "scribe",
                MockProvider([[("word_count", {"text": "Ship the patch."})], "three words"]),
                tools=cache.wrap_all([word_count]),
            ),
        ]
    )
    vote = asyncio.run(
        orch.consensus("Should we ship?", names=["alpha", "beta", "gamma"], judge="judge")
    )
    print(f"winner={vote.winner} judged={vote.judged} agreement={vote.agreement:.2f}")
    print(vote.output)

    note = asyncio.run(
        orch.workflow(
            "Summarise the decision",
            [
                {"agent": "scribe", "prompt": "Count words in: Ship the patch.", "save_as": "count"},
            ],
            blackboard=board,
        )
    )
    print(note.output)
    print("blackboard", board.snapshot())
    print(f"tool cache hits={cache.hits} misses={cache.misses}")


if __name__ == "__main__":
    main()
