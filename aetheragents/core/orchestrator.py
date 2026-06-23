"""Multi-agent orchestration patterns.

Three composable patterns cover most teams:

* **sequential** - a pipeline; each agent's output feeds the next.
* **parallel**   - fan out the same task to many agents, gather all results.
* **route**      - a selector picks the single best agent for the task.

For free-form delegation, expose an agent with :meth:`Agent.as_tool` and give it
to a "manager" agent's tool registry.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from .agent import Agent, AgentResult
from .errors import OrchestrationError

Selector = Callable[[str, dict[str, Agent]], str]


class Orchestrator:
    """Coordinates a team of named agents."""

    def __init__(self, agents: list[Agent] | None = None) -> None:
        self.agents: dict[str, Agent] = {}
        for agent in agents or []:
            self.add(agent)

    def add(self, agent: Agent) -> Agent:
        if agent.name in self.agents:
            raise OrchestrationError(f"Duplicate agent name: {agent.name!r}")
        self.agents[agent.name] = agent
        return agent

    def __getitem__(self, name: str) -> Agent:
        return self.agents[name]

    def _resolve(self, names: list[str] | None) -> list[Agent]:
        if names is None:
            return list(self.agents.values())
        try:
            return [self.agents[n] for n in names]
        except KeyError as exc:
            raise OrchestrationError(f"Unknown agent: {exc}") from exc

    async def sequential(
        self, prompt: str, order: list[str] | None = None
    ) -> list[AgentResult]:
        """Run agents in a pipeline, feeding each output to the next.

        Returns every step's result; the last one is the final output.
        """
        agents = self._resolve(order)
        if not agents:
            raise OrchestrationError("No agents to run.")
        results: list[AgentResult] = []
        message = prompt
        for agent in agents:
            result = await agent.arun(message)
            results.append(result)
            message = result.output or ""
        return results

    async def parallel(
        self, prompt: str, names: list[str] | None = None
    ) -> dict[str, AgentResult]:
        """Run the same prompt across several agents concurrently."""
        agents = self._resolve(names)
        if not agents:
            raise OrchestrationError("No agents to run.")
        results = await asyncio.gather(*(a.arun(prompt) for a in agents))
        return {a.name: r for a, r in zip(agents, results, strict=False)}

    async def route(self, prompt: str, *, selector: Selector) -> AgentResult:
        """Pick one agent via ``selector`` and run it."""
        if not self.agents:
            raise OrchestrationError("No agents registered.")
        choice = selector(prompt, self.agents)
        if choice not in self.agents:
            raise OrchestrationError(f"Selector chose unknown agent: {choice!r}")
        return await self.agents[choice].arun(prompt)


def keyword_router(mapping: dict[str, str], default: str | None = None) -> Selector:
    """Build a simple keyword-based :data:`Selector`.

    ``mapping`` maps a substring to an agent name; the first match wins.
    """

    def selector(prompt: str, agents: dict[str, Agent]) -> str:
        lowered = prompt.lower()
        for keyword, agent_name in mapping.items():
            if keyword.lower() in lowered:
                return agent_name
        if default is not None:
            return default
        return next(iter(agents))

    return selector
