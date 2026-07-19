"""Multi-agent orchestration patterns.

Composable patterns cover most teams:

* **sequential** - a pipeline; each agent's output feeds the next.
* **parallel**   - fan out the same task to many agents, gather all results.
* **route**      - a selector picks the single best agent for the task.
* **debate**     - multi-round discussion where agents respond to each other.
* **map_reduce** - parallel workers then a reducer synthesises their outputs.

For free-form delegation, expose an agent with :meth:`Agent.as_tool` and give it
to a "manager" agent's tool registry.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

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

    def _resolve_one(self, name: str) -> Agent:
        if name not in self.agents:
            raise OrchestrationError(f"Unknown agent: {name!r}")
        return self.agents[name]

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

    async def debate(
        self,
        prompt: str,
        *,
        agents: list[str] | list[Agent] | None = None,
        rounds: int = 2,
        synthesizer: str | Agent | None = None,
    ) -> dict[str, Any]:
        """Multi-round debate: agents respond to each other, then optional synthesis.

        Each round, every participant sees the original prompt plus the running
        transcript of prior turns, then contributes a new reply. After
        ``rounds`` complete, an optional synthesizer agent produces a final
        answer from the full transcript.

        Returns a dict with:

        * ``transcript`` - list of ``{round, agent, output}`` entries
        * ``rounds`` - number of rounds run
        * ``results`` - flat list of every :class:`AgentResult`
        * ``synthesis`` - synthesizer :class:`AgentResult` or ``None``
        * ``output`` - final text (synthesis if present, else last speaker)
        """
        if rounds < 1:
            raise OrchestrationError("debate rounds must be >= 1")

        participants = self._resolve_agents(agents)
        if not participants:
            raise OrchestrationError("No agents to debate.")

        synth_agent: Agent | None = None
        if synthesizer is not None:
            if isinstance(synthesizer, Agent):
                synth_agent = synthesizer
            else:
                synth_agent = self._resolve_one(synthesizer)

        transcript: list[dict[str, Any]] = []
        all_results: list[AgentResult] = []
        history_lines: list[str] = []

        for r in range(1, rounds + 1):
            for agent in participants:
                turn_prompt = self._debate_prompt(prompt, history_lines, agent.name, r)
                result = await agent.arun(turn_prompt)
                all_results.append(result)
                text = result.output or ""
                entry = {"round": r, "agent": agent.name, "output": text}
                transcript.append(entry)
                history_lines.append(f"[Round {r}] {agent.name}: {text}")

        synthesis: AgentResult | None = None
        if synth_agent is not None:
            synth_prompt = (
                f"Original question:\n{prompt}\n\n"
                f"Debate transcript:\n" + "\n".join(history_lines) + "\n\n"
                "Synthesise a final, balanced answer that incorporates the best arguments."
            )
            synthesis = await synth_agent.arun(synth_prompt)
            all_results.append(synthesis)

        if synthesis is not None:
            final_output = synthesis.output
        elif transcript:
            final_output = transcript[-1]["output"]
        else:
            final_output = None

        return {
            "transcript": transcript,
            "rounds": rounds,
            "results": all_results,
            "synthesis": synthesis,
            "output": final_output,
        }

    async def map_reduce(
        self,
        prompt: str,
        worker_names: list[str],
        reducer_name: str,
        *,
        reducer_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Parallel workers then a reducer that sees all worker outputs.

        Each worker receives ``prompt``. Their outputs are concatenated and
        passed to the reducer (along with the original prompt). Returns a dict
        with ``workers`` (name -> result), ``reducer`` (AgentResult), and
        ``output`` (the reducer's text).
        """
        if not worker_names:
            raise OrchestrationError("map_reduce requires at least one worker.")
        workers = self._resolve(worker_names)
        reducer = self._resolve_one(reducer_name)

        worker_results_list = await asyncio.gather(*(w.arun(prompt) for w in workers))
        workers_map: dict[str, AgentResult] = {
            w.name: r for w, r in zip(workers, worker_results_list, strict=False)
        }

        parts = [
            f"### {name}\n{result.output or ''}" for name, result in workers_map.items()
        ]
        concatenated = "\n\n".join(parts)
        if reducer_prompt is not None:
            reduce_input = reducer_prompt.format(
                prompt=prompt, outputs=concatenated, workers=concatenated
            )
        else:
            reduce_input = (
                f"Original task:\n{prompt}\n\n"
                f"Worker outputs:\n{concatenated}\n\n"
                "Combine the worker outputs into a single coherent answer."
            )
        reducer_result = await reducer.arun(reduce_input)
        return {
            "workers": workers_map,
            "reducer": reducer_result,
            "output": reducer_result.output,
        }

    def to_mermaid(self) -> str:
        """Return a Mermaid flowchart diagram of the registered agent team.

        Useful for docs and debugging — each agent is a node labelled with its
        name and a short instructions snippet.
        """
        lines = ["flowchart LR"]
        if not self.agents:
            lines.append("    empty[No agents]")
            return "\n".join(lines)

        for name, agent in self.agents.items():
            node_id = _mermaid_id(name)
            label = name
            if agent.instructions:
                snippet = agent.instructions.strip().splitlines()[0][:40]
                snippet = snippet.replace('"', "'")
                label = f"{name}<br/>{snippet}"
            lines.append(f'    {node_id}["{label}"]')

        names = list(self.agents)
        for i in range(len(names) - 1):
            a, b = _mermaid_id(names[i]), _mermaid_id(names[i + 1])
            lines.append(f"    {a} --- {b}")
        return "\n".join(lines)

    # -- helpers ----------------------------------------------------------------
    def _resolve_agents(
        self, agents: list[str] | list[Agent] | None
    ) -> list[Agent]:
        if agents is None:
            return list(self.agents.values())
        if not agents:
            return []
        if isinstance(agents[0], Agent):
            return list(agents)  # type: ignore[arg-type]
        return self._resolve(list(agents))  # type: ignore[arg-type]

    @staticmethod
    def _debate_prompt(
        original: str, history: list[str], speaker: str, round_num: int
    ) -> str:
        if not history:
            return (
                f"You are participating in a multi-agent debate (round {round_num}).\n"
                f"Your name is {speaker}.\n\n"
                f"Question:\n{original}\n\n"
                "Give your opening position clearly and concisely."
            )
        return (
            f"You are participating in a multi-agent debate (round {round_num}).\n"
            f"Your name is {speaker}.\n\n"
            f"Question:\n{original}\n\n"
            f"Transcript so far:\n" + "\n".join(history) + "\n\n"
            "Respond to the other agents. Build on strong points, challenge weak ones."
        )


def _mermaid_id(name: str) -> str:
    """Sanitise an agent name into a valid Mermaid node id."""
    cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"a_{cleaned}"
    return cleaned


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
