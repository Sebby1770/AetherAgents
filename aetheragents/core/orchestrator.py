"""Multi-agent orchestration patterns.

Composable patterns cover most teams:

* **sequential** - a pipeline; each agent's output feeds the next.
* **parallel**   - fan out the same task to many agents, gather all results.
* **route**      - a selector picks the single best agent for the task.
* **debate**     - multi-round discussion where agents respond to each other.
* **map_reduce** - parallel workers then a reducer synthesises their outputs.
* **handoff**    - run one agent, then pass its output to another with a prefix.
* **supervise**  - worker/critic loop; critic replies ``ACCEPT`` or ``REVISE:``.
* **consensus**  - parallel answers, majority vote, optional ``PICK:`` judge.
* **workflow**   - ordered steps; each output feeds the next and can be saved.

For free-form delegation, expose an agent with :meth:`Agent.as_tool` and give it
to a "manager" agent's tool registry.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .agent import Agent, AgentResult
from .blackboard import Blackboard, BlackboardError
from .errors import OrchestrationError
from .messages import Message

Selector = Callable[[str, dict[str, Agent]], str]


class SuperviseResult(BaseModel):
    """Outcome of :meth:`Orchestrator.supervise`."""

    accepted: bool
    rounds: int
    final: AgentResult

    @property
    def output(self) -> str | None:
        return self.final.output


class HandoffResult(BaseModel):
    """Outcome of :meth:`Orchestrator.handoff` — both legs of the transfer."""

    from_name: str
    to_name: str
    from_result: AgentResult
    to_result: AgentResult

    @property
    def output(self) -> str | None:
        return self.to_result.output

    @property
    def result(self) -> AgentResult:
        """The receiving agent's result (the handoff output)."""
        return self.to_result


class ConsensusResult(BaseModel):
    """Outcome of :meth:`Orchestrator.consensus`."""

    winner: str
    output: str | None
    agreement: float
    judged: bool
    quorum: bool = True
    results: dict[str, AgentResult]

    @property
    def votes(self) -> dict[str, str]:
        """Agent name to stripped output."""
        return {name: (result.output or "").strip() for name, result in self.results.items()}


class WorkflowStep(BaseModel):
    """One step in :meth:`Orchestrator.workflow`."""

    agent: str
    prompt: str | None = None
    save_as: str | None = None
    #: Run this step only when the blackboard contains ``when``.
    when: str | None = None
    #: If set, the blackboard value at ``when`` must render as this text.
    equals: str | None = None


class WorkflowResult(BaseModel):
    """Outcome of :meth:`Orchestrator.workflow`."""

    steps: list[AgentResult]
    saved: dict[str, str]
    skipped: list[str] = Field(default_factory=list)
    #: ``"budget"`` when a team cost cap stopped the remaining steps.
    stopped: str | None = None
    spent_usd: float = 0.0

    @property
    def output(self) -> str | None:
        if not self.steps:
            return None
        return self.steps[-1].output


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

    async def handoff(
        self,
        prompt: str,
        from_name: str,
        to_name: str,
    ) -> HandoffResult:
        """Run ``from_name``, then feed its output to ``to_name``.

        The receiving agent sees a short system/user handoff prefix plus the
        original ``prompt``. Returns a :class:`HandoffResult` with both legs.
        """
        source = self._resolve_one(from_name)
        dest = self._resolve_one(to_name)
        from_result = await source.arun(prompt)
        prior = from_result.output or ""
        extra = [
            Message.system(
                f"Handoff from agent '{from_name}'. "
                "Continue their work; use the prior output as context."
            ),
        ]
        dest_prompt = (
            f"[handoff from {from_name}]\n"
            f"{prior}\n\n"
            f"{prompt}"
        )
        to_result = await dest.arun(dest_prompt, extra_messages=extra)
        return HandoffResult(
            from_name=from_name,
            to_name=to_name,
            from_result=from_result,
            to_result=to_result,
        )

    async def supervise(
        self,
        prompt: str,
        worker: str | Agent,
        critic: str | Agent,
        max_rounds: int = 2,
    ) -> SuperviseResult:
        """Run a worker, then a critic, revising until ``ACCEPT`` or ``max_rounds``.

        The worker answers ``prompt``. The critic is asked to reply with a first
        line of ``ACCEPT`` or ``REVISE: <notes>``. On ``REVISE``, the worker is
        re-run with the original task, its previous answer, and the notes.

        ``worker`` / ``critic`` may be registered agent names or :class:`Agent`
        instances. Returns a :class:`SuperviseResult` with ``accepted``,
        ``rounds`` and the last worker :class:`AgentResult` as ``final``.
        """
        if max_rounds < 1:
            raise OrchestrationError("supervise max_rounds must be >= 1")
        worker_agent = self._coerce_agent(worker, "worker")
        critic_agent = self._coerce_agent(critic, "critic")

        worker_prompt = prompt
        last_result: AgentResult | None = None
        accepted = False
        rounds_used = 0

        for round_num in range(1, max_rounds + 1):
            rounds_used = round_num
            last_result = await worker_agent.arun(worker_prompt)
            critic_prompt = _critic_prompt(prompt, last_result.output or "")
            critic_result = await critic_agent.arun(critic_prompt)
            decision, notes = _parse_critic_verdict(critic_result.output)
            if decision == "accept":
                accepted = True
                break
            if round_num < max_rounds:
                worker_prompt = _revision_prompt(
                    prompt, last_result.output or "", notes
                )

        assert last_result is not None  # max_rounds >= 1 guarantees a worker run
        return SuperviseResult(accepted=accepted, rounds=rounds_used, final=last_result)

    async def consensus(
        self,
        prompt: str,
        names: list[str] | None = None,
        *,
        judge: str | Agent | None = None,
        min_agreement: float = 0.0,
    ) -> ConsensusResult:
        """Run agents on the same prompt and pick a winning answer.

        Without ``judge``, the most common stripped output wins. Ties keep the
        answer of the earliest agent in ``names`` (or registration order).
        With ``judge``, that agent must reply with a first line of
        ``PICK: <agent name>``. An unusable verdict falls back to the vote.

        ``min_agreement`` does not change the winner. ``quorum`` is true when
        the share of agents matching that winner is at least the threshold.
        """
        if not 0.0 <= min_agreement <= 1.0:
            raise OrchestrationError("min_agreement must be between 0 and 1")
        agents = self._resolve(names)
        if not agents:
            raise OrchestrationError("No agents to run.")
        results = await asyncio.gather(*(agent.arun(prompt) for agent in agents))
        by_name = {agent.name: result for agent, result in zip(agents, results, strict=False)}
        winner = _majority_winner(agents, by_name)
        judged = False
        if judge is not None:
            judge_agent = self._coerce_agent(judge, "judge")
            verdict = await judge_agent.arun(_judge_prompt(prompt, agents, by_name))
            picked = _parse_pick(verdict.output, [agent.name for agent in agents])
            if picked is not None:
                winner = picked
                judged = True
        output = by_name[winner].output
        matching = sum(
            1
            for result in by_name.values()
            if (result.output or "").strip() == (output or "").strip()
        )
        agreement = matching / len(agents)
        return ConsensusResult(
            winner=winner,
            output=output,
            agreement=agreement,
            judged=judged,
            quorum=agreement + 1e-12 >= min_agreement,
            results=by_name,
        )

    async def workflow(
        self,
        prompt: str,
        steps: list[WorkflowStep | dict[str, Any]],
        *,
        blackboard: Blackboard | None = None,
        max_cost_usd: float | None = None,
    ) -> WorkflowResult:
        """Run agents in order. Each step sees the previous output.

        A step ``prompt`` replaces that hand-off text. ``{key}`` placeholders
        in that text are filled from the blackboard. ``when`` / ``equals``
        skip a step unless the blackboard matches. ``save_as`` records the
        step output and, when ``blackboard`` is set, stores it there.

        ``max_cost_usd`` stops before the next step once estimated spend
        reaches the cap. Unknown model costs count as zero. Completed steps
        are kept; ``stopped`` is ``"budget"``.
        """
        if not steps:
            raise OrchestrationError("workflow requires at least one step")
        if max_cost_usd is not None and max_cost_usd < 0:
            raise OrchestrationError("max_cost_usd must be >= 0")
        parsed = [_as_step(step) for step in steps]
        current = prompt
        results: list[AgentResult] = []
        saved: dict[str, str] = {}
        skipped: list[str] = []
        stopped: str | None = None
        spent = 0.0
        for step in parsed:
            if step.when and not _when_matches(blackboard, step):
                skipped.append(step.agent)
                continue
            if max_cost_usd is not None and spent >= max_cost_usd:
                stopped = "budget"
                break
            agent = self._resolve_one(step.agent)
            text = current if step.prompt is None else step.prompt
            text = _fill_prompt(text, blackboard)
            result = await agent.arun(text)
            results.append(result)
            current = result.output or ""
            spent += result.cost_usd or 0.0
            if step.save_as:
                saved[step.save_as] = current
                if blackboard is not None:
                    try:
                        blackboard.put(step.save_as, current)
                    except BlackboardError as exc:
                        raise OrchestrationError(
                            f"cannot save workflow key {step.save_as!r}: {exc}"
                        ) from exc
        return WorkflowResult(
            steps=results,
            saved=saved,
            skipped=skipped,
            stopped=stopped,
            spent_usd=spent,
        )

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
    def _coerce_agent(self, value: str | Agent, label: str) -> Agent:
        if isinstance(value, Agent):
            return value
        try:
            return self._resolve_one(value)
        except OrchestrationError as exc:
            raise OrchestrationError(f"Unknown {label}: {value!r}") from exc

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


def _critic_prompt(original: str, worker_output: str) -> str:
    return (
        "You are a critic reviewing another agent's work.\n\n"
        f"Original task:\n{original}\n\n"
        f"Worker's answer:\n{worker_output}\n\n"
        "Reply with a decision on the FIRST line, exactly one of:\n"
        "ACCEPT\n"
        "REVISE: <notes>\n"
        "If you revise, put actionable notes after the colon. "
        "You may add extra explanation after the first line."
    )


def _revision_prompt(original: str, previous: str, notes: str) -> str:
    return (
        f"Original task:\n{original}\n\n"
        f"Your previous answer:\n{previous}\n\n"
        f"A critic requested a revision:\n{notes or '(no notes)'}\n\n"
        "Produce an improved answer that addresses the critic's notes."
    )


def _parse_critic_verdict(text: str | None) -> tuple[str, str]:
    """Return ``('accept', '')`` or ``('revise', notes)`` from critic output."""
    raw = (text or "").strip()
    if not raw:
        return "revise", ""
    lines = raw.splitlines()
    first = lines[0].strip()
    head = first.split(None, 1)[0].rstrip(":").upper()
    rest_first = first[len(first.split(None, 1)[0]) :].strip()
    if rest_first.startswith(":"):
        rest_first = rest_first[1:].strip()
    extra = "\n".join(lines[1:]).strip()
    notes = "\n".join(p for p in (rest_first, extra) if p)
    if head == "ACCEPT":
        return "accept", notes
    if head == "REVISE":
        return "revise", notes
    return "revise", raw


def _mermaid_id(name: str) -> str:
    """Sanitise an agent name into a valid Mermaid node id."""
    cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"a_{cleaned}"
    return cleaned


def _render_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _when_matches(blackboard: Blackboard | None, step: WorkflowStep) -> bool:
    if blackboard is None or not step.when or step.when not in blackboard.keys():
        return False
    if step.equals is None:
        return True
    return _render_value(blackboard.get(step.when)) == step.equals


def _fill_prompt(text: str, blackboard: Blackboard | None) -> str:
    if blackboard is None:
        return text
    for key, value in blackboard.snapshot().items():
        text = text.replace("{" + key + "}", _render_value(value))
    return text


def _majority_winner(agents: list[Agent], results: dict[str, AgentResult]) -> str:
    """Largest identical stripped output. Ties follow ``agents`` order."""
    counts: dict[str, int] = {}
    first_agent: dict[str, str] = {}
    for agent in agents:
        text = (results[agent.name].output or "").strip()
        counts[text] = counts.get(text, 0) + 1
        first_agent.setdefault(text, agent.name)
    best_count = max(counts.values())
    for agent in agents:
        text = (results[agent.name].output or "").strip()
        if counts[text] == best_count:
            return first_agent[text]
    return agents[0].name


def _judge_prompt(original: str, agents: list[Agent], results: dict[str, AgentResult]) -> str:
    lines = [
        "You are judging parallel answers to the same task.",
        "Reply with a decision on the FIRST line, exactly:",
        "PICK: <agent name>",
        "",
        f"Task:\n{original}",
        "",
        "Answers:",
    ]
    for agent in agents:
        lines.append(f"[{agent.name}]\n{(results[agent.name].output or '').strip()}")
    return "\n".join(lines)


def _parse_pick(text: str | None, valid_names: list[str]) -> str | None:
    raw = (text or "").strip()
    if not raw:
        return None
    first = raw.splitlines()[0].strip()
    head = first.split(None, 1)[0].rstrip(":").upper()
    if head != "PICK":
        return None
    rest = first[len(first.split(None, 1)[0]) :].strip()
    if rest.startswith(":"):
        rest = rest[1:].strip()
    for name in valid_names:
        if name == rest or name.lower() == rest.lower():
            return name
    return None


def _as_step(step: WorkflowStep | dict[str, Any]) -> WorkflowStep:
    if isinstance(step, WorkflowStep):
        return step
    if isinstance(step, dict):
        try:
            return WorkflowStep.model_validate(step)
        except ValidationError as exc:
            raise OrchestrationError(f"invalid workflow step: {exc}") from exc
    raise OrchestrationError(f"invalid workflow step: {step!r}")


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
