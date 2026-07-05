"""The :class:`Agent`: a reasoning loop over an LLM provider, tools and memory.

An agent takes a prompt, lets the model think and call tools in a loop, and
returns a structured :class:`AgentResult`. The loop itself is exposed as an
async event stream (:meth:`Agent.astream`) so callers can render text deltas
and tool activity live; :meth:`Agent.arun` is the buffered form built on top of
it. Agents can also be exposed *as tools* (:meth:`Agent.as_tool`) so one agent
can delegate to another - the basis for the multi-agent orchestration in
:mod:`aetheragents.core.orchestrator`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

from pydantic import BaseModel, Field

from ..llm.base import LLMProvider, LLMResponse, Usage
from ..telemetry import span
from .errors import MaxStepsExceeded, ProviderError, StructuredOutputError
from .guardrails import Guardrail, apply_guardrails
from .messages import Message, Role
from .session import Session
from .structured import parse_structured, schema_instruction
from .tools import Tool, ToolRegistry, make_tool

StepCallback = Callable[["Step"], None]


class Step(BaseModel):
    """One observable event in an agent run (tool call, result, or final answer)."""

    type: str  # "tool_call" | "tool_result" | "final"
    name: str | None = None
    content: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True


class AgentResult(BaseModel):
    """The result of :meth:`Agent.arun`."""

    agent: str
    output: str | None
    steps: list[Step] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    #: The model that produced the final answer (as reported by the provider).
    model: str | None = None
    #: The validated ``response_model`` instance when structured output was requested.
    parsed: Any = None

    @property
    def tool_calls(self) -> list[Step]:
        return [s for s in self.steps if s.type == "tool_call"]

    @property
    def cost_usd(self) -> float | None:
        """Estimated USD cost of this run, or ``None`` for unknown models."""
        from ..costs import estimate_cost

        return estimate_cost(self.model, self.usage)

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.output or ""


class AgentEvent(BaseModel):
    """A live event yielded by :meth:`Agent.astream`.

    * ``delta``  - a fragment of assistant text as it is generated
    * ``step``   - a completed :class:`Step` (tool call / tool result / final)
    * ``result`` - the terminal event, carrying the full :class:`AgentResult`
    """

    type: str  # "delta" | "step" | "result"
    delta: str | None = None
    step: Step | None = None
    result: AgentResult | None = None


def _as_registry(tools: Any) -> ToolRegistry:
    if tools is None:
        return ToolRegistry()
    if isinstance(tools, ToolRegistry):
        return tools
    return ToolRegistry(list(tools))


class Agent:
    """A single autonomous agent."""

    def __init__(
        self,
        name: str,
        provider: LLMProvider,
        *,
        instructions: str = "",
        tools: ToolRegistry | list[Tool | Callable[..., Any]] | None = None,
        memory: Any | None = None,
        model: str | None = None,
        max_steps: int = 8,
        temperature: float | None = None,
        recall_k: int = 4,
        history_k: int = 10,
        on_step: StepCallback | None = None,
        input_guardrails: list[Guardrail] | None = None,
        output_guardrails: list[Guardrail] | None = None,
    ) -> None:
        self.name = name
        self.provider = provider
        self.instructions = instructions
        self.tools = _as_registry(tools)
        self.memory = memory
        self.model = model
        self.max_steps = max_steps
        self.temperature = temperature
        self.recall_k = recall_k
        self.history_k = history_k
        self.on_step = on_step
        self.input_guardrails = input_guardrails or []
        self.output_guardrails = output_guardrails or []

    # -- execution --------------------------------------------------------------
    async def astream(
        self,
        prompt: str,
        *,
        extra_messages: list[Message] | None = None,
        session: Session | None = None,
        response_model: type[BaseModel] | None = None,
        structured_retries: int = 2,
    ) -> AsyncIterator[AgentEvent]:
        """Run the agent, yielding :class:`AgentEvent`s as they happen.

        The stream ends with exactly one ``result`` event. When
        ``response_model`` is set, schema-violating answers are fed back to the
        model up to ``structured_retries`` times; every model call (including
        those retries) counts against ``max_steps``.
        """
        prompt = apply_guardrails(prompt, self.input_guardrails)
        messages = self._build_messages(prompt, extra_messages, session, response_model)
        session_start = len(messages) - 1  # the user prompt onwards is new history
        if self.memory is not None:
            self.memory.add_message("user", prompt)

        schemas = self.tools.get_schemas() or None
        steps: list[Step] = []
        usage = Usage()
        output: str | None = None
        model_used: str | None = None
        parsed: BaseModel | None = None
        parse_failures = 0
        finished = False

        with span("agent.run", agent=self.name):
            for _ in range(self.max_steps):
                resp: LLMResponse | None = None
                async for event in self.provider.stream(
                    messages,
                    tools=schemas,
                    model=self.model,
                    temperature=self.temperature,
                ):
                    if event.type == "delta" and event.delta:
                        yield AgentEvent(type="delta", delta=event.delta)
                    elif event.type == "done":
                        resp = event.response
                if resp is None:
                    raise ProviderError(
                        f"Provider '{self.provider.name}' stream ended without a 'done' event."
                    )

                usage = usage + resp.usage
                model_used = resp.model or model_used
                messages.append(
                    Message.assistant(content=resp.content, tool_calls=resp.tool_calls)
                )

                if resp.has_tool_calls:
                    for call in resp.tool_calls:
                        step = Step(type="tool_call", name=call.name, arguments=call.arguments)
                        self._record(steps, step)
                        yield AgentEvent(type="step", step=step)
                        result = await self.tools.execute(call.name, call.arguments)
                        step = Step(
                            type="tool_result", name=call.name, content=result.content, ok=result.ok
                        )
                        self._record(steps, step)
                        yield AgentEvent(type="step", step=step)
                        messages.append(
                            Message.tool(content=result.content, tool_call_id=call.id, name=call.name)
                        )
                    continue

                final_text = resp.content
                if final_text is not None and self.output_guardrails:
                    final_text = apply_guardrails(final_text, self.output_guardrails)

                if response_model is not None:
                    try:
                        parsed = parse_structured(final_text, response_model)
                    except StructuredOutputError as exc:
                        parse_failures += 1
                        if parse_failures > structured_retries:
                            raise
                        messages.append(
                            Message.user(
                                f"Your previous answer was rejected: {exc} "
                                "Answer again with ONLY a valid JSON object matching the schema."
                            )
                        )
                        continue

                output = final_text
                step = Step(type="final", content=output)
                self._record(steps, step)
                yield AgentEvent(type="step", step=step)
                finished = True
                break

            if not finished:
                raise MaxStepsExceeded(
                    f"Agent '{self.name}' exceeded max_steps={self.max_steps}"
                )

        if self.memory is not None and output is not None:
            self.memory.add_message("assistant", output)
        if session is not None:
            session.extend(messages[session_start:])
            session.maybe_autosave()

        yield AgentEvent(
            type="result",
            result=AgentResult(
                agent=self.name,
                output=output,
                steps=steps,
                messages=messages,
                usage=usage,
                model=model_used,
                parsed=parsed,
            ),
        )

    async def arun(self, prompt: str, **kwargs: Any) -> AgentResult:
        """Run the agent to completion and return a structured result.

        Accepts the same keyword arguments as :meth:`astream`
        (``extra_messages``, ``session``, ``response_model``,
        ``structured_retries``).
        """
        result: AgentResult | None = None
        async for event in self.astream(prompt, **kwargs):
            if event.type == "result":
                result = event.result
        if result is None:  # pragma: no cover - astream always ends with result
            raise ProviderError("Agent stream ended without a result event.")
        return result

    def run(self, prompt: str, **kwargs: Any) -> AgentResult:
        """Synchronous wrapper around :meth:`arun`.

        Raises if called from within a running event loop - use ``await
        agent.arun(...)`` there instead.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(prompt, **kwargs))
        raise RuntimeError(
            "Agent.run() cannot be called from a running event loop; use 'await agent.arun(...)'."
        )

    # -- delegation -------------------------------------------------------------
    def as_tool(self, name: str | None = None, description: str | None = None) -> Tool:
        """Expose this agent as a :class:`Tool` so another agent can delegate to it."""
        agent = self

        async def delegate(input: str) -> str:
            result = await agent.arun(input)
            return result.output or ""

        return make_tool(
            delegate,
            name=name or self.name,
            description=description
            or self.instructions
            or f"Delegate a sub-task to the '{self.name}' agent.",
        )

    # -- internals --------------------------------------------------------------
    def _build_messages(
        self,
        prompt: str,
        extra_messages: list[Message] | None,
        session: Session | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> list[Message]:
        messages: list[Message] = []
        system_text = self.instructions
        if response_model is not None:
            system_text = f"{system_text}\n\n{schema_instruction(response_model)}".strip()
        if self.memory is not None:
            hits = self.memory.recall(prompt, k=self.recall_k)
            if hits:
                recalled = "\n".join(f"- {h['document']}" for h in hits)
                system_text = f"{system_text}\n\nRelevant memory:\n{recalled}".strip()
        if system_text:
            messages.append(Message.system(system_text))
        if session is not None:
            # The session owns the verbatim history; memory then only
            # contributes semantic recall (above), never duplicate turns.
            messages.extend(m for m in session.messages if m.role is not Role.SYSTEM)
        elif self.memory is not None:
            for entry in self.memory.get_recent_context(self.history_k):
                if entry.get("role") in ("user", "assistant") and entry.get("content"):
                    messages.append(Message(role=Role(entry["role"]), content=entry["content"]))
        if extra_messages:
            messages.extend(extra_messages)
        messages.append(Message.user(prompt))
        return messages

    def _record(self, steps: list[Step], step: Step) -> None:
        steps.append(step)
        if self.on_step is not None:
            try:
                self.on_step(step)
            except Exception:  # callbacks must never break the run
                pass
