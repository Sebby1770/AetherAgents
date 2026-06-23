"""The :class:`Agent`: a reasoning loop over an LLM provider, tools and memory.

An agent takes a prompt, lets the model think and call tools in a loop, and
returns a structured :class:`AgentResult`. Agents can also be exposed *as tools*
(:meth:`Agent.as_tool`) so one agent can delegate to another - the basis for the
multi-agent orchestration in :mod:`aetheragents.core.orchestrator`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from ..llm.base import LLMProvider, Usage
from ..telemetry import span
from .errors import MaxStepsExceeded
from .messages import Message, Role
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

    @property
    def tool_calls(self) -> list[Step]:
        return [s for s in self.steps if s.type == "tool_call"]

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.output or ""


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

    # -- execution --------------------------------------------------------------
    async def arun(
        self, prompt: str, *, extra_messages: list[Message] | None = None
    ) -> AgentResult:
        """Run the agent to completion and return a structured result."""
        messages = self._build_messages(prompt, extra_messages)
        if self.memory is not None:
            self.memory.add_message("user", prompt)

        schemas = self.tools.get_schemas() or None
        steps: list[Step] = []
        usage = Usage()
        output: str | None = None

        with span("agent.run", agent=self.name):
            for _ in range(self.max_steps):
                resp = await self.provider.complete(
                    messages,
                    tools=schemas,
                    model=self.model,
                    temperature=self.temperature,
                )
                usage = usage + resp.usage
                messages.append(
                    Message.assistant(content=resp.content, tool_calls=resp.tool_calls)
                )

                if resp.has_tool_calls:
                    for call in resp.tool_calls:
                        self._record(steps, Step(type="tool_call", name=call.name, arguments=call.arguments))
                        result = await self.tools.execute(call.name, call.arguments)
                        self._record(
                            steps,
                            Step(type="tool_result", name=call.name, content=result.content, ok=result.ok),
                        )
                        messages.append(
                            Message.tool(content=result.content, tool_call_id=call.id, name=call.name)
                        )
                    continue

                output = resp.content
                self._record(steps, Step(type="final", content=output))
                break
            else:
                raise MaxStepsExceeded(
                    f"Agent '{self.name}' exceeded max_steps={self.max_steps}"
                )

        if self.memory is not None and output is not None:
            self.memory.add_message("assistant", output)

        return AgentResult(
            agent=self.name, output=output, steps=steps, messages=messages, usage=usage
        )

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
        self, prompt: str, extra_messages: list[Message] | None
    ) -> list[Message]:
        messages: list[Message] = []
        system_text = self.instructions
        if self.memory is not None:
            hits = self.memory.recall(prompt, k=self.recall_k)
            if hits:
                recalled = "\n".join(f"- {h['document']}" for h in hits)
                system_text = f"{system_text}\n\nRelevant memory:\n{recalled}".strip()
        if system_text:
            messages.append(Message.system(system_text))
        if self.memory is not None:
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
