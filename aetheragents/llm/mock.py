"""A deterministic, offline provider for tests, demos and local development.

``MockProvider`` lets you drive an agent loop without any network access or API
key. You can either queue up scripted ``responses`` or pass a ``handler``
callable that inspects the running conversation and decides what to return.

Each queued/handled item may be:

* a ``str``               -> assistant message with that text
* an :class:`LLMResponse` -> returned verbatim
* a ``ToolCall`` / list   -> assistant message requesting those tool calls
* a ``dict``              -> ``{"content": ..., "tool_calls": [(name, args), ...]}``
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..core.messages import Message, ToolCall
from .base import LLMProvider, LLMResponse, Usage

ScriptItem = Any
Handler = Callable[[list[Message]], ScriptItem]


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(
        self,
        responses: list[ScriptItem] | None = None,
        *,
        handler: Handler | None = None,
        default: str = "",
        model: str = "mock-1",
    ) -> None:
        self._queue: list[ScriptItem] = list(responses or [])
        self._handler = handler
        self._default = default
        self._model = model
        self._call_count = 0
        #: Conversations seen, useful for assertions in tests.
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        item: ScriptItem
        if self._queue:
            item = self._queue.pop(0)
        elif self._handler is not None:
            item = self._handler(messages)
        else:
            item = self._default or _echo_last_user(messages)
        self._call_count += 1
        return self._coerce(item, model or self._model)

    # -- normalisation ----------------------------------------------------------
    def _coerce(self, item: ScriptItem, model: str) -> LLMResponse:
        if isinstance(item, LLMResponse):
            if item.model is None:
                item.model = model
            return item
        if isinstance(item, str):
            return LLMResponse(content=item, finish_reason="stop", model=model, usage=_usage(item))
        if isinstance(item, ToolCall):
            return self._tool_response([item], model)
        if isinstance(item, list):
            return self._tool_response([self._as_tool_call(tc) for tc in item], model)
        if isinstance(item, dict):
            raw_calls = item.get("tool_calls") or []
            calls = [self._as_tool_call(tc) for tc in raw_calls]
            if calls:
                return self._tool_response(calls, model, content=item.get("content"))
            return LLMResponse(
                content=item.get("content", ""), finish_reason="stop", model=model
            )
        raise TypeError(f"MockProvider cannot interpret response item: {item!r}")

    def _as_tool_call(self, tc: Any) -> ToolCall:
        if isinstance(tc, ToolCall):
            return tc
        if isinstance(tc, tuple):
            name, args = tc
            return ToolCall(id=self._next_id(), name=name, arguments=dict(args or {}))
        if isinstance(tc, dict):
            return ToolCall(
                id=tc.get("id", self._next_id()),
                name=tc["name"],
                arguments=dict(tc.get("arguments", {})),
            )
        raise TypeError(f"Cannot interpret tool call: {tc!r}")

    def _tool_response(
        self, calls: list[ToolCall], model: str, content: str | None = None
    ) -> LLMResponse:
        return LLMResponse(
            content=content, tool_calls=calls, finish_reason="tool_calls", model=model
        )

    def _next_id(self) -> str:
        return f"call_{self._call_count}_{len(self.calls)}"


def _usage(text: str) -> Usage:
    # A coarse, deterministic estimate (~4 chars/token) so traces look realistic.
    n = max(1, len(text) // 4)
    return Usage(completion_tokens=n, total_tokens=n)


def _echo_last_user(messages: list[Message]) -> str:
    for msg in reversed(messages):
        if msg.role.value == "user" and msg.content:
            return f"(mock) You said: {msg.content}"
    return "(mock) no input"
