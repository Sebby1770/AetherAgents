"""Provider abstraction.

A provider turns a list of :class:`~aetheragents.core.messages.Message` (plus an
optional tool catalogue) into an :class:`LLMResponse`. Concrete providers live
alongside this module (``mock``, ``litellm_provider``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from ..core.messages import Message, ToolCall


class Usage(BaseModel):
    """Token accounting for a single completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class LLMResponse(BaseModel):
    """A normalised completion returned by any provider."""

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    model: str | None = None
    usage: Usage = Field(default_factory=Usage)
    raw: Any = Field(default=None, repr=False)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class StreamEvent(BaseModel):
    """An incremental event produced by :meth:`LLMProvider.stream`.

    ``delta`` events carry a fragment of assistant text as it is generated; the
    single terminal ``done`` event carries the complete :class:`LLMResponse`
    (including any tool calls, finish reason and usage).
    """

    type: str  # "delta" | "done"
    delta: str = ""
    response: LLMResponse | None = None


class LLMProvider(ABC):
    """Abstract base class for model providers."""

    #: A human-friendly identifier, e.g. ``"litellm:gpt-4o"``.
    name: str = "provider"

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Produce a single completion for ``messages``."""
        raise NotImplementedError

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a completion as :class:`StreamEvent`s.

        The default implementation is a non-streaming fallback: it awaits
        :meth:`complete`, emits the whole content as one ``delta``, then
        ``done``. Providers with native streaming should override this; every
        override must end with exactly one ``done`` event carrying the full
        response.
        """
        resp = await self.complete(
            messages, tools=tools, model=model, temperature=temperature, **kwargs
        )
        if resp.content:
            yield StreamEvent(type="delta", delta=resp.content)
        yield StreamEvent(type="done", response=resp)
