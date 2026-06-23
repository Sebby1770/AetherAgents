"""Provider abstraction.

A provider turns a list of :class:`~aetheragents.core.messages.Message` (plus an
optional tool catalogue) into an :class:`LLMResponse`. Concrete providers live
alongside this module (``mock``, ``litellm_provider``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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
