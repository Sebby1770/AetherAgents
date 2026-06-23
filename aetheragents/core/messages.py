"""Message primitives shared across providers, agents and orchestration.

These are deliberately provider-agnostic. :meth:`Message.to_provider_dict`
renders the OpenAI / LiteLLM chat-completion wire format, which is the de-facto
standard most model providers accept.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    """The author of a message in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A request from the model to invoke a tool."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """A single turn in a conversation.

    A message may carry plain ``content``, a set of ``tool_calls`` requested by
    the assistant, or the result of a tool execution (with ``tool_call_id``).
    """

    role: Role
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None

    # -- ergonomic constructors -------------------------------------------------
    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(
        cls, content: str | None = None, tool_calls: list[ToolCall] | None = None
    ) -> Message:
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str | None = None) -> Message:
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id, name=name)

    # -- serialisation ----------------------------------------------------------
    def to_provider_dict(self) -> dict[str, Any]:
        """Render this message in OpenAI / LiteLLM chat-completion format."""
        data: dict[str, Any] = {"role": self.role.value}
        # Tool/assistant messages may legitimately have ``content=None``.
        if self.content is not None or not self.tool_calls:
            data["content"] = self.content
        if self.name:
            data["name"] = self.name
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        return data

    def __str__(self) -> str:  # pragma: no cover - convenience only
        if self.tool_calls:
            calls = ", ".join(f"{c.name}({c.arguments})" for c in self.tool_calls)
            return f"[{self.role.value}] -> {calls}"
        return f"[{self.role.value}] {self.content or ''}"
