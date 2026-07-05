"""A native Anthropic (Claude) provider using the official ``anthropic`` SDK.

Optional dependency - install with ``pip install 'aetheragents[anthropic]'``.
The SDK is imported lazily; the message/tool conversion helpers below are pure
functions so they can be tested without it.

Anthropic's Messages API differs from the OpenAI wire format in three ways this
module bridges:

* the system prompt is a top-level parameter, not a message;
* assistant tool calls are ``tool_use`` content blocks;
* tool results are ``tool_result`` blocks inside a *user* message.
"""

from __future__ import annotations

from typing import Any

from ..core.errors import ProviderError
from ..core.messages import Message, Role, ToolCall
from .base import LLMProvider, LLMResponse, Usage

DEFAULT_MODEL = "claude-opus-4-8"


def to_anthropic_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Convert framework messages to ``(system_prompt, anthropic_messages)``.

    Consecutive tool results are merged into a single user turn, as the API
    requires every ``tool_use`` to be answered in the immediately following
    user message.
    """
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def flush_tool_results() -> None:
        if pending_tool_results:
            out.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for msg in messages:
        if msg.role is Role.SYSTEM:
            if msg.content:
                system_parts.append(msg.content)
            continue
        if msg.role is Role.TOOL:
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id or "",
                    "content": msg.content or "",
                }
            )
            continue
        flush_tool_results()
        if msg.role is Role.ASSISTANT and msg.tool_calls:
            blocks: list[dict[str, Any]] = []
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                )
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": msg.role.value, "content": msg.content or ""})

    flush_tool_results()
    return "\n\n".join(system_parts), out


def to_anthropic_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style function schemas to Anthropic tool definitions."""
    tools: list[dict[str, Any]] = []
    for schema in schemas:
        fn = schema.get("function", schema)
        tools.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return tools


def parse_anthropic_response(resp: Any, fallback_model: str) -> LLMResponse:
    """Normalise an Anthropic Messages API response into :class:`LLMResponse`."""
    content_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in getattr(resp, "content", None) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            content_parts.append(getattr(block, "text", "") or "")
        elif btype == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=getattr(block, "id", "") or "",
                    name=getattr(block, "name", "") or "",
                    arguments=dict(getattr(block, "input", None) or {}),
                )
            )

    usage = Usage()
    raw_usage = getattr(resp, "usage", None)
    if raw_usage is not None:
        prompt = getattr(raw_usage, "input_tokens", 0) or 0
        completion = getattr(raw_usage, "output_tokens", 0) or 0
        usage = Usage(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
        )

    return LLMResponse(
        content="".join(content_parts) or None,
        tool_calls=tool_calls,
        finish_reason=getattr(resp, "stop_reason", None),
        model=getattr(resp, "model", None) or fallback_model,
        usage=usage,
        raw=resp,
    )


class AnthropicProvider(LLMProvider):
    """Talk to Claude models via the official Anthropic SDK.

    ``client`` may be injected (any object with ``messages.create``) - useful
    for tests and custom transports; otherwise ``anthropic.AsyncAnthropic`` is
    constructed lazily.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        max_tokens: int = 4096,
        client: Any | None = None,
        **default_kwargs: Any,
    ) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ProviderError(
                    "anthropic is not installed. Install it with: "
                    "pip install 'aetheragents[anthropic]'"
                ) from exc
            client = (
                anthropic.AsyncAnthropic(api_key=api_key) if api_key else anthropic.AsyncAnthropic()
            )
        self._client = client
        self.model = model
        self.name = f"anthropic:{model}"
        self.max_tokens = max_tokens
        self._default_kwargs = default_kwargs

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        system, converted = to_anthropic_messages(messages)
        params: dict[str, Any] = {
            "model": model or self.model,
            "messages": converted,
            "max_tokens": self.max_tokens,
            **self._default_kwargs,
        }
        if system:
            params["system"] = system
        if temperature is not None:
            params["temperature"] = temperature
        if tools:
            params["tools"] = to_anthropic_tools(tools)
        params.update(kwargs)

        try:
            resp = await self._client.messages.create(**params)
        except ProviderError:
            raise
        except Exception as exc:  # pragma: no cover - network/credential dependent
            raise ProviderError(f"anthropic completion failed: {exc}") from exc

        return parse_anthropic_response(resp, params["model"])
