"""A real model provider backed by `LiteLLM <https://docs.litellm.ai>`_.

LiteLLM gives a single OpenAI-style interface to 100+ models (OpenAI,
Anthropic, Gemini, local models via Ollama, ...). It is an *optional* dependency
- install it with ``pip install 'aetheragents[litellm]'``. Importing this module
never imports litellm; the import happens lazily when a provider is created.
"""

from __future__ import annotations

import json
from typing import Any

from ..core.errors import ProviderError
from ..core.messages import Message, ToolCall
from .base import LLMProvider, LLMResponse, Usage


class LiteLLMProvider(LLMProvider):
    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        **default_kwargs: Any,
    ) -> None:
        try:
            import litellm  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ProviderError(
                "litellm is not installed. Install it with: pip install 'aetheragents[litellm]'"
            ) from exc
        self._litellm = litellm
        self.model = model
        self.name = f"litellm:{model}"
        self._api_key = api_key
        self._api_base = api_base
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
        params: dict[str, Any] = {
            "model": model or self.model,
            "messages": [m.to_provider_dict() for m in messages],
            **self._default_kwargs,
        }
        if self._api_key:
            params["api_key"] = self._api_key
        if self._api_base:
            params["api_base"] = self._api_base
        if temperature is not None:
            params["temperature"] = temperature
        if tools:
            params["tools"] = tools
            params.setdefault("tool_choice", "auto")
        params.update(kwargs)

        try:
            resp = await self._litellm.acompletion(**params)
        except Exception as exc:  # pragma: no cover - network/credential dependent
            raise ProviderError(f"litellm completion failed: {exc}") from exc

        return self._parse(resp)

    def _parse(self, resp: Any) -> LLMResponse:
        choice = resp.choices[0]
        message = choice.message
        tool_calls: list[ToolCall] = []
        for raw in getattr(message, "tool_calls", None) or []:
            try:
                args = json.loads(raw.function.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append(
                ToolCall(id=raw.id, name=raw.function.name, arguments=args)
            )

        usage = Usage()
        if getattr(resp, "usage", None):
            usage = Usage(
                prompt_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(resp.usage, "total_tokens", 0) or 0,
            )

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=getattr(choice, "finish_reason", None),
            model=getattr(resp, "model", self.model),
            usage=usage,
            raw=resp,
        )
