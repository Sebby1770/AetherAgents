"""Try providers in order until one completion succeeds.

Useful when a primary model is down and a second endpoint can answer::

    provider = FallbackProvider([
        AnthropicProvider("claude-sonnet-5"),
        LiteLLMProvider("gpt-4o-mini"),
    ])

Only exceptions listed in ``retry_on`` (default: :class:`ProviderError`) move
to the next provider. Programming errors propagate immediately.
"""

from __future__ import annotations

from typing import Any

from ..core.errors import ProviderError
from ..core.messages import Message
from .base import LLMProvider, LLMResponse


class FallbackProvider(LLMProvider):
    """Wraps an ordered list of providers and uses the first one that succeeds."""

    def __init__(
        self,
        providers: list[LLMProvider],
        *,
        retry_on: tuple[type[Exception], ...] = (ProviderError,),
    ) -> None:
        if not providers:
            raise ValueError("FallbackProvider requires at least one provider")
        self.providers = list(providers)
        self.retry_on = retry_on
        self.name = "fallback(" + ",".join(p.name for p in self.providers) + ")"
        #: Provider names attempted on the most recent ``complete`` call.
        self.attempts: list[str] = []
        #: Name of the provider that produced the last successful response.
        self.last_provider: str | None = None

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.attempts = []
        self.last_provider = None
        errors: list[str] = []
        for provider in self.providers:
            self.attempts.append(provider.name)
            try:
                response = await provider.complete(
                    messages, tools=tools, model=model, temperature=temperature, **kwargs
                )
            except self.retry_on as exc:
                errors.append(f"{provider.name}: {exc}")
                continue
            self.last_provider = provider.name
            return response
        raise ProviderError(
            "all fallback providers failed: " + "; ".join(errors)
        )
