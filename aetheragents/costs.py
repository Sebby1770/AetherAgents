"""Per-model token cost estimation.

Prices are USD per million tokens ``(input, output)``. Anthropic prices follow
the published rates as of mid-2026; entries for other providers are
approximations. Rates change - treat estimates as indicative and override or
extend the table at runtime with :func:`register_model_cost`.

Usage::

    from aetheragents import estimate_cost

    cost = estimate_cost("claude-opus-4-8", result.usage)   # -> float | None
    print(result.cost_usd)                                   # same, via AgentResult
"""

from __future__ import annotations

from .llm.base import Usage

# Model-name prefix -> (input $/MTok, output $/MTok). Longest prefix wins.
_COSTS: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-fable-5": (10.00, 50.00),
    "claude-mythos-5": (10.00, 50.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-opus": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku": (1.00, 5.00),
    # Other providers (approximate - override with register_model_cost)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

# Provider prefixes commonly prepended by gateways (Bedrock, OpenRouter, ...).
_STRIP_PREFIXES = ("anthropic.", "anthropic/", "openai/", "bedrock/", "litellm:")


def register_model_cost(
    model_prefix: str, input_per_mtok: float, output_per_mtok: float
) -> None:
    """Add or override the price for any model name starting with ``model_prefix``."""
    _COSTS[model_prefix.lower()] = (input_per_mtok, output_per_mtok)


def _normalise(model: str) -> str:
    name = model.lower().strip()
    for prefix in _STRIP_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def estimate_cost(model: str | None, usage: Usage) -> float | None:
    """Estimate the USD cost of ``usage`` on ``model``.

    Returns ``None`` when the model is unknown - callers should treat that as
    "no estimate available", not zero.
    """
    if not model:
        return None
    name = _normalise(model)
    match = max(
        (prefix for prefix in _COSTS if name.startswith(prefix)),
        key=len,
        default=None,
    )
    if match is None:
        return None
    input_rate, output_rate = _COSTS[match]
    return (usage.prompt_tokens * input_rate + usage.completion_tokens * output_rate) / 1_000_000
