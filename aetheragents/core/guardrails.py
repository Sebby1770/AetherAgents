"""Guardrails: validation/transformation hooks on agent input and output.

A guardrail is any callable ``(text: str) -> str``. It may return the text
unchanged, return a transformed version, or raise :class:`GuardrailError` to
block the run. Attach them to an agent::

    agent = Agent(
        "support",
        provider,
        input_guardrails=[max_length(2000)],
        output_guardrails=[blocklist(["password", "ssn"])],
    )

Input guardrails run on the user prompt before it reaches the model; output
guardrails run on the final answer before it is recorded (and, for structured
output, before parsing). Guardrails run in list order, each receiving the
previous one's output.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from .errors import GuardrailError

Guardrail = Callable[[str], str]


def apply_guardrails(text: str, guardrails: Iterable[Guardrail]) -> str:
    """Run ``text`` through each guardrail in order. Raises :class:`GuardrailError`."""
    for guard in guardrails:
        text = guard(text)
    return text


def max_length(limit: int, *, truncate: bool = False) -> Guardrail:
    """Reject (or truncate) text longer than ``limit`` characters."""

    def guard(text: str) -> str:
        if len(text) <= limit:
            return text
        if truncate:
            return text[:limit]
        raise GuardrailError(f"Text exceeds the {limit}-character limit ({len(text)} chars)")

    return guard


def blocklist(words: Iterable[str], *, message: str | None = None) -> Guardrail:
    """Reject text containing any of ``words`` (case-insensitive)."""
    lowered = [w.lower() for w in words]

    def guard(text: str) -> str:
        haystack = text.lower()
        for word in lowered:
            if word in haystack:
                raise GuardrailError(message or f"Text contains blocked term: {word!r}")
        return text

    return guard


def redact(words: Iterable[str], *, replacement: str = "[REDACTED]") -> Guardrail:
    """Replace occurrences of ``words`` (case-insensitive) with ``replacement``."""
    lowered = [w for w in words if w]

    def guard(text: str) -> str:
        for word in lowered:
            index = text.lower().find(word.lower())
            while index != -1:
                text = text[:index] + replacement + text[index + len(word) :]
                index = text.lower().find(word.lower(), index + len(replacement))
        return text

    return guard
