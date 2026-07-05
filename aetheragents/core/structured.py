"""Structured output: parse a model's final answer into a Pydantic model.

Used by ``Agent.arun(prompt, response_model=MyModel)``. The helpers here are
deliberately forgiving about the *packaging* of the JSON (code fences,
surrounding prose) while remaining strict about the schema itself - validation
is Pydantic's job.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .errors import StructuredOutputError

T = TypeVar("T", bound=BaseModel)


def extract_json(text: str) -> Any:
    """Pull the first JSON value out of ``text``.

    Handles raw JSON, ```` ```json ```` fences, and JSON embedded in
    surrounding prose. Raises ``ValueError`` if nothing parseable is found.
    """
    candidate = text.strip()

    # Strip a Markdown code fence if present.
    if "```" in candidate:
        parts = candidate.split("```")
        # fence contents are at odd indices; take the first non-empty one
        for part in parts[1::2]:
            body = part.strip()
            if body.startswith("json"):
                body = body[4:].strip()
            if body:
                candidate = body
                break

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost {...} or [...] span.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = candidate.find(open_ch)
        end = candidate.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("No parseable JSON found in model output")


def schema_instruction(model_cls: type[BaseModel]) -> str:
    """Build the system-prompt addendum asking for schema-conformant JSON."""
    schema = json.dumps(model_cls.model_json_schema(), indent=None)
    return (
        "You must answer with a single JSON object that conforms to this JSON Schema "
        f"(no prose, no explanations):\n{schema}"
    )


def parse_structured(text: str | None, model_cls: type[T]) -> T:
    """Parse ``text`` into ``model_cls`` or raise :class:`StructuredOutputError`.

    The exception message describes what went wrong so it can be fed back to
    the model as a correction.
    """
    if not text:
        raise StructuredOutputError("Model returned an empty response; expected JSON.")
    try:
        data = extract_json(text)
    except ValueError as exc:
        raise StructuredOutputError(f"Could not find valid JSON in the response: {exc}") from exc
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise StructuredOutputError(f"JSON does not match the schema: {exc}") from exc
