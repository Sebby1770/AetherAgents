"""Tools: callables an agent can invoke, with JSON-Schema generated from the
function signature.

The previous implementation registered every tool with an *empty* parameter
schema, so models could never learn what arguments to pass. This version
introspects the signature and type hints to build a proper schema, and adds a
:func:`tool` decorator for ergonomic definitions.
"""

from __future__ import annotations

import inspect
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from .errors import ToolError, ToolNotFoundError

_PRIMITIVE_SCHEMA = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    tuple: "array",
    set: "array",
}

# Fallback mapping by *name*, used when annotations are strings (e.g. when the
# defining module uses ``from __future__ import annotations``).
_NAME_SCHEMA = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "List": "array",
    "dict": "object",
    "Dict": "object",
    "tuple": "array",
    "set": "array",
    "Any": "string",
}


class ToolResult(BaseModel):
    """The outcome of executing a tool."""

    content: str
    summary: str
    ok: bool = True
    error: str | None = None


@dataclass
class Tool:
    """A registered tool: a callable plus the schema advertised to the model."""

    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict[str, Any]
    is_async: bool

    def to_schema(self) -> dict[str, Any]:
        """Render in OpenAI / LiteLLM ``tools`` format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _resolve_type(annotation: Any) -> tuple[str, bool]:
    """Return ``(json_type, is_optional)`` for a type annotation.

    Handles real types, ``typing`` generics, PEP 604 unions (``int | None``)
    and string annotations.
    """
    if annotation is inspect.Parameter.empty or annotation is None:
        return "string", False

    # String annotation (deferred evaluation): parse leniently.
    if isinstance(annotation, str):
        optional = "None" in annotation or "Optional" in annotation
        base = annotation.split("[", 1)[0].split("|", 1)[0].strip()
        base = base.rsplit(".", 1)[-1]  # strip module qualifiers
        return _NAME_SCHEMA.get(base, "string"), optional

    origin = typing.get_origin(annotation)
    # Optional[X] / Union[..., None] / X | None
    if origin is typing.Union or origin is getattr(types, "UnionType", object()):
        args = typing.get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        optional = len(non_none) != len(args)
        if non_none:
            inner, _ = _resolve_type(non_none[0])
            return inner, optional
        return "string", optional

    if origin in (list, set, tuple, frozenset):
        return "array", False
    if origin is dict:
        return "object", False

    if isinstance(annotation, type):
        return _PRIMITIVE_SCHEMA.get(annotation, "string"), False

    name = getattr(annotation, "__name__", str(annotation))
    return _NAME_SCHEMA.get(name, "string"), False


def build_parameters(func: Callable[..., Any]) -> dict[str, Any]:
    """Build a JSON-Schema ``parameters`` object from a function signature."""
    sig = inspect.signature(func)
    try:
        hints = typing.get_type_hints(func)
    except Exception:  # pragma: no cover - exotic annotations
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        annotation = hints.get(pname, param.annotation)
        json_type, optional = _resolve_type(annotation)
        prop: dict[str, Any] = {"type": json_type}
        if json_type == "array":
            prop["items"] = {}
        properties[pname] = prop

        if param.default is inspect.Parameter.empty and not optional:
            required.append(pname)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def make_tool(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool:
    """Wrap a callable into a :class:`Tool` with a generated schema."""
    return Tool(
        name=name or func.__name__,
        description=(description or inspect.getdoc(func) or "").strip(),
        func=func,
        parameters=build_parameters(func),
        is_async=inspect.iscoroutinefunction(func),
    )


def tool(
    name: str | None = None, description: str | None = None
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator that turns a function into a :class:`Tool`.

    Usage::

        @tool()
        def add(a: int, b: int) -> int:
            "Add two numbers."
            return a + b
    """

    def decorator(func: Callable[..., Any]) -> Tool:
        return make_tool(func, name=name, description=description)

    return decorator


class ToolRegistry:
    """A collection of tools an agent can call."""

    def __init__(self, tools: list[Tool | Callable[..., Any]] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(
        self,
        func: Callable[..., Any] | Tool,
        name: str | None = None,
        description: str | None = None,
    ) -> Tool:
        """Register a callable or pre-built :class:`Tool`. Returns the Tool."""
        t = func if isinstance(func, Tool) else make_tool(func, name=name, description=description)
        if name:
            t.name = name
        if description:
            t.description = description
        self._tools[t.name] = t
        return t

    def tool(
        self, name: str | None = None, description: str | None = None
    ) -> Callable[[Callable[..., Any]], Tool]:
        """Decorator form that also registers the tool."""

        def decorator(func: Callable[..., Any]) -> Tool:
            return self.register(func, name=name, description=description)

        return decorator

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolNotFoundError(name)
        return self._tools[name]

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return all tool schemas in OpenAI / LiteLLM format."""
        return [t.to_schema() for t in self._tools.values()]

    async def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Execute a tool by name, capturing errors into the result."""
        if name not in self._tools:
            raise ToolNotFoundError(name)
        t = self._tools[name]
        try:
            result = t.func(**(args or {}))
            if inspect.isawaitable(result):
                result = await result
        except ToolError as exc:
            return ToolResult(content=f"Error: {exc}", summary="Tool error", ok=False, error=str(exc))
        except Exception as exc:  # tools must never crash the agent loop
            return ToolResult(
                content=f"Error: {exc}", summary="Execution failed", ok=False, error=str(exc)
            )
        text = str(result)
        return ToolResult(content=text, summary=text[:300], ok=True)
