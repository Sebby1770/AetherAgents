"""Built-in tools: safe, dependency-free capabilities for agents.

* :func:`calculator` - arithmetic via a whitelisted AST walk (never ``eval``)
* :func:`utc_now`    - the current UTC timestamp
* :func:`http_get`   - fetch a http(s) URL with size and time limits

Use :func:`builtin_tools` to get them as ready-to-register :class:`Tool`\\ s::

    agent = Agent("helper", provider, tools=builtin_tools())
"""

from __future__ import annotations

import ast
import operator
import urllib.request
from datetime import datetime, timezone
from urllib.parse import urlparse

from ..core.errors import ToolError
from ..core.tools import Tool, make_tool

_MAX_EXPRESSION_LENGTH = 500
_MAX_POW_EXPONENT = 1000

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError(f"Only numbers are allowed, got {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXPONENT:
            raise ToolError(f"Exponent too large (max {_MAX_POW_EXPONENT})")
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ToolError(f"Unsupported syntax: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression (+, -, *, /, //, %, **, parentheses)."""
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ToolError(f"Expression too long (max {_MAX_EXPRESSION_LENGTH} characters)")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"Invalid expression: {exc.msg}") from exc
    try:
        result = _eval_node(tree)
    except ZeroDivisionError as exc:
        raise ToolError("Division by zero") from exc
    except OverflowError as exc:
        raise ToolError("Result too large") from exc
    # Render integers without a trailing .0 for cleaner model consumption.
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        return str(int(result))
    return str(result)


def utc_now() -> str:
    """Get the current date and time in UTC (ISO 8601)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def http_get(url: str, max_bytes: int = 65536) -> str:
    """Fetch the body of an http(s) URL as text (truncated to max_bytes)."""
    scheme = urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise ToolError(f"Only http/https URLs are allowed, got scheme {scheme!r}")
    req = urllib.request.Request(url, headers={"User-Agent": "aetheragents/0.3"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 - scheme checked above
            body = resp.read(max_bytes)
    except OSError as exc:
        raise ToolError(f"Fetch failed: {exc}") from exc
    return body.decode("utf-8", errors="replace")


def builtin_tools() -> list[Tool]:
    """All built-in tools, ready to pass to an :class:`Agent` or ``ToolRegistry``."""
    return [make_tool(calculator), make_tool(utc_now), make_tool(http_get)]
