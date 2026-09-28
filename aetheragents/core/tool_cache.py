"""Memoize tool calls by name and arguments.

Identical arguments return the first result without calling the function
again. The cache is process-local and thread-safe. It does not persist.
"""

from __future__ import annotations

import inspect
import json
import threading
from typing import Any

from .tools import Tool, make_tool


class ToolCache:
    """Memoises tool return values for identical keyword arguments."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: dict[tuple[str, str], Any] = {}
        self.hits = 0
        self.misses = 0

    def wrap(self, tool: Tool | Any) -> Tool:
        """Return a tool with the same schema whose function is cached."""
        wrapped_tool = tool if isinstance(tool, Tool) else make_tool(tool)
        original = wrapped_tool.func
        cache = self
        name = wrapped_tool.name

        def lookup(kwargs: dict[str, Any]) -> tuple[tuple[str, str], Any, bool]:
            key = (name, _canonical(kwargs))
            with cache._lock:
                if key in cache._store:
                    cache.hits += 1
                    return key, cache._store[key], True
            return key, None, False

        def remember(key: tuple[str, str], result: Any) -> Any:
            with cache._lock:
                cache.misses += 1
                cache._store[key] = result
            return result

        if wrapped_tool.is_async:

            async def cached(**kwargs: Any) -> Any:
                key, found, hit = lookup(kwargs)
                if hit:
                    return found
                result = original(**kwargs)
                if inspect.isawaitable(result):
                    result = await result
                return remember(key, result)

            func = cached
        else:

            def cached(**kwargs: Any) -> Any:
                key, found, hit = lookup(kwargs)
                if hit:
                    return found
                return remember(key, original(**kwargs))

            func = cached

        return Tool(
            name=wrapped_tool.name,
            description=wrapped_tool.description,
            func=func,
            parameters=wrapped_tool.parameters,
            is_async=wrapped_tool.is_async,
        )

    def wrap_all(self, tools: list[Tool | Any]) -> list[Tool]:
        return [self.wrap(tool) for tool in tools]

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0


def _canonical(kwargs: dict[str, Any]) -> str:
    return json.dumps(kwargs, sort_keys=True, default=str, separators=(",", ":"))
