"""Shared team memory for multi-agent runs.

A :class:`Blackboard` is a small JSON-safe key/value store. Bind it to an
agent to expose ``blackboard_read`` and ``blackboard_write`` tools, or pass it
to :meth:`Orchestrator.workflow` so a step can ``save_as`` a key for later
agents.

Values are copied on the way in and out. Keys are short identifiers.
"""

from __future__ import annotations

import copy
import json
import re
import threading
from typing import Any

from .errors import AetherError

_KEY_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_MAX_TEXT = 8000
_MAX_ITEMS = 200
_MAX_DEPTH = 8


class BlackboardError(AetherError):
    """A blackboard key or value was rejected."""


class Blackboard:
    """Thread-safe JSON-safe scratchpad shared by a team of agents."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()

    def put(self, key: str, value: Any) -> None:
        """Store a copy of ``value`` under ``key``."""
        checked = _check_key(key)
        _validate(value)
        stored = copy.deepcopy(value)
        with self._lock:
            self._data[checked] = stored

    def get(self, key: str, default: Any = None) -> Any:
        """Return a copy of the value, or ``default`` when the key is absent."""
        checked = _check_key(key)
        with self._lock:
            if checked not in self._data:
                return default
            return copy.deepcopy(self._data[checked])

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data)

    def snapshot(self) -> dict[str, Any]:
        """Return a deep copy of every entry."""
        with self._lock:
            return copy.deepcopy(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def bind(self, agent: Any) -> Any:
        """Register read/write tools on ``agent`` and return it.

        ``blackboard_write`` accepts a string. JSON objects, arrays, numbers,
        booleans and null are stored as those types; anything else is stored
        as text.
        """
        board = self

        def blackboard_read(key: str) -> str:
            """Read one blackboard key. Returns the text 'missing' if absent."""
            value = board.get(key, None)
            if value is None and key not in board.keys():
                return "missing"
            if isinstance(value, str):
                return value
            return json.dumps(value, sort_keys=True)

        def blackboard_write(key: str, value: str) -> str:
            """Write one blackboard key. JSON values are stored structured."""
            board.put(key, _decode_tool_value(value))
            return f"stored {key}"

        agent.tools.register(blackboard_read)
        agent.tools.register(blackboard_write)
        return agent


def _check_key(key: str) -> str:
    if not isinstance(key, str) or not _KEY_RE.fullmatch(key):
        raise BlackboardError(
            "key must be 1-64 characters of letters, digits, '_', '.', ':' or '-'"
        )
    return key


def _validate(value: Any, depth: int = 0) -> None:
    if depth > _MAX_DEPTH:
        raise BlackboardError("value is nested too deeply")
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, str):
        if len(value) > _MAX_TEXT:
            raise BlackboardError("value is too long")
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise BlackboardError("value must be finite")
        return
    if isinstance(value, list):
        if len(value) > _MAX_ITEMS:
            raise BlackboardError("list is too long")
        for item in value:
            _validate(item, depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > _MAX_ITEMS:
            raise BlackboardError("object is too large")
        for item_key, item_value in value.items():
            if not isinstance(item_key, str):
                raise BlackboardError("object keys must be strings")
            _validate(item_value, depth + 1)
        return
    raise BlackboardError(f"unsupported value type: {type(value).__name__}")


def _decode_tool_value(value: str) -> Any:
    if not isinstance(value, str):
        value = str(value)
    if len(value) > _MAX_TEXT:
        raise BlackboardError("value is too long")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(parsed, (dict, list, int, float, bool)) or parsed is None:
        _validate(parsed)
        return parsed
    return value
