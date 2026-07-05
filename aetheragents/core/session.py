"""Persistent conversation sessions.

A :class:`Session` holds the message history of an ongoing conversation so an
agent can be resumed across runs - and, with a ``path``, across processes::

    session = Session("support-42", path="sessions/support-42.json")
    agent.run("My printer is on fire", session=session)
    agent.run("It's still on fire",  session=session)   # sees the first turn

Persistence is plain JSON, human-readable and diff-friendly.
"""

from __future__ import annotations

import json
from pathlib import Path

from .messages import Message


class Session:
    """An ordered message history with optional JSON file persistence."""

    def __init__(
        self,
        id: str = "default",
        *,
        path: str | Path | None = None,
        autosave: bool = True,
    ) -> None:
        self.id = id
        self.path = Path(path) if path else None
        self.autosave = autosave
        self.messages: list[Message] = []
        if self.path is not None and self.path.exists():
            self._load_from(self.path)

    # -- history ------------------------------------------------------------
    def append(self, message: Message) -> None:
        self.messages.append(message)

    def extend(self, messages: list[Message]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages.clear()

    def __len__(self) -> int:
        return len(self.messages)

    # -- persistence ----------------------------------------------------------
    def save(self, path: str | Path | None = None) -> Path:
        """Write the session to ``path`` (or the configured path) as JSON."""
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("Session has no path; pass one to save() or the constructor.")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": self.id,
            "messages": [m.model_dump(mode="json", exclude_none=True) for m in self.messages],
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    def maybe_autosave(self) -> None:
        if self.autosave and self.path is not None:
            self.save()

    @classmethod
    def load(cls, path: str | Path) -> Session:
        """Load a session from a JSON file written by :meth:`save`."""
        session = cls.__new__(cls)
        session.path = Path(path)
        session.autosave = True
        session.messages = []
        session.id = "default"
        session._load_from(session.path)
        return session

    def _load_from(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.id = data.get("id", self.id)
        self.messages = [Message.model_validate(m) for m in data.get("messages", [])]
