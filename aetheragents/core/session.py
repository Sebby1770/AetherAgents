"""Persistent conversation sessions.

A :class:`Session` holds the message history of an ongoing conversation so an
agent can be resumed across runs - and, with a ``path``, across processes::

    session = Session("support-42", path="sessions/support-42.json")
    agent.run("My printer is on fire", session=session)
    agent.run("It's still on fire",  session=session)   # sees the first turn

Branch with :meth:`Session.fork` to explore alternate paths without mutating
the parent history. Call :meth:`Session.compact` to keep system messages and
the last N user/assistant turns. Persistence is plain JSON, human-readable
and diff-friendly.
"""

from __future__ import annotations

import json
from pathlib import Path

from .messages import Message, Role


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

    def replay_prompt(self) -> str:
        """Return the last user message so an agent can re-run that turn."""
        for msg in reversed(self.messages):
            if msg.role is Role.USER and msg.content:
                return msg.content
        raise ValueError(f"Session {self.id!r} has no user message to replay")

    def fork(
        self,
        name: str | None = None,
        *,
        path: str | Path | None = None,
        autosave: bool | None = None,
    ) -> Session:
        """Deep-copy this session's history into a new branching session.

        The forked session starts with an independent copy of every message so
        further runs on either branch do not affect the other. ``name`` becomes
        the new session id (defaults to ``"{id}-fork"``). Persistence settings
        default to *no* autosave / path unless explicitly provided, so forks do
        not overwrite the parent file by accident.
        """
        if autosave is None:
            # Only autosave by default when a path was given for the fork.
            autosave = path is not None
        forked = Session(
            id=name or f"{self.id}-fork",
            path=path,
            autosave=autosave,
        )
        # model_copy(deep=True) clones each pydantic Message independently.
        forked.messages = [m.model_copy(deep=True) for m in self.messages]
        return forked

    def __len__(self) -> int:
        return len(self.messages)

    def compact(self, keep_last: int = 4) -> Session:
        """Drop older turns, keeping system messages and recent dialogue.

        All ``system`` messages are retained. Of the remaining history, only
        the last ``keep_last`` user/assistant turns are kept. A turn is a
        user message plus the assistant/tool messages that follow it (or,
        if there is no user, a trailing non-user group).

        Returns ``self`` for chaining. Autosaves when a path is configured.
        """
        if keep_last < 0:
            raise ValueError("keep_last must be >= 0")

        systems = [m for m in self.messages if m.role is Role.SYSTEM]
        rest = [m for m in self.messages if m.role is not Role.SYSTEM]
        turns: list[list[Message]] = []
        current: list[Message] = []
        for msg in rest:
            if msg.role is Role.USER and current:
                turns.append(current)
                current = [msg]
            else:
                current.append(msg)
        if current:
            turns.append(current)
        kept = turns[-keep_last:] if keep_last else []
        self.messages = systems + [m for turn in kept for m in turn]
        self.maybe_autosave()
        return self

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
