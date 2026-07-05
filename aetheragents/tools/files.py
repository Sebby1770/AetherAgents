"""Sandboxed file tools: read/write/list confined to a root directory.

Every model-supplied path is resolved to its canonical form and must remain
inside the sandbox root - traversal (``..``), absolute paths, and symlink
escapes are rejected with :class:`~aetheragents.core.errors.ToolError`.

Usage::

    from aetheragents.tools import file_tools

    agent = Agent("writer", provider, tools=file_tools("./workspace"))
    agent = Agent("reader", provider, tools=file_tools("./docs", readonly=True))
"""

from __future__ import annotations

from pathlib import Path

from ..core.errors import ToolError
from ..core.tools import Tool, make_tool

_MAX_READ_BYTES = 1_000_000
_MAX_WRITE_BYTES = 1_000_000


def _resolve(root: Path, path: str) -> Path:
    if Path(path).is_absolute():
        raise ToolError("Absolute paths are not allowed; use a path relative to the sandbox root")
    target = (root / path).resolve()
    if target != root and root not in target.parents:
        raise ToolError(f"Path escapes the sandbox root: {path!r}")
    return target


def file_tools(root: str | Path, *, readonly: bool = False) -> list[Tool]:
    """Build file tools confined to ``root``. Omit the write tool with ``readonly``."""
    base = Path(root).resolve()
    base.mkdir(parents=True, exist_ok=True)

    def read_file(path: str) -> str:
        """Read a text file from the sandbox. Path is relative to the sandbox root."""
        target = _resolve(base, path)
        if not target.is_file():
            raise ToolError(f"No such file: {path!r}")
        if target.stat().st_size > _MAX_READ_BYTES:
            raise ToolError(f"File too large to read (max {_MAX_READ_BYTES} bytes)")
        return target.read_text(encoding="utf-8", errors="replace")

    def write_file(path: str, content: str) -> str:
        """Write a text file in the sandbox, creating parent directories as needed."""
        if len(content.encode("utf-8")) > _MAX_WRITE_BYTES:
            raise ToolError(f"Content too large to write (max {_MAX_WRITE_BYTES} bytes)")
        target = _resolve(base, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {path}"

    def list_files(path: str = ".") -> str:
        """List files and directories at a path relative to the sandbox root."""
        target = _resolve(base, path)
        if not target.is_dir():
            raise ToolError(f"No such directory: {path!r}")
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
        lines = [f"{entry.name}/" if entry.is_dir() else entry.name for entry in entries]
        return "\n".join(lines) if lines else "(empty)"

    tools = [make_tool(read_file), make_tool(list_files)]
    if not readonly:
        tools.append(make_tool(write_file))
    return tools
