import asyncio

import pytest

from aetheragents import ToolRegistry, file_tools
from aetheragents.core.errors import ToolError
from aetheragents.tools.files import _resolve


def make_registry(tmp_path, **kwargs):
    return ToolRegistry(file_tools(tmp_path, **kwargs))


def run(reg, name, args):
    return asyncio.run(reg.execute(name, args))


def test_write_read_list_roundtrip(tmp_path):
    reg = make_registry(tmp_path)
    assert run(reg, "write_file", {"path": "notes/a.txt", "content": "hello"}).ok
    read = run(reg, "read_file", {"path": "notes/a.txt"})
    assert read.content == "hello"
    listing = run(reg, "list_files", {"path": "notes"})
    assert "a.txt" in listing.content


def test_traversal_rejected(tmp_path):
    root = tmp_path / "sandbox"
    reg = make_registry(root)
    res = run(reg, "read_file", {"path": "../outside.txt"})
    assert not res.ok
    assert "escape" in res.error.lower()


def test_absolute_path_rejected(tmp_path):
    reg = make_registry(tmp_path)
    res = run(reg, "read_file", {"path": "/etc/passwd"})
    assert not res.ok
    assert "absolute" in res.error.lower()


def test_resolve_allows_root_itself(tmp_path):
    assert _resolve(tmp_path.resolve(), ".") == tmp_path.resolve()


def test_readonly_omits_write_tool(tmp_path):
    reg = make_registry(tmp_path, readonly=True)
    assert "write_file" not in reg.names
    assert set(reg.names) == {"read_file", "list_files"}


def test_missing_file_and_directory(tmp_path):
    reg = make_registry(tmp_path)
    assert not run(reg, "read_file", {"path": "nope.txt"}).ok
    assert not run(reg, "list_files", {"path": "nope"}).ok


def test_symlink_escape_rejected(tmp_path):
    root = tmp_path / "sandbox"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    reg = make_registry(root)
    (root / "link").symlink_to(outside)
    res = run(reg, "read_file", {"path": "link/secret.txt"})
    assert not res.ok


def test_write_size_cap(tmp_path):
    reg = make_registry(tmp_path)
    res = run(reg, "write_file", {"path": "big.txt", "content": "x" * 1_000_001})
    assert not res.ok
    assert "large" in res.error.lower()


def test_resolve_is_toolerror_not_generic(tmp_path):
    with pytest.raises(ToolError):
        _resolve(tmp_path.resolve(), "../evil")
