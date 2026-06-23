import asyncio

import pytest

from aetheragents import ToolNotFoundError, ToolRegistry, tool
from aetheragents.core.tools import build_parameters


def add(a: int, b: int = 2) -> int:
    "Add two numbers."
    return a + b


def test_schema_required_and_optional():
    params = build_parameters(add)
    assert params["properties"]["a"]["type"] == "integer"
    assert params["properties"]["b"]["type"] == "integer"
    assert params["required"] == ["a"]  # b has a default -> not required


def test_optional_union_not_required():
    def f(x: str | None = None):
        return x

    params = build_parameters(f)
    assert params["properties"]["x"]["type"] == "string"
    assert "required" not in params


def test_list_annotation_is_array():
    def f(items: list[str]):
        return items

    params = build_parameters(f)
    assert params["properties"]["items"]["type"] == "array"
    assert params["required"] == ["items"]


def test_execute_sync_tool():
    reg = ToolRegistry([add])
    res = asyncio.run(reg.execute("add", {"a": 3, "b": 4}))
    assert res.ok
    assert res.content == "7"


def test_execute_async_tool():
    async def fetch(url: str) -> str:
        return f"got {url}"

    reg = ToolRegistry([fetch])
    res = asyncio.run(reg.execute("fetch", {"url": "x"}))
    assert res.content == "got x"


def test_execute_captures_error():
    reg = ToolRegistry([add])
    res = asyncio.run(reg.execute("add", {"a": "oops"}))
    assert not res.ok
    assert res.error


def test_unknown_tool_raises():
    reg = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        asyncio.run(reg.execute("nope", {}))


def test_decorator_registers_with_schema():
    @tool(description="Adds two integers")
    def addx(a: int, b: int) -> int:
        return a + b

    reg = ToolRegistry([addx])
    schema = reg.get_schemas()[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "addx"
    assert schema["function"]["description"] == "Adds two integers"
    assert schema["function"]["parameters"]["required"] == ["a", "b"]
