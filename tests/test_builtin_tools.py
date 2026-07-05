import asyncio
from datetime import datetime

import pytest

from aetheragents import ToolRegistry, builtin_tools
from aetheragents.core.errors import ToolError
from aetheragents.tools import calculator, http_get, utc_now


def test_calculator_basics():
    assert calculator("2 + 3 * 4") == "14"
    assert calculator("(2 + 3) * 4") == "20"
    assert calculator("7 / 2") == "3.5"
    assert calculator("2 ** 10") == "1024"
    assert calculator("-5 % 3") == "1"
    assert calculator("10 // 3") == "3"


def test_calculator_rejects_names_and_calls():
    with pytest.raises(ToolError):
        calculator("__import__('os').system('true')")
    with pytest.raises(ToolError):
        calculator("abs(-1)")
    with pytest.raises(ToolError):
        calculator("'a' + 'b'")


def test_calculator_rejects_huge_exponent():
    with pytest.raises(ToolError):
        calculator("9 ** 999999")


def test_calculator_division_by_zero():
    with pytest.raises(ToolError):
        calculator("1 / 0")


def test_calculator_syntax_error():
    with pytest.raises(ToolError):
        calculator("2 +")


def test_utc_now_is_iso():
    stamp = utc_now()
    parsed = datetime.fromisoformat(stamp)
    assert parsed.tzinfo is not None


def test_http_get_rejects_non_http_schemes():
    with pytest.raises(ToolError):
        http_get("file:///etc/passwd")
    with pytest.raises(ToolError):
        http_get("ftp://example.com/x")


def test_builtin_tools_register_with_schemas():
    reg = ToolRegistry(builtin_tools())
    assert set(reg.names) == {"calculator", "utc_now", "http_get"}
    calc_schema = reg.get("calculator").to_schema()["function"]
    assert calc_schema["parameters"]["required"] == ["expression"]
    assert calc_schema["parameters"]["properties"]["expression"]["type"] == "string"


def test_builtin_tool_error_is_captured_not_raised():
    reg = ToolRegistry(builtin_tools())
    res = asyncio.run(reg.execute("calculator", {"expression": "1 / 0"}))
    assert not res.ok
    assert "zero" in res.error.lower()
