import asyncio

from aetheragents import Message, MockProvider
from aetheragents.core.messages import ToolCall


def _run(provider, messages):
    return asyncio.run(provider.complete(messages))


def test_scripted_text_response():
    p = MockProvider(["hello"])
    r = _run(p, [Message.user("hi")])
    assert r.content == "hello"
    assert r.finish_reason == "stop"
    assert not r.has_tool_calls


def test_tool_call_from_tuple():
    p = MockProvider([[("search", {"q": "x"})]])
    r = _run(p, [Message.user("hi")])
    assert r.has_tool_calls
    assert r.tool_calls[0].name == "search"
    assert r.tool_calls[0].arguments == {"q": "x"}


def test_tool_call_from_object():
    p = MockProvider([ToolCall(id="c1", name="t", arguments={"a": 1})])
    r = _run(p, [Message.user("hi")])
    assert r.tool_calls[0].id == "c1"


def test_dict_with_tool_calls():
    p = MockProvider([{"content": None, "tool_calls": [{"name": "t", "arguments": {"a": 1}}]}])
    r = _run(p, [Message.user("x")])
    assert r.tool_calls[0].name == "t"


def test_handler_inspects_conversation():
    def handler(messages):
        return f"saw {len(messages)} messages"

    p = MockProvider(handler=handler)
    r = _run(p, [Message.user("a"), Message.user("b")])
    assert r.content == "saw 2 messages"


def test_default_echo():
    p = MockProvider(default="")
    r = _run(p, [Message.user("ping")])
    assert "ping" in r.content
