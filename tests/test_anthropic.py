import asyncio
from types import SimpleNamespace

from aetheragents import Message
from aetheragents.core.messages import ToolCall
from aetheragents.llm.anthropic_provider import (
    AnthropicProvider,
    to_anthropic_messages,
    to_anthropic_tools,
)


def test_system_messages_are_lifted_out():
    system, msgs = to_anthropic_messages(
        [Message.system("be nice"), Message.user("hi"), Message.system("be brief")]
    )
    assert system == "be nice\n\nbe brief"
    assert msgs == [{"role": "user", "content": "hi"}]


def test_assistant_tool_calls_become_tool_use_blocks():
    _, msgs = to_anthropic_messages(
        [
            Message.user("compute"),
            Message.assistant(
                content="on it",
                tool_calls=[ToolCall(id="t1", name="add", arguments={"a": 1})],
            ),
        ]
    )
    blocks = msgs[1]["content"]
    assert blocks[0] == {"type": "text", "text": "on it"}
    assert blocks[1] == {"type": "tool_use", "id": "t1", "name": "add", "input": {"a": 1}}


def test_consecutive_tool_results_merge_into_one_user_turn():
    _, msgs = to_anthropic_messages(
        [
            Message.assistant(tool_calls=[
                ToolCall(id="t1", name="a", arguments={}),
                ToolCall(id="t2", name="b", arguments={}),
            ]),
            Message.tool("r1", "t1", name="a"),
            Message.tool("r2", "t2", name="b"),
            Message.assistant(content="done"),
        ]
    )
    assert msgs[1]["role"] == "user"
    assert [b["tool_use_id"] for b in msgs[1]["content"]] == ["t1", "t2"]
    assert msgs[2] == {"role": "assistant", "content": "done"}


def test_tool_schema_conversion():
    openai_style = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers",
                "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}},
            },
        }
    ]
    tools = to_anthropic_tools(openai_style)
    assert tools == [
        {
            "name": "add",
            "description": "Add two numbers",
            "input_schema": {"type": "object", "properties": {"a": {"type": "integer"}}},
        }
    ]


class FakeMessages:
    def __init__(self, response):
        self._response = response
        self.last_params = None

    async def create(self, **params):
        self.last_params = params
        return self._response


def _fake_client(response):
    return SimpleNamespace(messages=FakeMessages(response))


def test_complete_parses_text_and_tool_use():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="thinking... "),
            SimpleNamespace(type="tool_use", id="tu1", name="add", input={"a": 2, "b": 3}),
        ],
        stop_reason="tool_use",
        model="claude-sonnet-5",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    provider = AnthropicProvider(client=_fake_client(response))
    result = asyncio.run(
        provider.complete(
            [Message.system("sys"), Message.user("add 2+3")],
            tools=[{"type": "function", "function": {"name": "add", "parameters": {}}}],
            temperature=0.1,
        )
    )
    assert result.content == "thinking... "
    assert result.tool_calls[0].name == "add"
    assert result.tool_calls[0].arguments == {"a": 2, "b": 3}
    assert result.finish_reason == "tool_use"
    assert result.usage.total_tokens == 15

    params = provider._client.messages.last_params
    assert params["system"] == "sys"
    assert params["temperature"] == 0.1
    assert params["tools"][0]["name"] == "add"
    assert params["messages"] == [{"role": "user", "content": "add 2+3"}]


def test_agent_loop_with_fake_anthropic_client():
    """Full agent loop: the provider requests a tool, then answers."""
    from aetheragents import Agent

    first = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id="tu1", name="add", input={"a": 2, "b": 3})],
        stop_reason="tool_use",
        model="claude-sonnet-5",
        usage=None,
    )
    second = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="The sum is 5.")],
        stop_reason="end_turn",
        model="claude-sonnet-5",
        usage=None,
    )

    class SequenceMessages:
        def __init__(self, responses):
            self._responses = list(responses)

        async def create(self, **params):
            return self._responses.pop(0)

    def add(a: int, b: int) -> int:
        "Add."
        return a + b

    provider = AnthropicProvider(client=SimpleNamespace(messages=SequenceMessages([first, second])))
    agent = Agent("calc", provider, tools=[add])
    result = agent.run("add 2 and 3")
    assert result.output == "The sum is 5."
    assert [s.type for s in result.steps] == ["tool_call", "tool_result", "final"]
