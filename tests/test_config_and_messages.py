import json

from aetheragents import Message, Settings
from aetheragents.core.messages import ToolCall


def test_settings_defaults():
    s = Settings()
    assert s.default_model == "gpt-4o"
    assert s.max_steps == 8


def test_settings_from_env():
    s = Settings.from_env(
        {"AETHER_DEFAULT_MODEL": "claude-3", "AETHER_MAX_STEPS": "5", "OPENAI_API_KEY": "sk-x"}
    )
    assert s.default_model == "claude-3"
    assert s.max_steps == 5
    assert s.openai_api_key == "sk-x"


def test_user_message_provider_dict():
    assert Message.user("hi").to_provider_dict() == {"role": "user", "content": "hi"}


def test_assistant_tool_call_dict_omits_none_content():
    m = Message.assistant(tool_calls=[ToolCall(id="c1", name="f", arguments={"a": 1})])
    d = m.to_provider_dict()
    assert d["role"] == "assistant"
    assert "content" not in d
    fn = d["tool_calls"][0]["function"]
    assert fn["name"] == "f"
    assert json.loads(fn["arguments"]) == {"a": 1}


def test_tool_message_dict():
    d = Message.tool("result", "c1", name="f").to_provider_dict()
    assert d["role"] == "tool"
    assert d["tool_call_id"] == "c1"
    assert d["content"] == "result"
