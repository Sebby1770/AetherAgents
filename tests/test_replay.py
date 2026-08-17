"""Session.replay_prompt and Agent.replay re-run the last user turn."""

import pytest

from aetheragents import Agent, Message, MockProvider, Session, tool


def test_replay_prompt_returns_last_user_message():
    session = Session("s")
    session.append(Message.user("first"))
    session.append(Message.assistant("ok"))
    session.append(Message.user("second question"))
    assert session.replay_prompt() == "second question"


def test_replay_prompt_empty_session_raises():
    session = Session("empty")
    with pytest.raises(ValueError, match="no user message"):
        session.replay_prompt()


def test_agent_replay_hits_tools_again():
    hits: list[str] = []

    @tool()
    def ping() -> str:
        "Ping tool."
        hits.append("ping")
        return "pong"

    provider = MockProvider(
        [[("ping", {})], "first", [("ping", {})], "replayed"]
    )
    agent = Agent("a", provider, tools=[ping])
    session = Session()
    first = agent.run("call ping", session=session)
    assert hits == ["ping"]
    assert first.output == "first"
    assert session.replay_prompt() == "call ping"

    result = agent.replay(session)
    assert hits == ["ping", "ping"]
    assert result.output == "replayed"
    tool_steps = [s for s in result.steps if s.type == "tool_call"]
    assert tool_steps[0].name == "ping"


def test_agent_replay_explicit_prompt():
    hits: list[str] = []

    @tool()
    def ping() -> str:
        "Ping tool."
        hits.append("ping")
        return "pong"

    provider = MockProvider(
        [[("ping", {})], "first", [("ping", {})], "again"]
    )
    agent = Agent("a", provider, tools=[ping])
    session = Session()
    agent.run("original", session=session)
    result = agent.replay(session, prompt="explicit replay")
    assert hits == ["ping", "ping"]
    assert result.output == "again"
    assert any(m.content == "explicit replay" for m in session.messages)
