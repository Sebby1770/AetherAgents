from aetheragents import Agent, Message, MockProvider, Session
from aetheragents.core.messages import Role


def test_session_carries_history_between_runs():
    provider = MockProvider(["first answer", "second answer"])
    agent = Agent("a", provider)
    session = Session("s1")

    agent.run("first question", session=session)
    agent.run("second question", session=session)

    # The second call must include the whole first exchange.
    second_call = provider.calls[1]
    contents = [m.content for m in second_call]
    assert "first question" in contents
    assert "first answer" in contents
    assert contents[-1] == "second question"

    # Session accumulated both exchanges in order.
    roles = [m.role for m in session.messages]
    assert roles == [Role.USER, Role.ASSISTANT, Role.USER, Role.ASSISTANT]


def test_session_records_tool_turns():
    def add(a: int, b: int) -> int:
        "Add."
        return a + b

    provider = MockProvider([[("add", {"a": 1, "b": 2})], "3 it is"])
    agent = Agent("a", provider, tools=[add])
    session = Session()
    agent.run("1+2?", session=session)
    roles = [m.role for m in session.messages]
    assert roles == [Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT]


def test_session_with_memory_does_not_duplicate_history():
    from aetheragents import MemoryManager

    provider = MockProvider(["first answer", "second answer"])
    agent = Agent("a", provider, memory=MemoryManager("dedup"))
    session = Session("dedup")
    agent.run("alpha question", session=session)
    agent.run("beta question", session=session)

    second_call = [m.content for m in provider.calls[1] if m.content]
    # The first turn must appear exactly once (from the session), not twice
    # (session + memory recent-context).
    assert second_call.count("alpha question") == 1
    assert second_call.count("first answer") == 1


def test_session_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "s.json"
    session = Session("roundtrip", path=path, autosave=False)
    session.append(Message.user("hello"))
    session.append(Message.assistant("hi there"))
    session.save()

    loaded = Session.load(path)
    assert loaded.id == "roundtrip"
    assert [m.content for m in loaded.messages] == ["hello", "hi there"]
    assert loaded.messages[0].role is Role.USER


def test_session_autosave_on_agent_run(tmp_path):
    path = tmp_path / "auto.json"
    session = Session("auto", path=path)
    agent = Agent("a", MockProvider(["ok"]))
    agent.run("save me", session=session)
    assert path.exists()
    loaded = Session.load(path)
    assert len(loaded) == 2


def test_constructor_loads_existing_file(tmp_path):
    path = tmp_path / "existing.json"
    first = Session("x", path=path)
    first.append(Message.user("remembered"))
    first.save()

    resumed = Session("x", path=path)
    assert [m.content for m in resumed.messages] == ["remembered"]


def test_save_without_path_raises(tmp_path):
    session = Session()
    try:
        session.save()
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # but an explicit path works
    out = session.save(tmp_path / "explicit.json")
    assert out.exists()
