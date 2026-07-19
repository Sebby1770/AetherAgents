"""Session.fork deep-copies history for branching conversations."""

from aetheragents import Agent, Message, MockProvider, Session
from aetheragents.core.messages import Role


def test_fork_copies_messages():
    parent = Session("main")
    parent.append(Message.user("hello"))
    parent.append(Message.assistant("hi there"))

    branch = parent.fork(name="alt")
    assert branch.id == "alt"
    assert len(branch) == 2
    assert [m.content for m in branch.messages] == ["hello", "hi there"]


def test_fork_is_independent():
    parent = Session("main")
    parent.append(Message.user("shared"))

    branch = parent.fork("branch")
    branch.append(Message.user("only-on-branch"))
    parent.append(Message.user("only-on-parent"))

    assert [m.content for m in parent.messages] == ["shared", "only-on-parent"]
    assert [m.content for m in branch.messages] == ["shared", "only-on-branch"]


def test_fork_deep_copy_not_shared_objects():
    parent = Session("main")
    parent.append(Message.user("mutate-me"))
    branch = parent.fork()
    # Mutating the forked message must not touch the parent.
    branch.messages[0].content = "changed"
    assert parent.messages[0].content == "mutate-me"


def test_fork_default_name():
    parent = Session("chat-1")
    branch = parent.fork()
    assert branch.id == "chat-1-fork"


def test_fork_with_agent_runs(tmp_path):
    parent = Session("s", path=tmp_path / "parent.json")
    agent = Agent("a", MockProvider(["first", "on-parent", "on-branch"]))
    agent.run("start", session=parent)

    branch = parent.fork(name="side", path=tmp_path / "branch.json")
    agent.run("parent path", session=parent)
    agent.run("branch path", session=branch)

    parent_contents = [m.content for m in parent.messages if m.role is Role.USER]
    branch_contents = [m.content for m in branch.messages if m.role is Role.USER]
    assert parent_contents == ["start", "parent path"]
    assert branch_contents == ["start", "branch path"]


def test_fork_does_not_inherit_parent_path_by_default():
    parent = Session("s", path="/tmp/parent-session.json")
    branch = parent.fork("b")
    assert branch.path is None
    assert branch.autosave is False
