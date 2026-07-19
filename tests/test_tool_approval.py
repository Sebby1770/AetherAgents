"""Human-in-the-loop tool approval hooks."""

from aetheragents import Agent, MockProvider, tool


@tool()
def secret_action(x: str) -> str:
    "A sensitive tool."
    return f"did:{x}"


@tool()
def safe_action(x: str) -> str:
    "A safe tool."
    return f"safe:{x}"


def test_approval_denies_tool():
    def deny_all(name: str, args: dict) -> bool:
        return False

    provider = MockProvider(
        [[("secret_action", {"x": "nuke"})], "I was denied."]
    )
    agent = Agent("a", provider, tools=[secret_action], tool_approval=deny_all)
    result = agent.run("do it")
    denied = [s for s in result.steps if s.type == "tool_result"]
    assert denied[0].content == "User denied tool execution"
    assert denied[0].ok is False
    assert result.output == "I was denied."


def test_approval_allows_tool():
    def allow_all(name: str, args: dict) -> bool:
        return True

    provider = MockProvider(
        [[("safe_action", {"x": "ok"})], "Done safely."]
    )
    agent = Agent("a", provider, tools=[safe_action], tool_approval=allow_all)
    result = agent.run("do it")
    results = [s for s in result.steps if s.type == "tool_result"]
    assert results[0].content == "safe:ok"
    assert results[0].ok is True


def test_selective_approval():
    def only_safe(name: str, args: dict) -> bool:
        return name == "safe_action"

    provider = MockProvider(
        [
            [("secret_action", {"x": "nope"}), ("safe_action", {"x": "yes"})],
            "mixed",
        ]
    )
    agent = Agent(
        "a",
        provider,
        tools=[secret_action, safe_action],
        tool_approval=only_safe,
    )
    result = agent.run("mix")
    by_name = {s.name: s for s in result.steps if s.type == "tool_result"}
    assert by_name["secret_action"].content == "User denied tool execution"
    assert by_name["safe_action"].content == "safe:yes"


def test_default_none_auto_approves():
    provider = MockProvider([[("safe_action", {"x": "auto"})], "ok"])
    agent = Agent("a", provider, tools=[safe_action])  # tool_approval=None
    result = agent.run("go")
    results = [s for s in result.steps if s.type == "tool_result"]
    assert results[0].content == "safe:auto"
