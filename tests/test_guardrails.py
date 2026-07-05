import pytest

from aetheragents import (
    Agent,
    GuardrailError,
    MockProvider,
    blocklist,
    max_length,
    redact,
)
from aetheragents.core.guardrails import apply_guardrails


def test_max_length_blocks():
    guard = max_length(5)
    assert guard("short") == "short"
    with pytest.raises(GuardrailError):
        guard("too long for this")


def test_max_length_truncates():
    guard = max_length(4, truncate=True)
    assert guard("abcdefgh") == "abcd"


def test_blocklist_case_insensitive():
    guard = blocklist(["secret"])
    assert guard("all fine") == "all fine"
    with pytest.raises(GuardrailError, match="secret"):
        guard("this is SECRET stuff")


def test_redact_replaces_all_occurrences():
    guard = redact(["hunter2"])
    assert guard("pw is hunter2, repeat HUNTER2") == "pw is [REDACTED], repeat [REDACTED]"


def test_guardrails_run_in_order():
    out = apply_guardrails("  hello  ", [str.strip, str.upper])
    assert out == "HELLO"


def test_input_guardrail_transforms_prompt():
    provider = MockProvider(["ok"])
    agent = Agent("a", provider, input_guardrails=[str.upper])
    agent.run("shout this")
    assert provider.calls[0][-1].content == "SHOUT THIS"


def test_input_guardrail_blocks_run():
    provider = MockProvider(["ok"])
    agent = Agent("a", provider, input_guardrails=[blocklist(["forbidden"])])
    with pytest.raises(GuardrailError):
        agent.run("this is forbidden input")
    assert provider.calls == []  # the model was never called


def test_output_guardrail_transforms_result():
    agent = Agent(
        "a",
        MockProvider(["the password is hunter2"]),
        output_guardrails=[redact(["hunter2"])],
    )
    result = agent.run("tell me")
    assert result.output == "the password is [REDACTED]"


def test_output_guardrail_blocks_result():
    agent = Agent(
        "a",
        MockProvider(["leaking a secret here"]),
        output_guardrails=[blocklist(["secret"])],
    )
    with pytest.raises(GuardrailError):
        agent.run("go")
