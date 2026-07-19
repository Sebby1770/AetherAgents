"""Cost budget caps on Agent runs."""

import pytest

from aetheragents import Agent, BudgetExceeded, MockProvider, Usage, register_model_cost
from aetheragents.llm.base import LLMResponse


def test_budget_not_exceeded_when_under_cap():
    # Known model with tiny usage stays under a generous budget.
    agent = Agent(
        "a",
        MockProvider(["short"], model="claude-haiku-4-5"),
        max_cost_usd=1.0,
    )
    result = agent.run("hi")
    assert result.output == "short"
    assert result.cost_usd is not None
    assert result.cost_usd < 1.0


def test_budget_exceeded_raises():
    # Force a huge token count so cost blows past a tiny budget.
    big = LLMResponse(
        content="expensive answer",
        model="claude-opus-4-8",
        usage=Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000),
        finish_reason="stop",
    )
    agent = Agent("spendthrift", MockProvider([big]), max_cost_usd=0.01)
    with pytest.raises(BudgetExceeded) as exc_info:
        agent.run("go")
    err = exc_info.value
    assert err.budget == 0.01
    assert err.spent > err.budget
    assert err.agent == "spendthrift"


def test_run_level_budget_override():
    big = LLMResponse(
        content="x",
        model="claude-opus-4-8",
        usage=Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000),
        finish_reason="stop",
    )
    # Agent has no default budget, but run-level cap fires.
    agent = Agent("a", MockProvider([big]))
    with pytest.raises(BudgetExceeded):
        agent.run("go", max_cost_usd=0.01)


def test_unknown_model_cost_treated_as_zero_skips_budget():
    # MockProvider default model "mock-1" has no price -> cost is None -> spent=0.
    agent = Agent("a", MockProvider(["hello"]), max_cost_usd=0.000001)
    result = agent.run("hi")
    assert result.output == "hello"
    assert result.cost_usd is None


def test_budget_with_registered_mock_model():
    register_model_cost("pricey-mock", 1000.0, 1000.0)  # $1000 / MTok
    # MockProvider usage is ~len//4 completion tokens for "hello" -> 1 token
    # cost = 1 * 1000 / 1e6 = 0.001
    agent = Agent(
        "a",
        MockProvider(["hello"], model="pricey-mock"),
        max_cost_usd=0.0001,
    )
    with pytest.raises(BudgetExceeded):
        agent.run("hi")
