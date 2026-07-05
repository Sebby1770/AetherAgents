from aetheragents import Agent, MockProvider, Usage, estimate_cost, register_model_cost


def test_known_anthropic_models():
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert estimate_cost("claude-opus-4-8", usage) == 30.00      # 5 + 25
    assert estimate_cost("claude-sonnet-4-6", usage) == 18.00    # 3 + 15
    assert estimate_cost("claude-haiku-4-5", usage) == 6.00      # 1 + 5


def test_longest_prefix_wins():
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=0)
    # claude-haiku-4-5 (specific) not claude-haiku (generic) — same price here,
    # but a dated variant must resolve through the specific prefix.
    assert estimate_cost("claude-haiku-4-5-20251001", usage) == 1.00


def test_provider_prefixes_are_stripped():
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=0)
    assert estimate_cost("anthropic.claude-opus-4-8", usage) == 5.00
    assert estimate_cost("litellm:gpt-4o", usage) == 2.50


def test_unknown_model_returns_none():
    usage = Usage(prompt_tokens=100, completion_tokens=100)
    assert estimate_cost("some-mystery-model", usage) is None
    assert estimate_cost(None, usage) is None
    assert estimate_cost("", usage) is None


def test_register_model_cost_override():
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    register_model_cost("my-local-model", 0.0, 0.0)
    assert estimate_cost("my-local-model-v2", usage) == 0.0


def test_agent_result_carries_model_and_cost():
    agent = Agent("a", MockProvider(["four words of answer"], model="claude-opus-4-8"))
    result = agent.run("hi")
    assert result.model == "claude-opus-4-8"
    assert result.cost_usd is not None and result.cost_usd > 0


def test_agent_result_unknown_model_cost_is_none():
    agent = Agent("a", MockProvider(["answer"]))  # default mock-1 model
    result = agent.run("hi")
    assert result.model == "mock-1"
    assert result.cost_usd is None
