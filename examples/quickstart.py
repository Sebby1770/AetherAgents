"""Quickstart: a single agent that uses a tool.

Runs fully offline with MockProvider - no API key required::

    python examples/quickstart.py

To use a real model instead, install the extra and swap the provider::

    pip install 'aetheragents[litellm]'
    export OPENAI_API_KEY=sk-...

    from aetheragents import LiteLLMProvider
    provider = LiteLLMProvider("gpt-4o")
"""

from aetheragents import Agent, MockProvider, tool


@tool()
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def main() -> None:
    # The mock is scripted: first it asks to call `multiply`, then it answers.
    provider = MockProvider(
        [
            [("multiply", {"a": 6, "b": 7})],
            "6 times 7 is 42.",
        ]
    )
    agent = Agent(
        "calculator",
        provider,
        instructions="You are a precise calculator. Use tools for arithmetic.",
        tools=[multiply],
    )

    result = agent.run("What is 6 times 7?")
    print("Answer:", result.output)
    print("Tokens:", result.usage.total_tokens)
    print("\nTrace:")
    for step in result.steps:
        if step.type == "tool_call":
            print(f"  -> call {step.name}({step.arguments})")
        elif step.type == "tool_result":
            print(f"  <- {step.name} returned {step.content}")
        else:
            print(f"  = final: {step.content}")


if __name__ == "__main__":
    main()
