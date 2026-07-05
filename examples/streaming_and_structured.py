"""v0.3 features: streaming, structured output, sessions and built-in tools.

Runs fully offline with MockProvider::

    python examples/streaming_and_structured.py
"""

import asyncio

from pydantic import BaseModel

from aetheragents import Agent, MockProvider, Session, builtin_tools


class CityFacts(BaseModel):
    city: str
    country: str
    population_millions: float


async def demo_streaming() -> None:
    print("== Streaming ==")
    agent = Agent("narrator", MockProvider(["Once upon a time an agent streamed its answer."]))
    async for event in agent.astream("tell me a story"):
        if event.type == "delta":
            print(event.delta, end="", flush=True)
        elif event.type == "result":
            print(f"\n(done - {event.result.usage.total_tokens} tokens)\n")


def demo_structured() -> None:
    print("== Structured output ==")
    provider = MockProvider(
        [
            "Melbourne is a city in Australia with about 5.2M people.",  # rejected: not JSON
            '{"city": "Melbourne", "country": "Australia", "population_millions": 5.2}',
        ]
    )
    agent = Agent("geo", provider)
    result = agent.run("Tell me about Melbourne", response_model=CityFacts)
    facts = result.parsed
    print(f"parsed -> {facts.city}, {facts.country}, {facts.population_millions}M")
    print("(first answer was rejected and retried automatically)\n")


def demo_session_and_tools() -> None:
    print("== Session + built-in calculator ==")
    provider = MockProvider(
        [
            [("calculator", {"expression": "17 * 23"})],
            "17 * 23 = 391.",
            "As I said, the answer was 391.",
        ]
    )
    agent = Agent("calc", provider, tools=builtin_tools())
    session = Session("demo")
    print("Q1:", agent.run("what is 17 * 23?", session=session).output)
    print("Q2:", agent.run("what did you just tell me?", session=session).output)
    print(f"(session now holds {len(session)} messages)")


if __name__ == "__main__":
    asyncio.run(demo_streaming())
    demo_structured()
    demo_session_and_tools()
