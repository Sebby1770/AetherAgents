import pytest
from pydantic import BaseModel

from aetheragents import Agent, MockProvider, StructuredOutputError, extract_json
from aetheragents.core.messages import Role


class Person(BaseModel):
    name: str
    age: int


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    text = 'Here you go:\n```json\n{"a": 1}\n```\nDone.'
    assert extract_json(text) == {"a": 1}


def test_extract_json_embedded_in_prose():
    assert extract_json('Sure! {"name": "Ada"} hope that helps') == {"name": "Ada"}


def test_extract_json_array():
    assert extract_json("the list is [1, 2, 3] ok") == [1, 2, 3]


def test_extract_json_failure():
    with pytest.raises(ValueError):
        extract_json("no json here at all")


def test_agent_structured_success():
    agent = Agent("a", MockProvider(['{"name": "Ada", "age": 36}']))
    result = agent.run("who?", response_model=Person)
    assert result.parsed == Person(name="Ada", age=36)


def test_schema_instruction_in_system_message():
    provider = MockProvider(['{"name": "Ada", "age": 36}'])
    agent = Agent("a", provider, instructions="Be brief.")
    agent.run("who?", response_model=Person)
    system = provider.calls[0][0]
    assert system.role is Role.SYSTEM
    assert "JSON Schema" in system.content
    assert "Be brief." in system.content


def test_agent_structured_retry_then_success():
    provider = MockProvider(["not json at all", '{"name": "Ada", "age": 36}'])
    agent = Agent("a", provider)
    result = agent.run("who?", response_model=Person)
    assert result.parsed.name == "Ada"
    assert len(provider.calls) == 2
    # The correction turn tells the model what was wrong.
    correction = provider.calls[1][-1]
    assert correction.role is Role.USER
    assert "rejected" in correction.content


def test_agent_structured_retries_exhausted():
    provider = MockProvider(["bad", "still bad", "nope"])
    agent = Agent("a", provider)
    with pytest.raises(StructuredOutputError):
        agent.run("who?", response_model=Person, structured_retries=2)


def test_structured_with_tool_use():
    def lookup(name: str) -> str:
        "Look up a person."
        return "Ada, 36"

    provider = MockProvider([[("lookup", {"name": "Ada"})], '{"name": "Ada", "age": 36}'])
    agent = Agent("a", provider, tools=[lookup])
    result = agent.run("who is Ada?", response_model=Person)
    assert result.parsed.age == 36
    assert result.tool_calls
