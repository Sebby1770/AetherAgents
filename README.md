# AetherAgents

**A lightweight, provider-agnostic multi-agent orchestration framework for Python.**

[![CI](https://github.com/Sebby1770/AetherAgents/actions/workflows/ci.yml/badge.svg)](https://github.com/Sebby1770/AetherAgents/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

AetherAgents lets you build tool-using agents and compose them into teams, with a
tiny core (only `pydantic`) and **zero required network dependencies**. Drop in a
real model via [LiteLLM](https://docs.litellm.ai) when you're ready — or stay
fully offline with the built-in deterministic `MockProvider` for tests and demos.

```python
from aetheragents import Agent, MockProvider, tool

@tool()
def add(a: int, b: int) -> int:
    "Add two numbers."
    return a + b

agent = Agent("calc", MockProvider([[("add", {"a": 2, "b": 2})], "It's 4."]), tools=[add])
print(agent.run("What is 2 + 2?").output)   # -> It's 4.
```

---

## Why AetherAgents

| | |
|---|---|
| 🪶 **Tiny core** | The base install depends only on `pydantic`. Backends (LiteLLM, ChromaDB, FastAPI, OpenTelemetry) are opt-in extras. |
| 🔌 **Provider-agnostic** | One `LLMProvider` interface. `LiteLLMProvider` reaches 100+ models; `MockProvider` runs offline. |
| 🛠️ **Real tool schemas** | The `@tool` decorator generates JSON-Schema from your function signature and type hints — no hand-written specs. |
| 🤝 **Multi-agent** | Sequential pipelines, parallel fan-out, routing, and agent-as-tool delegation. |
| 🧠 **Pluggable memory** | Rolling short-term window + long-term vector recall. Embedding-free in-memory store by default; ChromaDB optional. |
| ✅ **Testable** | Deterministic mock provider + 35 unit tests means agent logic is testable without API keys or flakiness. |
| 🔭 **Observable** | Optional OpenTelemetry tracing; structured step traces and token accounting on every run. |

## Install

```bash
pip install aetheragents                 # tiny core (pydantic only)
pip install 'aetheragents[litellm]'      # real models via LiteLLM
pip install 'aetheragents[chroma]'       # ChromaDB-backed long-term memory
pip install 'aetheragents[server]'       # FastAPI HTTP server
pip install 'aetheragents[all,dev]'      # everything + test/lint tooling
```

> Until it's published to PyPI, install from source: `pip install -e '.[dev]'`.

## Core concepts

### Agents

An `Agent` runs a reasoning loop: call the model, execute any requested tools,
feed results back, repeat until the model answers or `max_steps` is hit. Every
run returns a structured `AgentResult` with the final output, a step-by-step
trace and token usage.

```python
from aetheragents import Agent, LiteLLMProvider, tool

@tool()
def get_weather(city: str) -> str:
    "Get the current weather for a city."
    return f"It's sunny in {city}."

agent = Agent(
    "assistant",
    LiteLLMProvider("gpt-4o"),               # needs: pip install 'aetheragents[litellm]'
    instructions="You are a helpful assistant.",
    tools=[get_weather],
    max_steps=6,
)

result = agent.run("What's the weather in Melbourne?")
print(result.output)
for step in result.steps:
    print(step.type, step.name or "", step.content or step.arguments)
```

### Tools

Decorate any function with `@tool()`. The argument schema is derived from the
signature — required vs optional, and JSON types from your annotations.

```python
@tool(description="Search the knowledge base.")
def search(query: str, limit: int = 5) -> list[str]:
    ...
# -> parameters: {query: string (required), limit: integer}
```

### Multi-agent orchestration

```python
import asyncio
from aetheragents import Agent, MockProvider, Orchestrator, keyword_router

team = Orchestrator([
    Agent("researcher", MockProvider(handler=lambda m: "facts...")),
    Agent("writer", MockProvider(handler=lambda m: "# Article")),
])

# Pipeline: researcher's output feeds the writer
final = asyncio.run(team.sequential("Write about X", order=["researcher", "writer"]))[-1]

# Fan-out: run everyone on the same task
results = asyncio.run(team.parallel("Summarise X"))

# Route: pick one agent by keyword
router = keyword_router({"write": "writer"}, default="researcher")
chosen = asyncio.run(team.route("please write a post", selector=router))
```

**Delegation** — expose any agent as a tool so a "manager" agent can call it:

```python
manager = Agent("manager", provider, tools=[team["researcher"].as_tool()])
```

### Memory

```python
from aetheragents import Agent, MemoryManager, MockProvider

mem = MemoryManager("assistant")            # in-memory vector store by default
agent = Agent("assistant", MockProvider(["ok"]), memory=mem)
agent.run("Remember: my favourite colour is teal.")
# Later runs automatically recall relevant past messages.
```

Use ChromaDB for persistence:

```python
from aetheragents import MemoryManager
from aetheragents.core.memory import ChromaVectorStore   # needs [chroma]

mem = MemoryManager("assistant", vector_store=ChromaVectorStore("assistant"))
```

### Serving over HTTP

```python
from aetheragents import Agent, MockProvider
from aetheragents.server import create_app     # needs [server]

app = create_app({"echo": Agent("echo", MockProvider(default="hi"))})
# uvicorn mymodule:app   ->  POST /agents/echo/run  {"prompt": "..."}
```

## Architecture

```
                ┌─────────────────────────────────────────────┐
                │                Orchestrator                  │
                │   sequential · parallel · route · delegate   │
                └───────────────┬──────────────┬──────────────┘
                                │              │
                        ┌───────▼──────┐ ┌─────▼────────┐
                        │    Agent     │ │    Agent     │   reasoning loop
                        └───┬──────┬───┘ └──────────────┘
                            │      │
              ┌─────────────▼─┐  ┌─▼──────────────┐  ┌──────────────────┐
              │ ToolRegistry  │  │ MemoryManager  │  │   LLMProvider    │
              │ (auto schema) │  │ short + vector │  │ Mock | LiteLLM   │
              └───────────────┘  └────────────────┘  └──────────────────┘
```

## Development

```bash
pip install -e '.[dev]'
pytest          # 35 tests, fully offline
ruff check .    # lint
```

Run the examples:

```bash
python examples/quickstart.py
python examples/research_team.py
```

## Roadmap

- Streaming responses (`provider.stream`)
- Structured output / response models
- Built-in tools (HTTP fetch, Python sandbox)
- Persistent conversation sessions
- Anthropic-native provider

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

[MIT](LICENSE) © Sebby1770
