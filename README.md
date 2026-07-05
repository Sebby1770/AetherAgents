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
| 🪶 **Tiny core** | The base install depends only on `pydantic`. Backends (LiteLLM, Anthropic, ChromaDB, FastAPI, OpenTelemetry) are opt-in extras. |
| 🔌 **Provider-agnostic** | One `LLMProvider` interface. `LiteLLMProvider` reaches 100+ models, `AnthropicProvider` talks to Claude natively, `MockProvider` runs offline. |
| 🌊 **Streaming** | `agent.astream()` yields live text deltas and tool events; every provider streams (native or fallback). |
| 📦 **Structured output** | `run(prompt, response_model=MyModel)` returns a validated Pydantic instance, with automatic schema-violation retries. |
| 🛠️ **Real tool schemas** | The `@tool` decorator generates JSON-Schema from your function signature and type hints — no hand-written specs. Built-in calculator / clock / HTTP tools included. |
| 🤝 **Multi-agent** | Sequential pipelines, parallel fan-out, routing, and agent-as-tool delegation. |
| 🧠 **Memory & sessions** | Rolling window + long-term vector recall, plus JSON-persisted `Session`s that survive restarts. |
| 🛡️ **Guardrails** | Input/output validation hooks (`max_length`, `blocklist`, `redact`, or any custom callable). |
| 💰 **Cost tracking** | `result.cost_usd` estimates spend per run from a per-model price table you can extend. |
| ✅ **Testable** | Deterministic mock provider + 105 unit tests means agent logic is testable without API keys or flakiness. |
| 🔭 **Observable & resilient** | Optional OpenTelemetry tracing, per-run step traces and token accounting, `RetryingProvider` backoff. |

## Install

```bash
pip install aetheragents                 # tiny core (pydantic only)
pip install 'aetheragents[litellm]'      # 100+ models via LiteLLM
pip install 'aetheragents[anthropic]'    # Claude via the official Anthropic SDK
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

Prefer Claude natively? Swap the provider:

```python
from aetheragents import AnthropicProvider     # needs: pip install 'aetheragents[anthropic]'
agent = Agent("assistant", AnthropicProvider("claude-sonnet-5"), tools=[get_weather])
```

Wrap any provider for resilience:

```python
from aetheragents import RetryingProvider
provider = RetryingProvider(AnthropicProvider(), max_retries=3)   # exponential backoff
```

### Streaming

`astream()` exposes the whole run as live events — text deltas, tool calls,
tool results, and a terminal result:

```python
async for event in agent.astream("What's the weather in Melbourne?"):
    if event.type == "delta":
        print(event.delta, end="", flush=True)      # tokens as they arrive
    elif event.type == "step" and event.step.type == "tool_call":
        print(f"\n[calling {event.step.name}...]")
    elif event.type == "result":
        final = event.result                        # full AgentResult
```

### Structured output

Ask for a Pydantic model and get a validated instance back. Invalid answers are
automatically fed back to the model for correction:

```python
from pydantic import BaseModel

class CityFacts(BaseModel):
    city: str
    country: str
    population_millions: float

result = agent.run("Tell me about Melbourne", response_model=CityFacts)
result.parsed.country        # -> "Australia", guaranteed schema-valid
```

### Sessions

Persist a conversation across runs — and across process restarts:

```python
from aetheragents import Session

session = Session("support-42", path="sessions/support-42.json")
agent.run("My printer is on fire", session=session)
agent.run("It's STILL on fire",   session=session)   # sees the first turn
# restart the process...
session = Session("support-42", path="sessions/support-42.json")  # history restored
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

Or start from the built-ins — a safe AST-based calculator (no `eval`), a UTC
clock, and a capped HTTP fetcher:

```python
from aetheragents import builtin_tools
agent = Agent("helper", provider, tools=builtin_tools())
```

Need file access? `file_tools` confines reads/writes to a sandbox directory
(traversal, absolute paths and symlink escapes are rejected):

```python
from aetheragents import file_tools
agent = Agent("writer", provider, tools=file_tools("./workspace"))
agent = Agent("reader", provider, tools=file_tools("./docs", readonly=True))
```

### Guardrails

Validate or transform what goes into and comes out of an agent. A guardrail is
any `(str) -> str` callable; raise `GuardrailError` to block the run:

```python
from aetheragents import Agent, blocklist, max_length, redact

agent = Agent(
    "support",
    provider,
    input_guardrails=[max_length(2000)],                  # reject huge prompts
    output_guardrails=[redact(["hunter2"]), blocklist(["ssn"])],
)
```

### Cost tracking

Every result reports the model used and an estimated cost (or `None` for
unknown models). Extend or override the price table at runtime:

```python
result = agent.run("summarise this")
print(result.model, result.cost_usd)

from aetheragents import register_model_cost
register_model_cost("my-local-model", 0.0, 0.0)   # $/MTok input, output
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
# uvicorn mymodule:app
#   POST /agents/echo/run     {"prompt": "..."}   -> JSON result (+ model, cost_usd)
#   POST /agents/echo/stream  {"prompt": "..."}   -> Server-Sent Events (delta/step/result)
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
              ┌─────────────▼─┐  ┌─▼──────────────┐  ┌───────────────────────────┐
              │ ToolRegistry  │  │ MemoryManager  │  │        LLMProvider        │
              │ (auto schema) │  │ short + vector │  │ Mock | LiteLLM | Anthropic│
              │ + built-ins   │  │ + Session      │  │ (+ RetryingProvider wrap) │
              └───────────────┘  └────────────────┘  └───────────────────────────┘
```

## Development

```bash
pip install -e '.[dev]'
pytest          # 105 tests, fully offline
ruff check .    # lint
```

Run the examples:

```bash
python examples/quickstart.py
python examples/research_team.py
python examples/streaming_and_structured.py
```

## Roadmap

- Parallel tool execution within a single agent step
- Conversation branching / forking on `Session`
- Pluggable embedding backends for `MemoryManager`
- OpenTelemetry span coverage for tools and providers

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

[MIT](LICENSE) © Sebby1770
