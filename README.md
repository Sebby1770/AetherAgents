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
| 🤝 **Multi-agent** | Sequential, parallel, route, **debate**, **map-reduce**, **handoff**, **supervise** (worker/critic), and agent-as-tool delegation. |
| 🧠 **Memory & sessions** | Rolling window + long-term vector recall, JSON-persisted `Session`s, **session forks**, and **compact** to drop old turns. |
| 🛡️ **Guardrails & budgets** | Input/output hooks plus hard **cost budgets** (`max_cost_usd`) and human **tool approval**. |
| 💰 **Cost tracking** | `result.cost_usd` estimates spend per run; export full traces as JSON or **self-contained HTML**. |
| ✅ **Testable** | Deterministic mock provider, offline **eval harness**, and a full unit suite — no API keys. |
| 🔭 **Observable & resilient** | Optional OpenTelemetry, step traces, `RetryingProvider` backoff, `CircuitBreakerProvider`, Mermaid team diagrams. |
| ⌨️ **CLI** | `aetheragents run` (including `--agent-file`), `eval --html`, `trace`, `version`, and `doctor`. |
| 🔁 **Eval, supervise, breaker** | JSON-shape eval (`expect_json` / `expect_json_keys`), `Orchestrator.supervise`, thread-safe `CircuitBreakerProvider`, `RateLimitedProvider`. |

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
from aetheragents import CircuitBreakerProvider, RateLimitedProvider, RetryingProvider
provider = RetryingProvider(AnthropicProvider(), max_retries=3)   # exponential backoff
provider = CircuitBreakerProvider(provider, failure_threshold=3, reset_after=30)
provider = RateLimitedProvider(provider, min_interval_s=0.2)      # space out calls
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

### Cost tracking & budgets

Every result reports the model used and an estimated cost (or `None` for
unknown models). Cap spend per agent or per run — exceeding the budget raises
`BudgetExceeded` (unknown models count as `$0` so MockProvider stays offline-friendly):

```python
from aetheragents import Agent, BudgetExceeded, register_model_cost

agent = Agent("assistant", provider, max_cost_usd=0.05)
try:
    result = agent.run("summarise this")
    print(result.model, result.cost_usd)
except BudgetExceeded as e:
    print(f"stopped at ${e.spent:.4f} / ${e.budget}")

register_model_cost("my-local-model", 0.0, 0.0)   # $/MTok input, output
result.export_trace("traces/last-run.json")       # full step + message dump
result.write_trace_html("traces/last-run.html")   # self-contained HTML (escaped)
```

### Tool approval & parallel tools

Gate tool execution with a callable (CLI prompt, policy engine, …). Default
`None` auto-approves everything. When the model returns multiple tool calls in
one step, set `parallel_tools=True` to run them concurrently:

```python
def approve(name: str, args: dict) -> bool:
    return name != "delete_everything"

agent = Agent(
    "safe",
    provider,
    tools=[...],
    tool_approval=approve,     # False -> "User denied tool execution"
    parallel_tools=True,       # asyncio.gather independent tool calls
    tool_timeout_s=5.0,        # timed-out tools return an error ToolResult
)
```

### Multi-agent orchestration

```python
import asyncio
from aetheragents import Agent, MockProvider, Orchestrator, keyword_router

team = Orchestrator([
    Agent("researcher", MockProvider(handler=lambda m: "facts...")),
    Agent("writer", MockProvider(handler=lambda m: "# Article")),
    Agent("critic", MockProvider(handler=lambda m: "push back...")),
    Agent("judge", MockProvider(handler=lambda m: "final synthesis")),
])

# Pipeline: researcher's output feeds the writer
final = asyncio.run(team.sequential("Write about X", order=["researcher", "writer"]))[-1]

# Fan-out: run everyone on the same task
results = asyncio.run(team.parallel("Summarise X"))

# Route: pick one agent by keyword
router = keyword_router({"write": "writer"}, default="researcher")
chosen = asyncio.run(team.route("please write a post", selector=router))

# Debate: multi-round discussion + optional synthesizer
debate = asyncio.run(team.debate(
    "Should we ship Friday?",
    agents=["writer", "critic"],
    rounds=2,
    synthesizer="judge",
))
print(debate["output"])

# Map-reduce: parallel workers, then a reducer over their outputs
mr = asyncio.run(team.map_reduce(
    "Research topic X",
    worker_names=["researcher", "critic"],
    reducer_name="writer",
))
print(mr["output"])

# Handoff: researcher output feeds the writer with a handoff prefix
handed = asyncio.run(team.handoff("Write about X", from_name="researcher", to_name="writer"))
print(handed.output)

# Supervise: worker drafts, critic replies ACCEPT or REVISE: <notes>
reviewed = asyncio.run(team.supervise(
    "Write about X",
    worker="writer",
    critic="critic",
    max_rounds=2,
))
print(reviewed.accepted, reviewed.rounds, reviewed.output)

# Docs: Mermaid diagram of the team
print(team.to_mermaid())
```

**Delegation** — expose any agent as a tool so a "manager" agent can call it:

```python
manager = Agent("manager", provider, tools=[team["researcher"].as_tool()])
```

### Sessions & forks

```python
from aetheragents import Session

session = Session("support-42", path="sessions/support-42.json")
agent.run("My printer is on fire", session=session)

# Branch the conversation without mutating the parent history
branch = session.fork(name="try-reset")
agent.run("Have you tried turning it off and on?", session=branch)

# Re-run the last user turn against the current tools
agent.replay(session)                 # or session.replay_prompt()

# Drop older turns; keep system messages + the last 4 user/assistant pairs
session.compact(keep_last=4)
```

### Offline eval

```python
from aetheragents import Agent, MockProvider
from aetheragents.eval import run_cases

agent = Agent("demo", MockProvider(["hello world", "42"]))
report = run_cases(agent, [
    {"prompt": "greet", "expect_contains": "hello"},
    {"prompt": "answer", "expect_contains": "42", "expect_not_contains": "error"},
])
assert report.ok
report.write_html("eval-report.html")
```

JSONL cases (`expect_contains`, `expect_not_contains`, `expect_tool`, `expect_regex`,
`expect_json`, `expect_json_keys`) load with `load_cases("examples/eval_cases.jsonl")`.

```python
report = run_cases(agent, [
    {"prompt": "as json", "expect_json": True, "expect_json_keys": ["city", "temp"]},
])
```

### CLI

```bash
aetheragents version
aetheragents doctor                          # which extras are installed?
aetheragents run --agent demo "hello"        # offline MockProvider demo
aetheragents run --agent-file examples/quickstart.py --factory build_agent "What is 6 times 7?"
aetheragents run "hello" --html-trace traces/run.html
aetheragents eval examples/eval_cases.jsonl --html report.html
aetheragents trace traces/last-run.json      # JSON -> self-contained HTML
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
                │ sequential · parallel · route · debate · map-reduce · handoff · supervise │
                └───────────────┬──────────────┬──────────────┘
                                │              │
                        ┌───────▼──────┐ ┌─────▼────────┐
                        │    Agent     │ │    Agent     │   reasoning loop
                        └───┬──────┬───┘ └──────────────┘
                            │      │
              ┌─────────────▼─┐  ┌─▼──────────────┐  ┌───────────────────────────┐
              │ ToolRegistry  │  │ MemoryManager  │  │        LLMProvider        │
              │ (auto schema) │  │ short + vector │  │ Mock | LiteLLM | Anthropic│
              │ + built-ins   │  │ + Session      │  │ (+ Retrying/breaker/rate) │
              └───────────────┘  └────────────────┘  └───────────────────────────┘
```

## Development

```bash
pip install -e '.[dev]'
pytest          # fully offline
ruff check .    # lint
aetheragents doctor
```

Run the examples:

```bash
python examples/quickstart.py
python examples/research_team.py
python examples/streaming_and_structured.py
```

## Roadmap

- Pluggable embedding backends for `MemoryManager`
- OpenTelemetry span coverage for tools and providers
- Richer CLI (session resume, team run)

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

[MIT](LICENSE) © Sebby1770
