# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-07-05

Ships the remaining v0.3 roadmap: SSE streaming over HTTP, per-model cost
estimation, guardrails, and sandboxed file tools.

### Added
- **SSE streaming endpoint**: the FastAPI server now serves
  `POST /agents/{name}/stream`, emitting the agent's live event stream as
  Server-Sent Events (`delta`, `step`, `result`). The `run` endpoint response
  now also includes `model` and `cost_usd`.
- **Cost estimation** (`aetheragents.costs`): `estimate_cost(model, usage)`
  with a longest-prefix price table (current Anthropic rates; other providers
  approximate), `register_model_cost()` for overrides, and provider-prefix
  normalisation (`anthropic.`, `litellm:`, ...). `AgentResult` gained `model`
  and a `cost_usd` property; unknown models yield `None`, never zero.
- **Guardrails** (`aetheragents.core.guardrails`): input/output hooks on
  `Agent` (`input_guardrails=` / `output_guardrails=`). A guardrail is any
  `(str) -> str` callable; raise `GuardrailError` to block. Built-ins:
  `max_length`, `blocklist`, `redact`. Input guards run before the model is
  called; output guards run on the final answer before structured parsing.
- **Sandboxed file tools** (`aetheragents.tools.file_tools(root, readonly=)`):
  `read_file` / `write_file` / `list_files` confined to a root directory with
  canonical-path traversal protection (rejects `..`, absolute paths and
  symlink escapes) and size caps.
- 29 new offline tests (105 total), including real FastAPI TestClient coverage
  of the run and SSE endpoints.

### Changed
- `AnthropicProvider` default model updated to `claude-opus-4-8` (current
  Anthropic default recommendation).

### Fixed
- The FastAPI endpoints previously rejected request bodies with 422: the
  request model was defined in a closure, unresolvable under postponed
  annotation evaluation, so FastAPI treated the body as a query parameter.
  `RunRequest` now lives at module level.

## [0.3.0] - 2026-07-04

This release ships every item from the v0.2 roadmap: streaming, structured
output, built-in tools, persistent sessions and an Anthropic-native provider —
plus automatic retries.

### Added
- **Streaming** end to end:
  - `LLMProvider.stream()` yielding `StreamEvent`s (`delta` fragments followed
    by a single `done` carrying the full `LLMResponse`), with a non-streaming
    fallback in the base class so every provider supports it.
  - Native streaming in `MockProvider` (deterministic word-by-word) and
    `LiteLLMProvider` (true network streaming with chunked tool-call
    aggregation).
  - `Agent.astream()` — the agent loop as a live `AgentEvent` stream
    (`delta` / `step` / `result`). `Agent.arun()` is now built on top of it, so
    tool-calling, memory and sessions behave identically in both forms.
- **Structured output**: `agent.run(prompt, response_model=MyModel)` parses the
  final answer into a Pydantic model (available as `AgentResult.parsed`). The
  JSON schema is injected into the system prompt; schema-violating answers are
  fed back to the model up to `structured_retries` times before
  `StructuredOutputError` is raised. Includes tolerant `extract_json()`
  (code fences, JSON embedded in prose).
- **Persistent sessions** (`aetheragents.core.session.Session`): conversation
  history that spans multiple `run()` calls and, with a `path`, survives
  process restarts via human-readable JSON. Autosaves after each run by
  default; tool turns are recorded too.
- **Anthropic-native provider** (`AnthropicProvider`): talks to Claude models
  through the official SDK (optional `[anthropic]` extra) with full tool-use
  support — system-prompt lifting, `tool_use`/`tool_result` block conversion
  and usage mapping. Conversion helpers are pure functions with offline tests;
  a `client` can be injected for testing or custom transports.
- **Automatic retries** (`RetryingProvider`): wraps any provider with
  exponential backoff on transient failures; configurable `max_retries`,
  delays and retryable exception types.
- **Built-in tools** (`aetheragents.tools`): `calculator` (whitelisted-AST
  arithmetic — never `eval`), `utc_now`, and `http_get` (http/https only, size
  and time capped). `builtin_tools()` returns them ready to register.
- 40 new offline unit tests (75 total) covering streaming, structured output,
  sessions, built-in tools, the Anthropic provider and retries.
- New runnable example: `examples/streaming_and_structured.py`.

### Changed
- `AgentResult` gained a `parsed` field for structured output.
- The FastAPI server now reports the real package version.
- `Agent.run()`/`arun()` accept `session=`, `response_model=` and
  `structured_retries=` keyword arguments.
- When both a `Session` and a `MemoryManager` are configured, the session now
  owns the verbatim conversation history and memory contributes only semantic
  recall — previously the previous turns could be sent to the model twice.

## [0.2.0] - 2026-06-23

The first functional release. The repository previously contained only a
skeleton (a basic memory manager and a tool registry with no schema generation,
and no agents at all). This release turns AetherAgents into a working,
testable multi-agent framework.

### Added
- **Agents** (`aetheragents.core.agent`): a full reasoning loop over a provider,
  tools and memory, returning a structured `AgentResult` with a step trace and
  token usage. Sync `run()` and async `arun()`, configurable `max_steps`.
- **Multi-agent orchestration** (`aetheragents.core.orchestrator`):
  `Orchestrator` with `sequential`, `parallel` and `route` patterns, plus a
  `keyword_router` helper.
- **Agent-as-tool delegation** via `Agent.as_tool()`, enabling manager/worker
  agent hierarchies.
- **Provider abstraction** (`aetheragents.llm`): `LLMProvider` interface,
  `LLMResponse`/`Usage` models, a deterministic offline `MockProvider`, and a
  `LiteLLMProvider` for 100+ real models (optional dependency).
- **Message primitives** (`aetheragents.core.messages`): `Message`, `Role`,
  `ToolCall` with OpenAI/LiteLLM wire-format serialisation.
- **`@tool` decorator and JSON-Schema generation** from function signatures and
  type hints, including PEP 604 unions and optional/required detection.
- **Pluggable memory backends**: dependency-free `InMemoryVectorStore`
  (term-frequency cosine similarity) as the default, with an optional
  `ChromaVectorStore`.
- **Optional FastAPI server** (`aetheragents.server.create_app`) exposing agents
  over HTTP.
- **Optional OpenTelemetry tracing** (`aetheragents.telemetry`) — a no-op by
  default with zero overhead.
- **Exception hierarchy** (`aetheragents.core.errors`) rooted at `AetherError`.
- **Test suite**: 35 offline unit tests covering tools, memory, providers,
  agents, orchestration, config and messages.
- **Examples**: `examples/quickstart.py` and `examples/research_team.py`, both
  runnable offline.
- **CI**: GitHub Actions running ruff + pytest on Python 3.10–3.13.
- Project metadata: MIT `LICENSE`, `.gitignore`, ruff/mypy/pytest configuration,
  and a comprehensive `README.md`.

### Changed
- **Slimmed the dependency footprint**: the core now depends only on `pydantic`.
  `litellm`, `chromadb`, `fastapi`/`uvicorn` and OpenTelemetry moved to opt-in
  `[project.optional-dependencies]` extras (`litellm`, `chroma`, `server`,
  `telemetry`, `all`, `dev`).
- **`ToolRegistry`** now introspects function signatures to produce real
  parameter schemas (previously every tool advertised empty parameters), returns
  richer `ToolResult`s (`ok`/`error`), and supports a decorator form.
- **`MemoryManager`** no longer hard-imports ChromaDB at module load, supports a
  pluggable `VectorStore`, a configurable short-term window and friendly
  `remember`/`recall` aliases.
- **`config.py`** loads from environment variables without requiring
  `pydantic-settings`, exposes `Settings.from_env()` and a cached
  `get_settings()`.

### Fixed
- The package is now importable without `litellm`, `chromadb` or
  `pydantic-settings` installed (previously `import aetheragents` could fail).

[Unreleased]: https://github.com/Sebby1770/AetherAgents/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/Sebby1770/AetherAgents/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Sebby1770/AetherAgents/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Sebby1770/AetherAgents/releases/tag/v0.2.0
