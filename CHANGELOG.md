# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/Sebby1770/AetherAgents/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Sebby1770/AetherAgents/releases/tag/v0.2.0
