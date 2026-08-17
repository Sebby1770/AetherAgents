"""AetherAgents - a lightweight, provider-agnostic multi-agent framework.

Quick start::

    from aetheragents import Agent, MockProvider, tool

    @tool()
    def add(a: int, b: int) -> int:
        "Add two numbers."
        return a + b

    agent = Agent("calc", MockProvider(["The answer is 4."]), tools=[add])
    print(agent.run("What is 2 + 2?").output)

Swap :class:`MockProvider` for :class:`LiteLLMProvider` or
:class:`AnthropicProvider` to talk to a real model.
"""

from __future__ import annotations

from .config import Settings, get_settings
from .core import (
    AetherError,
    Agent,
    AgentEvent,
    AgentResult,
    BudgetExceeded,
    CircuitOpenError,
    ConfigError,
    Guardrail,
    GuardrailError,
    HandoffResult,
    InMemoryVectorStore,
    MaxStepsExceeded,
    MemoryManager,
    Message,
    OrchestrationError,
    Orchestrator,
    ProviderError,
    Role,
    Session,
    Step,
    StructuredOutputError,
    Tool,
    ToolCall,
    ToolError,
    ToolNotFoundError,
    ToolRegistry,
    ToolResult,
    blocklist,
    extract_json,
    keyword_router,
    make_tool,
    max_length,
    redact,
    tool,
)
from .costs import estimate_cost, register_model_cost
from .eval import CaseResult, EvalCase, EvalReport, load_cases, run_cases
from .llm import (
    AnthropicProvider,
    CircuitBreakerProvider,
    LiteLLMProvider,
    LLMProvider,
    LLMResponse,
    MockProvider,
    RetryingProvider,
    StreamEvent,
    Usage,
)
from .telemetry import configure_tracing, span
from .tools import builtin_tools, file_tools

__version__ = "0.6.0"

__all__ = [
    "__version__",
    # config
    "Settings",
    "get_settings",
    # agents & orchestration
    "Agent",
    "AgentEvent",
    "AgentResult",
    "Step",
    "Orchestrator",
    "HandoffResult",
    "keyword_router",
    "Session",
    # tools
    "ToolRegistry",
    "Tool",
    "ToolResult",
    "tool",
    "make_tool",
    "builtin_tools",
    "file_tools",
    # guardrails
    "Guardrail",
    "max_length",
    "blocklist",
    "redact",
    # costs
    "estimate_cost",
    "register_model_cost",
    # eval
    "run_cases",
    "load_cases",
    "EvalCase",
    "EvalReport",
    "CaseResult",
    # memory
    "MemoryManager",
    "InMemoryVectorStore",
    # messages
    "Message",
    "Role",
    "ToolCall",
    # structured output
    "extract_json",
    # providers
    "LLMProvider",
    "LLMResponse",
    "StreamEvent",
    "Usage",
    "MockProvider",
    "LiteLLMProvider",
    "AnthropicProvider",
    "RetryingProvider",
    "CircuitBreakerProvider",
    # telemetry
    "configure_tracing",
    "span",
    # errors
    "AetherError",
    "BudgetExceeded",
    "ConfigError",
    "ProviderError",
    "CircuitOpenError",
    "StructuredOutputError",
    "GuardrailError",
    "ToolError",
    "ToolNotFoundError",
    "MaxStepsExceeded",
    "OrchestrationError",
]
