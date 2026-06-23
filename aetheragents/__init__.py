"""AetherAgents - a lightweight, provider-agnostic multi-agent framework.

Quick start::

    from aetheragents import Agent, MockProvider, tool

    @tool()
    def add(a: int, b: int) -> int:
        "Add two numbers."
        return a + b

    agent = Agent("calc", MockProvider(["The answer is 4."]), tools=[add])
    print(agent.run("What is 2 + 2?").output)

Swap :class:`MockProvider` for :class:`LiteLLMProvider` to talk to a real model.
"""

from __future__ import annotations

from .config import Settings, get_settings
from .core import (
    AetherError,
    Agent,
    AgentResult,
    ConfigError,
    InMemoryVectorStore,
    MaxStepsExceeded,
    MemoryManager,
    Message,
    OrchestrationError,
    Orchestrator,
    ProviderError,
    Role,
    Step,
    Tool,
    ToolCall,
    ToolError,
    ToolNotFoundError,
    ToolRegistry,
    ToolResult,
    keyword_router,
    make_tool,
    tool,
)
from .llm import LiteLLMProvider, LLMProvider, LLMResponse, MockProvider, Usage
from .telemetry import configure_tracing, span

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # config
    "Settings",
    "get_settings",
    # agents & orchestration
    "Agent",
    "AgentResult",
    "Step",
    "Orchestrator",
    "keyword_router",
    # tools
    "ToolRegistry",
    "Tool",
    "ToolResult",
    "tool",
    "make_tool",
    # memory
    "MemoryManager",
    "InMemoryVectorStore",
    # messages
    "Message",
    "Role",
    "ToolCall",
    # providers
    "LLMProvider",
    "LLMResponse",
    "Usage",
    "MockProvider",
    "LiteLLMProvider",
    # telemetry
    "configure_tracing",
    "span",
    # errors
    "AetherError",
    "ConfigError",
    "ProviderError",
    "ToolError",
    "ToolNotFoundError",
    "MaxStepsExceeded",
    "OrchestrationError",
]
