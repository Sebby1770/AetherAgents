"""Exception hierarchy for AetherAgents.

Catch :class:`AetherError` to handle anything raised by the framework.
"""

from __future__ import annotations


class AetherError(Exception):
    """Base class for all AetherAgents errors."""


class ConfigError(AetherError):
    """Raised when configuration is missing or invalid."""


class ProviderError(AetherError):
    """Raised when an LLM provider fails or is unavailable."""


class ToolError(AetherError):
    """Raised when a tool fails to execute."""


class ToolNotFoundError(ToolError):
    """Raised when an agent requests a tool that is not registered."""


class MaxStepsExceeded(AetherError):
    """Raised when an agent loop exceeds ``max_steps`` without finishing."""


class OrchestrationError(AetherError):
    """Raised when multi-agent orchestration fails."""
