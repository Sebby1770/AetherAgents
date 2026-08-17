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


class CircuitOpenError(ProviderError):
    """Raised when a :class:`~aetheragents.llm.CircuitBreakerProvider` is open.

    Attributes:
        provider: Name of the wrapped inner provider, if known.
        reset_after: Configured cooldown in seconds, if known.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        reset_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.reset_after = reset_after


class ToolError(AetherError):
    """Raised when a tool fails to execute."""


class ToolNotFoundError(ToolError):
    """Raised when an agent requests a tool that is not registered."""


class MaxStepsExceeded(AetherError):
    """Raised when an agent loop exceeds ``max_steps`` without finishing."""


class StructuredOutputError(AetherError):
    """Raised when a model's output cannot be parsed into the requested model."""


class GuardrailError(AetherError):
    """Raised when a guardrail blocks an agent's input or output."""


class OrchestrationError(AetherError):
    """Raised when multi-agent orchestration fails."""


class BudgetExceeded(AetherError):
    """Raised when an agent's estimated cost exceeds its ``max_cost_usd`` budget.

    Attributes:
        spent: Estimated USD spent so far (0.0 when cost is unknown).
        budget: The configured budget cap in USD.
        agent: Name of the agent that hit the cap, if known.
    """

    def __init__(
        self,
        message: str,
        *,
        spent: float = 0.0,
        budget: float = 0.0,
        agent: str | None = None,
    ) -> None:
        super().__init__(message)
        self.spent = spent
        self.budget = budget
        self.agent = agent
