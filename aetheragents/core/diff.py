"""Compare two agent results without re-running them."""

from __future__ import annotations

from typing import Any

from .agent import AgentResult


def diff_results(left: AgentResult, right: AgentResult) -> dict[str, Any]:
    """Return how outputs, tool names, and estimated cost differ.

    Tool names are compared as sets of names that appeared in a tool call.
    ``cost_delta_usd`` is ``right - left``, or ``None`` when either cost is
    unknown.
    """
    left_tools = _tool_names(left)
    right_tools = _tool_names(right)
    left_cost = left.cost_usd
    right_cost = right.cost_usd
    delta = None
    if left_cost is not None and right_cost is not None:
        delta = right_cost - left_cost
    return {
        "same_output": (left.output or "") == (right.output or ""),
        "left_agent": left.agent,
        "right_agent": right.agent,
        "tools_added": [name for name in right_tools if name not in left_tools],
        "tools_removed": [name for name in left_tools if name not in right_tools],
        "cost_delta_usd": delta,
    }


def _tool_names(result: AgentResult) -> list[str]:
    names: list[str] = []
    for step in result.steps:
        if step.type == "tool_call" and step.name and step.name not in names:
            names.append(step.name)
    return names
