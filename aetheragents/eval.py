"""Lightweight offline evaluation harness.

Run scripted cases against an agent (typically with :class:`MockProvider`)
without network access::

    from aetheragents import Agent, MockProvider
    from aetheragents.eval import run_cases

    agent = Agent("demo", MockProvider(["hello world"]))
    report = run_cases(agent, [
        {"prompt": "hi", "expect_contains": "hello"},
    ])
    assert report.passed == 1
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from .core.agent import Agent, AgentResult


class EvalCase(BaseModel):
    """A single evaluation case."""

    prompt: str
    expect_contains: str | list[str] | None = None
    name: str | None = None


class CaseResult(BaseModel):
    """Outcome of one evaluation case."""

    name: str
    prompt: str
    passed: bool
    output: str | None = None
    expected: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    error: str | None = None
    result: AgentResult | None = None


class EvalReport(BaseModel):
    """Aggregate results from :func:`run_cases`."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    cases: list[CaseResult] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed == 0 and self.total > 0

    def __str__(self) -> str:  # pragma: no cover - convenience
        return f"EvalReport(passed={self.passed}/{self.total})"


def _as_case(raw: EvalCase | dict[str, Any]) -> EvalCase:
    if isinstance(raw, EvalCase):
        return raw
    return EvalCase.model_validate(raw)


def _expected_list(expect: str | list[str] | None) -> list[str]:
    if expect is None:
        return []
    if isinstance(expect, str):
        return [expect]
    return list(expect)


def run_cases(
    agent: Agent,
    cases: Sequence[EvalCase | dict[str, Any]],
    *,
    stop_on_fail: bool = False,
) -> EvalReport:
    """Run ``cases`` against ``agent`` and return an :class:`EvalReport`.

    Each case is a dict (or :class:`EvalCase`) with:

    * ``prompt`` (required) - text sent to the agent
    * ``expect_contains`` - substring or list of substrings that must appear
      in the agent output (case-sensitive)
    * ``name`` - optional label used in the report

    Fully offline when the agent uses :class:`~aetheragents.llm.MockProvider`.
    """
    report = EvalReport()
    for i, raw in enumerate(cases):
        case = _as_case(raw)
        name = case.name or f"case-{i + 1}"
        expected = _expected_list(case.expect_contains)
        try:
            result = agent.run(case.prompt)
            output = result.output or ""
            missing = [e for e in expected if e not in output]
            # No expectations means "ran without error" counts as pass.
            passed = not missing
            case_result = CaseResult(
                name=name,
                prompt=case.prompt,
                passed=passed,
                output=output,
                expected=expected,
                missing=missing,
                result=result,
            )
        except Exception as exc:
            case_result = CaseResult(
                name=name,
                prompt=case.prompt,
                passed=False,
                expected=expected,
                error=f"{type(exc).__name__}: {exc}",
            )

        report.cases.append(case_result)
        report.total += 1
        if case_result.passed:
            report.passed += 1
        else:
            report.failed += 1
            if stop_on_fail:
                break
    return report
