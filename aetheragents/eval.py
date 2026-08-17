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

Load cases from JSONL with :func:`load_cases` and render a self-contained
HTML report via :meth:`EvalReport.to_html` / :meth:`EvalReport.write_html`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from html import escape
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .core.agent import Agent, AgentResult


class EvalCase(BaseModel):
    """A single evaluation case."""

    prompt: str
    expect_contains: str | list[str] | None = None
    expect_not_contains: str | list[str] | None = None
    expect_tool: str | list[str] | None = None
    expect_regex: str | None = None
    name: str | None = None


class CaseResult(BaseModel):
    """Outcome of one evaluation case."""

    name: str
    prompt: str
    passed: bool
    output: str | None = None
    expected: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    unexpected: list[str] = Field(default_factory=list)
    tools_expected: list[str] = Field(default_factory=list)
    tools_missing: list[str] = Field(default_factory=list)
    regex: str | None = None
    reasons: list[str] = Field(default_factory=list)
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

    def to_html(self) -> str:
        """Return a self-contained HTML report (no external CSS/JS/CDN)."""
        rows: list[str] = []
        for case in self.cases:
            status = "PASS" if case.passed else "FAIL"
            cls = "pass" if case.passed else "fail"
            details: list[str] = []
            if case.error:
                details.append(f"<div class='err'>error: {escape(case.error)}</div>")
            if case.missing:
                details.append(
                    "<div>missing: "
                    + escape(", ".join(case.missing))
                    + "</div>"
                )
            if case.unexpected:
                details.append(
                    "<div>unexpected: "
                    + escape(", ".join(case.unexpected))
                    + "</div>"
                )
            if case.tools_missing:
                details.append(
                    "<div>missing tools: "
                    + escape(", ".join(case.tools_missing))
                    + "</div>"
                )
            if case.regex and not case.passed and case.regex not in (case.error or ""):
                details.append(f"<div>regex: {escape(case.regex)}</div>")
            for reason in case.reasons:
                details.append(f"<div>{escape(reason)}</div>")
            output = case.output if case.output is not None else ""
            rows.append(
                "<tr class='{cls}'>"
                "<td><span class='badge {cls}'>{status}</span></td>"
                "<td>{name}</td>"
                "<td><pre>{prompt}</pre></td>"
                "<td><pre>{output}</pre>{details}</td>"
                "</tr>".format(
                    cls=cls,
                    status=status,
                    name=escape(case.name),
                    prompt=escape(case.prompt),
                    output=escape(output),
                    details="".join(details),
                )
            )
        summary_cls = "pass" if self.ok else "fail"
        body_rows = "\n".join(rows) or (
            "<tr><td colspan='4'>No cases.</td></tr>"
        )
        return (
            "<!DOCTYPE html>\n"
            "<html lang='en'>\n"
            "<head>\n"
            "<meta charset='utf-8'/>\n"
            "<title>AetherAgents eval report</title>\n"
            "<style>\n"
            "body{font-family:system-ui,sans-serif;margin:24px;color:#122;background:#f7f7f4;}\n"
            "h1{font-size:1.4rem;margin:0 0 8px;}\n"
            ".summary{font-size:1rem;margin:0 0 16px;}\n"
            ".summary.pass{color:#0a6;}\n"
            ".summary.fail{color:#b20;}\n"
            "table{border-collapse:collapse;width:100%;background:#fff;}\n"
            "th,td{border:1px solid #ddd;padding:8px 10px;vertical-align:top;text-align:left;}\n"
            "th{background:#eef1ea;}\n"
            "tr.fail{background:#fff4f2;}\n"
            "tr.pass{background:#f3faf5;}\n"
            "pre{white-space:pre-wrap;margin:0;font-family:ui-monospace,monospace;font-size:0.9em;}\n"
            ".badge{display:inline-block;padding:2px 8px;border-radius:4px;font-weight:600;color:#fff;}\n"
            ".badge.pass{background:#0a6;}\n"
            ".badge.fail{background:#c33;}\n"
            ".err{color:#b20;margin-top:6px;}\n"
            "</style>\n"
            "</head>\n"
            "<body>\n"
            "<h1>AetherAgents eval report</h1>\n"
            f"<p class='summary {summary_cls}'>"
            f"{self.passed}/{self.total} passed"
            f"{'' if self.failed == 0 else f' · {self.failed} failed'}"
            "</p>\n"
            "<table>\n"
            "<thead><tr><th>Status</th><th>Name</th><th>Prompt</th><th>Output</th></tr></thead>\n"
            "<tbody>\n"
            f"{body_rows}\n"
            "</tbody>\n"
            "</table>\n"
            "</body>\n"
            "</html>\n"
        )

    def write_html(self, path: str | Path) -> Path:
        """Write :meth:`to_html` to ``path`` and return the path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_html(), encoding="utf-8")
        return target


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


def _tools_in_result(result: AgentResult) -> set[str]:
    names: set[str] = set()
    for step in result.steps:
        if step.name and step.type in {"tool_call", "tool_result"}:
            names.add(step.name)
    return names


def load_cases(path: str | Path) -> list[EvalCase]:
    """Load evaluation cases from a JSONL file.

    Each non-empty, non-comment line is a JSON object with at least
    ``prompt``, plus optional ``name``, ``expect_contains``,
    ``expect_not_contains``, ``expect_tool`` and ``expect_regex``.
    """
    cases: list[EvalCase] = []
    text = Path(path).read_text(encoding="utf-8")
    for i, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{i}: invalid JSON: {exc}") from exc
        try:
            cases.append(EvalCase.model_validate(data))
        except Exception as exc:
            raise ValueError(f"{path}:{i}: invalid case: {exc}") from exc
    return cases


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
    * ``expect_not_contains`` - substring(s) that must *not* appear
    * ``expect_tool`` - tool name(s) that must appear in the result trace
    * ``expect_regex`` - optional regex that must match the output
    * ``name`` - optional label used in the report

    Fully offline when the agent uses :class:`~aetheragents.llm.MockProvider`.
    """
    report = EvalReport()
    for i, raw in enumerate(cases):
        case = _as_case(raw)
        name = case.name or f"case-{i + 1}"
        expected = _expected_list(case.expect_contains)
        forbidden = _expected_list(case.expect_not_contains)
        tools_expected = _expected_list(case.expect_tool)
        try:
            result = agent.run(case.prompt)
            output = result.output or ""
            missing = [e for e in expected if e not in output]
            unexpected = [f for f in forbidden if f in output]
            seen_tools = _tools_in_result(result)
            tools_missing = [t for t in tools_expected if t not in seen_tools]
            reasons: list[str] = []
            regex_ok = True
            if case.expect_regex:
                try:
                    regex_ok = re.search(case.expect_regex, output) is not None
                except re.error as exc:
                    regex_ok = False
                    reasons.append(f"invalid regex: {exc}")
                else:
                    if not regex_ok:
                        reasons.append(f"regex did not match: {case.expect_regex}")
            if missing:
                reasons.append("missing: " + ", ".join(missing))
            if unexpected:
                reasons.append("unexpected: " + ", ".join(unexpected))
            if tools_missing:
                reasons.append("missing tools: " + ", ".join(tools_missing))
            passed = not missing and not unexpected and not tools_missing and regex_ok
            case_result = CaseResult(
                name=name,
                prompt=case.prompt,
                passed=passed,
                output=output,
                expected=expected,
                missing=missing,
                unexpected=unexpected,
                tools_expected=tools_expected,
                tools_missing=tools_missing,
                regex=case.expect_regex,
                reasons=reasons,
                result=result,
            )
        except Exception as exc:
            case_result = CaseResult(
                name=name,
                prompt=case.prompt,
                passed=False,
                expected=expected,
                tools_expected=tools_expected,
                regex=case.expect_regex,
                reasons=[f"{type(exc).__name__}: {exc}"],
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
