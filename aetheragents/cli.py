"""Console entry point for the ``aetheragents`` command.

Commands::

    aetheragents version
    aetheragents run --agent demo "hello"
    aetheragents run --agent-file examples/quickstart.py --factory build_agent "prompt"
    aetheragents eval cases.jsonl --html report.html
    aetheragents trace run.json
    aetheragents doctor
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _cmd_version(_: argparse.Namespace) -> int:
    from aetheragents import __version__

    print(f"aetheragents {__version__}")
    return 0


def _load_factory_agent(path: str, factory: str) -> Any:
    """Import ``path`` and call ``factory()`` which must return an Agent."""
    from aetheragents import Agent

    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"agent file not found: {path}")
    spec = importlib.util.spec_from_file_location("aetheragents_user_agent", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load agent file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, factory):
        raise AttributeError(f"agent file {path} has no factory {factory!r}()")
    fn = getattr(module, factory)
    if not callable(fn):
        raise TypeError(f"{factory} is not callable")
    agent = fn()
    if not isinstance(agent, Agent):
        raise TypeError(f"{factory}() must return an Agent, got {type(agent).__name__}")
    return agent


def _cmd_run(args: argparse.Namespace) -> int:
    """Run a named offline demo agent, or a factory loaded from --agent-file."""
    from aetheragents import Agent, MockProvider

    prompt = args.prompt
    if not prompt:
        print("error: prompt is required", file=sys.stderr)
        return 2

    if args.agent_file:
        try:
            agent = _load_factory_agent(args.agent_file, args.factory)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    else:
        name = args.agent or "demo"
        provider = MockProvider(
            [f"[{name}] You said: {prompt}"],
            model="mock-1",
        )
        agent = Agent(
            name,
            provider,
            instructions=f"You are the '{name}' demo agent (offline MockProvider).",
        )
    result = agent.run(prompt)
    print(result.output or "")
    if args.html_trace:
        written = result.write_trace_html(args.html_trace)
        print(f"wrote HTML trace: {written}")
    if args.verbose:
        print(
            f"\n# steps={len(result.steps)} model={result.model} "
            f"tokens={result.usage.total_tokens}",
            file=sys.stderr,
        )
    return 0


def _check_extra(package: str, extra: str) -> tuple[str, bool, str]:
    """Return (label, installed, hint)."""
    try:
        found = importlib.util.find_spec(package) is not None
    except ModuleNotFoundError:
        # Namespace parents (e.g. opentelemetry.*) raise when absent.
        found = False
    hint = f"pip install 'aetheragents[{extra}]'" if not found else "ok"
    return extra, found, hint


def _cmd_doctor(_: argparse.Namespace) -> int:
    """Report which optional extras are installed."""
    from aetheragents import __version__

    print(f"aetheragents {__version__}")
    print(f"python {sys.version.split()[0]}")
    print()
    print("Optional extras:")

    checks = [
        _check_extra("litellm", "litellm"),
        _check_extra("anthropic", "anthropic"),
        _check_extra("chromadb", "chroma"),
        _check_extra("fastapi", "server"),
        _check_extra("opentelemetry.api", "telemetry"),
    ]
    # Core always present if we got here.
    print("  core (pydantic) ......... ok")
    all_ok = True
    for extra, found, hint in checks:
        status = "ok" if found else "MISSING"
        if not found:
            all_ok = False
        pad = "." * max(1, 22 - len(extra))
        print(f"  {extra} {pad} {status}" + ("" if found else f"  ({hint})"))

    print()
    if all_ok:
        print("All optional extras are installed.")
        return 0
    print("Some extras are missing (core still works offline with MockProvider).")
    return 0  # doctor is informational; never fail the shell hard


def _last_user_text(messages: Sequence[Any]) -> str:
    for msg in reversed(messages):
        role = getattr(getattr(msg, "role", None), "value", getattr(msg, "role", None))
        content = getattr(msg, "content", None)
        if role == "user" and content:
            return str(content)
    return ""


def _cmd_eval(args: argparse.Namespace) -> int:
    """Run JSONL eval cases against a MockProvider demo agent or a factory."""
    from aetheragents import Agent, MockProvider
    from aetheragents.eval import load_cases, run_cases

    try:
        cases = load_cases(args.cases)
    except FileNotFoundError:
        print(f"error: cases file not found: {args.cases}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"error: failed to load cases: {exc}", file=sys.stderr)
        return 2

    if args.agent_file:
        try:
            agent = _load_factory_agent(args.agent_file, args.factory)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    else:
        name = args.agent or "demo"

        def _reply(messages: object) -> str:
            return f"[{name}] You said: {_last_user_text(messages)}"

        provider = MockProvider(handler=_reply, model="mock-1")
        agent = Agent(
            name,
            provider,
            instructions=f"You are the '{name}' demo agent (offline MockProvider).",
        )
    report = run_cases(agent, cases)
    print(report)
    for case in report.cases:
        status = "PASS" if case.passed else "FAIL"
        print(f"  {status}  {case.name}")
        if not case.passed:
            if case.error:
                print(f"        error: {case.error}")
            for reason in case.reasons:
                print(f"        {reason}")
    if args.html:
        written = report.write_html(args.html)
        print(f"wrote HTML report: {written}")
    return 0 if report.ok else 1


def _cmd_trace(args: argparse.Namespace) -> int:
    """Render a JSON trace (from AgentResult.export_trace) as HTML."""
    from aetheragents.core.agent import render_trace_html

    path = Path(args.path)
    if not path.is_file():
        print(f"error: trace file not found: {path}", file=sys.stderr)
        return 2
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: failed to read trace: {exc}", file=sys.stderr)
        return 2
    if not isinstance(data, dict):
        print("error: trace JSON must be an object", file=sys.stderr)
        return 2
    html = render_trace_html(data)
    out = Path(args.out) if args.out else path.with_suffix(".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote HTML trace: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aetheragents",
        description="AetherAgents — lightweight multi-agent framework CLI",
    )
    sub = parser.add_subparsers(dest="command")

    p_version = sub.add_parser("version", help="Print package version")
    p_version.set_defaults(func=_cmd_version)

    p_run = sub.add_parser("run", help="Run an offline demo agent (MockProvider)")
    p_run.add_argument("prompt", nargs="?", default=None, help="Prompt text")
    p_run.add_argument(
        "--agent", "-a", default="demo", help="Agent name (default: demo)"
    )
    p_run.add_argument(
        "--agent-file",
        default=None,
        help="Python file that defines a factory returning an Agent",
    )
    p_run.add_argument(
        "--factory",
        default="build_agent",
        help="Factory function name in --agent-file (default: build_agent)",
    )
    p_run.add_argument(
        "--html-trace",
        default=None,
        help="Write a self-contained HTML trace to this path",
    )
    p_run.add_argument(
        "--verbose", "-v", action="store_true", help="Print step/usage summary"
    )
    p_run.set_defaults(func=_cmd_run)

    p_doctor = sub.add_parser("doctor", help="Check which optional extras are installed")
    p_doctor.set_defaults(func=_cmd_doctor)

    p_eval = sub.add_parser("eval", help="Run JSONL eval cases (offline MockProvider)")
    p_eval.add_argument("cases", help="Path to a JSONL file of eval cases")
    p_eval.add_argument(
        "--agent", "-a", default="demo", help="Demo agent name (default: demo)"
    )
    p_eval.add_argument(
        "--agent-file",
        default=None,
        help="Python file that defines a factory returning an Agent",
    )
    p_eval.add_argument(
        "--factory",
        default="build_agent",
        help="Factory function name in --agent-file (default: build_agent)",
    )
    p_eval.add_argument(
        "--html", default=None, help="Write a self-contained HTML report to this path"
    )
    p_eval.set_defaults(func=_cmd_eval)

    p_trace = sub.add_parser("trace", help="Convert a JSON trace to self-contained HTML")
    p_trace.add_argument("path", help="JSON file from AgentResult.export_trace")
    p_trace.add_argument(
        "--out",
        "-o",
        default=None,
        help="HTML output path (default: replace .json with .html)",
    )
    p_trace.set_defaults(func=_cmd_trace)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.command:
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
