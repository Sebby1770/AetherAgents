"""CLI entry point tests (offline)."""

import sys

import pytest

from aetheragents import __version__
from aetheragents.cli import main


def test_version(capsys):
    code = main(["version"])
    assert code == 0
    out = capsys.readouterr().out
    assert __version__ in out
    assert "aetheragents" in out


def test_run_demo(capsys):
    code = main(["run", "--agent", "demo", "hello there"])
    assert code == 0
    out = capsys.readouterr().out
    assert "hello there" in out
    assert "demo" in out


def test_run_requires_prompt(capsys):
    code = main(["run"])
    assert code == 2
    err = capsys.readouterr().err
    assert "prompt" in err.lower()


def test_doctor(capsys):
    code = main(["doctor"])
    assert code == 0
    out = capsys.readouterr().out
    assert __version__ in out
    assert "core" in out
    assert "litellm" in out
    assert "anthropic" in out


def test_no_command_prints_help(capsys):
    code = main([])
    assert code == 0
    out = capsys.readouterr().out
    assert "usage" in out.lower() or "aetheragents" in out.lower()


def test_module_entrypoint():
    # Ensure the console-script target is importable / callable.
    from aetheragents.cli import build_parser

    parser = build_parser()
    assert parser.prog == "aetheragents"
