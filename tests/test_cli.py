"""CLI entry point tests (offline)."""

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


def test_eval_passing_cases(tmp_path, capsys):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"name": "echo", "prompt": "hello world", "expect_contains": ["hello"]}\n'
        '{"name": "no-err", "prompt": "hello world", "expect_not_contains": "Traceback"}\n',
        encoding="utf-8",
    )
    code = main(["eval", str(path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "echo" in out


def test_eval_fails_exit_1(tmp_path, capsys):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"name": "miss", "prompt": "hello", "expect_contains": ["never-seen"]}\n',
        encoding="utf-8",
    )
    code = main(["eval", str(path)])
    assert code == 1
    out = capsys.readouterr().out
    assert "FAIL" in out


def test_eval_missing_file(capsys):
    code = main(["eval", "does-not-exist.jsonl"])
    assert code == 2
    err = capsys.readouterr().err
    assert "not found" in err.lower()


def test_eval_example_cases(capsys):
    code = main(["eval", "examples/eval_cases.jsonl"])
    assert code == 0
    out = capsys.readouterr().out
    assert "PASS" in out


def test_eval_writes_html(tmp_path, capsys):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        '{"name": "echo", "prompt": "hello", "expect_contains": "hello"}\n',
        encoding="utf-8",
    )
    html_path = tmp_path / "out.html"
    code = main(["eval", str(cases), "--html", str(html_path)])
    assert code == 0
    assert html_path.exists()
    assert "<html" in html_path.read_text(encoding="utf-8").lower()
