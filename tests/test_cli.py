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


def test_run_agent_file_factory(tmp_path, capsys):
    mod = tmp_path / "my_agent.py"
    mod.write_text(
        "from aetheragents import Agent, MockProvider\n"
        "\n"
        "def build_agent():\n"
        "    return Agent('fromfile', MockProvider(['loaded-from-file']))\n",
        encoding="utf-8",
    )
    code = main(["run", "--agent-file", str(mod), "--factory", "build_agent", "hello"])
    assert code == 0
    out = capsys.readouterr().out
    assert "loaded-from-file" in out


def test_run_agent_file_default_factory_quickstart(capsys):
    code = main(
        [
            "run",
            "--agent-file",
            "examples/quickstart.py",
            "What is 6 times 7?",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "42" in out


def test_run_agent_file_missing_factory(tmp_path, capsys):
    mod = tmp_path / "empty.py"
    mod.write_text("x = 1\n", encoding="utf-8")
    code = main(["run", "--agent-file", str(mod), "hi"])
    assert code == 2
    err = capsys.readouterr().err
    assert "build_agent" in err


def test_run_agent_file_not_found(capsys):
    code = main(["run", "--agent-file", "does-not-exist.py", "hi"])
    assert code == 2
    err = capsys.readouterr().err
    assert "not found" in err.lower()


def test_run_html_trace(tmp_path, capsys):
    html_path = tmp_path / "out.html"
    code = main(["run", "hello there", "--html-trace", str(html_path)])
    assert code == 0
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "hello there" in html
    assert "<html" in html.lower()
    out = capsys.readouterr().out
    assert "wrote HTML trace" in out


def test_trace_command(tmp_path, capsys):
    from aetheragents import Agent, MockProvider

    agent = Agent("a", MockProvider(["hello <script>alert(1)</script>"]))
    result = agent.run("hi")
    jpath = tmp_path / "t.json"
    result.export_trace(jpath)
    code = main(["trace", str(jpath)])
    assert code == 0
    html_path = tmp_path / "t.html"
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "hello" in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    out = capsys.readouterr().out
    assert "wrote HTML trace" in out


def test_trace_missing_file(capsys):
    code = main(["trace", "does-not-exist.json"])
    assert code == 2
    err = capsys.readouterr().err
    assert "not found" in err.lower()


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
