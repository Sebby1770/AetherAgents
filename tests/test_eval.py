"""Offline eval harness."""

from aetheragents import Agent, MockProvider, tool
from aetheragents.eval import EvalCase, load_cases, run_cases


def test_run_cases_pass_and_fail():
    agent = Agent("demo", MockProvider(["hello world", "goodbye"]))
    report = run_cases(
        agent,
        [
            {"prompt": "hi", "expect_contains": "hello"},
            {"prompt": "bye", "expect_contains": "hello"},  # will fail
        ],
    )
    assert report.total == 2
    assert report.passed == 1
    assert report.failed == 1
    assert report.ok is False
    assert report.cases[0].passed is True
    assert report.cases[1].passed is False
    assert "hello" in report.cases[1].missing


def test_run_cases_multiple_expects():
    agent = Agent("demo", MockProvider(["alpha beta gamma"]))
    report = run_cases(
        agent,
        [EvalCase(prompt="x", expect_contains=["alpha", "gamma"], name="multi")],
    )
    assert report.passed == 1
    assert report.cases[0].name == "multi"


def test_run_cases_no_expectation_counts_as_pass():
    agent = Agent("demo", MockProvider(["anything"]))
    report = run_cases(agent, [{"prompt": "x"}])
    assert report.passed == 1
    assert report.ok is True


def test_run_cases_stop_on_fail():
    agent = Agent("demo", MockProvider(["nope", "never-reached"]))
    report = run_cases(
        agent,
        [
            {"prompt": "a", "expect_contains": "yes"},
            {"prompt": "b", "expect_contains": "yes"},
        ],
        stop_on_fail=True,
    )
    assert report.total == 1
    assert report.failed == 1


@tool()
def add(a: int, b: int) -> int:
    "Add two numbers."
    return a + b


def test_expect_not_contains_pass_and_fail():
    agent = Agent("demo", MockProvider(["hello world", "hello world"]))
    report = run_cases(
        agent,
        [
            {"prompt": "ok", "expect_not_contains": "error"},
            {"prompt": "bad", "expect_not_contains": ["world"]},
        ],
    )
    assert report.cases[0].passed is True
    assert report.cases[1].passed is False
    assert "world" in report.cases[1].unexpected


def test_expect_tool_in_trace():
    provider = MockProvider([[("add", {"a": 1, "b": 2})], "sum is 3"])
    agent = Agent("calc", provider, tools=[add])
    report = run_cases(
        agent,
        [
            {
                "prompt": "1+2",
                "expect_contains": "3",
                "expect_tool": "add",
                "expect_regex": r"sum is \d+",
                "expect_not_contains": "error",
            }
        ],
    )
    assert report.passed == 1
    assert report.ok is True


def test_expect_tool_missing():
    agent = Agent("demo", MockProvider(["no tools here"]))
    report = run_cases(agent, [{"prompt": "x", "expect_tool": "add"}])
    assert report.failed == 1
    assert "add" in report.cases[0].tools_missing


def test_expect_regex_fail():
    agent = Agent("demo", MockProvider(["hello"]))
    report = run_cases(agent, [{"prompt": "x", "expect_regex": r"^world$"}])
    assert report.failed == 1
    assert any("regex" in r for r in report.cases[0].reasons)


def test_invalid_regex_fails_case():
    agent = Agent("demo", MockProvider(["hello"]))
    report = run_cases(agent, [{"prompt": "x", "expect_regex": "("}])
    assert report.failed == 1
    assert any("invalid regex" in r for r in report.cases[0].reasons)


def test_load_cases_jsonl(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(
            [
                "# comment",
                "",
                '{"name": "a", "prompt": "hi", "expect_contains": ["hello"]}',
                '{"name": "b", "prompt": "x", "expect_not_contains": "nope"}',
            ]
        ),
        encoding="utf-8",
    )
    cases = load_cases(path)
    assert len(cases) == 2
    assert cases[0].name == "a"
    assert cases[0].expect_contains == ["hello"]
    assert cases[1].expect_not_contains == "nope"


def test_report_html_self_contained(tmp_path):
    agent = Agent("demo", MockProvider(["hello world", "oops"]))
    report = run_cases(
        agent,
        [
            {"name": "ok", "prompt": "hi", "expect_contains": "hello"},
            {"name": "bad", "prompt": "x", "expect_contains": "never"},
        ],
    )
    html = report.to_html()
    assert "<html" in html.lower()
    assert "cdn" not in html.lower()
    assert "http://" not in html
    assert "https://" not in html
    assert "hello world" in html
    assert "PASS" in html
    assert "FAIL" in html
    out = report.write_html(tmp_path / "report.html")
    assert out.exists()
    assert "AetherAgents eval report" in out.read_text(encoding="utf-8")
