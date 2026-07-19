"""Offline eval harness."""

from aetheragents import Agent, MockProvider
from aetheragents.eval import EvalCase, run_cases


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
