"""AgentResult trace export helpers."""

import json

from aetheragents import Agent, MockProvider, tool


@tool()
def ping(msg: str) -> str:
    "Echo."
    return f"pong:{msg}"


def test_to_trace_dict_shape():
    provider = MockProvider([[("ping", {"msg": "hi"})], "all done"])
    agent = Agent("tracer", provider, tools=[ping])
    result = agent.run("ping me")
    trace = result.to_trace_dict()

    assert trace["agent"] == "tracer"
    assert trace["output"] == "all done"
    assert trace["model"] == "mock-1"
    assert "usage" in trace
    assert isinstance(trace["steps"], list)
    assert any(s["type"] == "tool_call" for s in trace["steps"])
    assert any(s["type"] == "tool_result" for s in trace["steps"])
    assert any(s["type"] == "final" for s in trace["steps"])
    assert isinstance(trace["messages"], list)
    assert len(trace["messages"]) >= 2


def test_export_trace_writes_json(tmp_path):
    agent = Agent("a", MockProvider(["hello world"]))
    result = agent.run("hi")
    path = tmp_path / "traces" / "run.json"
    written = result.export_trace(path)
    assert written == path
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["output"] == "hello world"
    assert data["agent"] == "a"


def test_to_trace_html_contains_steps_and_escapes_script(tmp_path):
    provider = MockProvider(
        [[("ping", {"msg": "<script>alert(1)</script>"})], "done <script>xss</script>"]
    )
    agent = Agent("tracer", provider, tools=[ping])
    result = agent.run("ping <script>me</script>")
    html = result.to_trace_html()
    assert "<html" in html.lower()
    assert "cdn" not in html.lower()
    assert "ping" in html
    assert "tool_call" in html
    assert "done" in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    written = result.write_trace_html(tmp_path / "traces" / "run.html")
    assert written.exists()
    assert "&lt;script&gt;" in written.read_text(encoding="utf-8")


def test_to_trace_dict_is_json_serialisable():
    agent = Agent("a", MockProvider(["ok"]))
    result = agent.run("x")
    # Must not raise
    raw = json.dumps(result.to_trace_dict())
    assert "ok" in raw
