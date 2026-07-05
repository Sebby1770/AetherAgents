import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")  # required by starlette's TestClient

from fastapi.testclient import TestClient  # noqa: E402

from aetheragents import Agent, MockProvider  # noqa: E402
from aetheragents.server import create_app  # noqa: E402


def make_client():
    agent = Agent("echo", MockProvider(handler=lambda msgs: "streamed reply here"))
    return TestClient(create_app({"echo": agent}))


def test_health_and_listing():
    client = make_client()
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/agents").json() == {"agents": ["echo"]}


def test_run_endpoint():
    client = make_client()
    resp = client.post("/agents/echo/run", json={"prompt": "hi"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["output"] == "streamed reply here"
    assert body["model"] == "mock-1"


def test_run_unknown_agent_404():
    client = make_client()
    assert client.post("/agents/ghost/run", json={"prompt": "hi"}).status_code == 404


def test_stream_endpoint_sends_sse_events():
    client = make_client()
    with client.stream("POST", "/agents/echo/stream", json={"prompt": "hi"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        text = "".join(chunk for chunk in resp.iter_text())
    assert "event: delta" in text
    assert "event: result" in text
    assert "streamed reply here" in text
