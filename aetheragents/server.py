"""Optional FastAPI server for exposing agents over HTTP.

Install with ``pip install 'aetheragents[server]'`` then::

    from aetheragents import Agent, MockProvider
    from aetheragents.server import create_app

    app = create_app({"echo": Agent("echo", MockProvider(default="hi"))})
    # uvicorn module:app

FastAPI is imported lazily so the rest of the framework has no web dependency.
"""

from __future__ import annotations

from typing import Any

from .core.agent import Agent


def create_app(agents: dict[str, Agent]) -> Any:
    """Build a FastAPI app that serves ``POST /agents/{name}/run``."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "FastAPI is not installed. Install it with: pip install 'aetheragents[server]'"
        ) from exc

    class RunRequest(BaseModel):
        prompt: str

    app = FastAPI(title="AetherAgents", version="0.2.0")

    @app.get("/agents")
    def list_agents() -> dict[str, list[str]]:
        return {"agents": list(agents)}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/agents/{name}/run")
    async def run_agent(name: str, body: RunRequest) -> dict[str, Any]:
        if name not in agents:
            raise HTTPException(status_code=404, detail=f"Unknown agent: {name}")
        result = await agents[name].arun(body.prompt)
        return {
            "agent": result.agent,
            "output": result.output,
            "steps": [s.model_dump() for s in result.steps],
            "usage": result.usage.model_dump(),
        }

    return app
