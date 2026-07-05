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

from pydantic import BaseModel

from .core.agent import Agent


class RunRequest(BaseModel):
    """Request body for the run and stream endpoints."""

    prompt: str


def create_app(agents: dict[str, Agent]) -> Any:
    """Build a FastAPI app serving ``POST /agents/{name}/run`` and ``.../stream``."""
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "FastAPI is not installed. Install it with: pip install 'aetheragents[server]'"
        ) from exc

    from . import __version__

    app = FastAPI(title="AetherAgents", version=__version__)

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
            "model": result.model,
            "cost_usd": result.cost_usd,
        }

    @app.post("/agents/{name}/stream")
    async def stream_agent(name: str, body: RunRequest) -> Any:
        """Stream the run as Server-Sent Events: `delta`, `step`, then `result`."""
        from fastapi.responses import StreamingResponse

        if name not in agents:
            raise HTTPException(status_code=404, detail=f"Unknown agent: {name}")

        async def gen() -> Any:
            import json

            async for event in agents[name].astream(body.prompt):
                if event.type == "delta":
                    payload = json.dumps({"text": event.delta})
                elif event.type == "step":
                    payload = json.dumps(event.step.model_dump())
                else:  # result
                    payload = json.dumps(
                        {
                            "agent": event.result.agent,
                            "output": event.result.output,
                            "usage": event.result.usage.model_dump(),
                            "model": event.result.model,
                            "cost_usd": event.result.cost_usd,
                        }
                    )
                yield f"event: {event.type}\ndata: {payload}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
