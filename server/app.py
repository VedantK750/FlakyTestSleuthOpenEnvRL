from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ValidationError

from env.environment import FlakySleuthEnv
from env.models import FlakySleuthAction, FlakySleuthObservation
from server.inference_runner import InferenceRunner
from server.ui import render_home_page

app = FastAPI(title="FlakySleuth Environment")
env = FlakySleuthEnv()
inference_runner = InferenceRunner(Path(__file__).resolve().parent.parent)


class FlakySleuthState(BaseModel):
    repo_url: str | None = None
    test_name: str | None = None
    task_type: str | None = None
    step_count: int
    files_read: list[str]
    cumulative_progress: float


class InferenceRunRequest(BaseModel):
    dataset_path: str = Field(default="dataset/py_tasks.csv")
    episodes_per_task: int = Field(default=1, ge=1, le=100)
    task_types: str = Field(default="classify,root_cause,fix_proposal")
    max_steps: int = Field(default=20, ge=1, le=100)
    benchmark_name: str = Field(default="flakysleuth")
    api_base_url: str | None = None
    model_name: str | None = None
    api_key: str | None = None


@app.post("/reset")
def reset() -> dict[str, Any]:
    observation = env.reset()
    return {
        "observation": observation.model_dump(),
        "reward": None,
        "done": False,
    }


@app.post("/step")
def step(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Accept either {'action': {...}} or direct action payload."""
    try:
        action_payload = payload.get("action", payload)
        action = FlakySleuthAction.model_validate(action_payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    try:
        observation, reward, done, info = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "observation": observation.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict[str, Any]:
    return env.state()


@app.get("/schema")
def schema() -> dict[str, Any]:
    return {
        "action": FlakySleuthAction.model_json_schema(),
        "observation": FlakySleuthObservation.model_json_schema(),
        "state": FlakySleuthState.model_json_schema(),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/", include_in_schema=False)
def root() -> HTMLResponse:
    return HTMLResponse(render_home_page())


@app.get("/web", include_in_schema=False)
def web() -> HTMLResponse:
    return HTMLResponse(render_home_page())


@app.post("/web/inference/start", include_in_schema=False)
def start_inference(payload: InferenceRunRequest) -> dict[str, Any]:
    request_payload = payload.model_dump()
    try:
        return inference_runner.start(request_payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/web/inference/status", include_in_schema=False)
def inference_status(tail: int = Query(default=450, ge=20, le=2000)) -> dict[str, Any]:
    return inference_runner.snapshot(tail=tail)


@app.post("/web/inference/stop", include_in_schema=False)
def stop_inference() -> dict[str, Any]:
    stopped = inference_runner.stop()
    snapshot = inference_runner.snapshot(tail=450)
    snapshot["stopped"] = stopped
    return snapshot


@app.get("/metadata")
def metadata() -> dict[str, str]:
    return {
        "name": "FlakySleuth Environment",
        "description": (
            "RL environment for flaky-test investigation in Python repositories."
        ),
    }


@app.post("/mcp")
def mcp(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    request_id = payload.get("id")
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"status": "ok"},
    }


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
