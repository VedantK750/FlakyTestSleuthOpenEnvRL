from __future__ import annotations

from typing import Any

from fastapi import Body, FastAPI, HTTPException
from pydantic import ValidationError

from env.environment import FlakySleuthEnv
from env.models import FlakySleuthAction, FlakySleuthObservation

app = FastAPI(title="FlakySleuth Environment")
env = FlakySleuthEnv()


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
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
