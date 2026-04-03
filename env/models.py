from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action, Observation
except Exception:  # pragma: no cover
    Action = BaseModel  # type: ignore[misc,assignment]
    Observation = BaseModel  # type: ignore[misc,assignment]

TaskType = Literal["classify", "root_cause", "fix_proposal"]


class FlakySleuthObservation(Observation):
    repo_url: str = Field(..., description="Repository URL or fixture reference")
    test_name: str = Field(..., description="Pytest test identifier")
    test_code: str = Field(..., description="Test source snippet")
    file_tree: list[str] = Field(default_factory=list, description="Top-level file tree")
    tool_output: str | None = Field(default=None, description="Result of the previous exploratory action")
    task_type: TaskType = Field(..., description="Current task type")
    task_description: str = Field(..., description="Instruction for the agent")
    step_count: int = Field(default=0, description="Current episode step count")


class FlakySleuthAction(Action):
    action_type: Literal[
        "read_file",
        "search_code",
        "run_test",
        "classify_flakiness",
        "classify_root_cause",
        "propose_fix",
    ] = Field(..., description="Action to execute")
    argument: str = Field(default="", description="Action argument")


class FlakySleuthReward(BaseModel):
    score: float
    breakdown: dict[str, Any]
    explanation: str
