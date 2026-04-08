from __future__ import annotations

from env.models import FlakySleuthAction
from graders.task1_grader import grade as grade_t1
from graders.task2_grader import grade as grade_t2
from graders.task3_grader import grade as grade_t3


def grade_action(action: FlakySleuthAction, task: dict) -> float:
    task_type = task.get("task_type")
    if task_type == "classify":
        return grade_t1(action, task)
    if task_type == "root_cause":
        return grade_t2(action, task)
    if task_type == "fix_proposal":
        return grade_t3(action, task)
    return 0.001
