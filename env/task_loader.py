from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any


class TaskLoader:
    def __init__(self, csv_path: str):
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Task CSV not found: {csv_path}")

        self.tasks: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                task_types = str(row.get("task_types", "")).split(";")
                for raw_type in task_types:
                    task_type = raw_type.strip()
                    if not task_type:
                        continue
                    entry = dict(row)
                    entry["task_type"] = task_type
                    self.tasks.append(entry)

        if not self.tasks:
            raise ValueError(f"No tasks loaded from {csv_path}")

        self._forced_type: str | None = None

    def sample(self) -> dict[str, Any]:
        pool = self.tasks
        if self._forced_type:
            pool = [task for task in self.tasks if task["task_type"] == self._forced_type]
        if not pool:
            raise ValueError(f"No tasks available for task type: {self._forced_type}")

        task = random.choice(pool).copy()
        task["task_description"] = self._make_description(task)
        return task

    def force_task_type(self, task_type: str | None) -> None:
        self._forced_type = task_type

    def _make_description(self, task: dict[str, Any]) -> str:
        task_type = task["task_type"]
        if task_type == "classify":
            return (
                "Investigate the given test and determine whether it is FLAKY or STABLE. "
                "Use read_file and search_code to gather evidence. "
                "When confident, call classify_flakiness with argument 'flaky' or 'stable'."
            )
        if task_type == "root_cause":
            return (
                "This test is confirmed flaky. Identify its root cause category. "
                "Valid categories: OD, OD-Brit, OD-Vic, NIO, NOD, TD, TZD, ID, NDOI. "
                "Use read_file and search_code to find evidence. "
                "Call classify_root_cause with the category code when confident."
            )
        if task_type == "fix_proposal":
            return (
                f"This test is confirmed flaky with root cause: {task.get('category', 'unknown')}. "
                "Propose a concrete fix as a unified diff. "
                "Use read_file and search_code to understand the code. "
                "Call propose_fix with a valid unified diff string."
            )
        return "Investigate the flaky test."
