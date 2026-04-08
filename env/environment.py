from __future__ import annotations

from typing import Any

from env.models import FlakySleuthAction, FlakySleuthObservation
from env.sandbox import Sandbox
from env.task_loader import TaskLoader
from graders import grade_action

FLAKY_SIGNAL_PATTERNS = [
    "sleep",
    "random",
    "time",
    "datetime",
    "thread",
    "asyncio",
    "fixture",
    "setup",
    "teardown",
    "global",
    "shared",
    "singleton",
    "os.environ",
    "socket",
    "timeout",
    "retry",
    "mock",
    "patch",
]

TERMINAL_ACTIONS = ("classify_flakiness", "classify_root_cause", "propose_fix")


class FlakySleuthEnv:
    def __init__(self, dataset_path: str = "dataset/py_tasks.csv", max_steps: int = 20):
        self.loader = TaskLoader(dataset_path)
        self.sandbox: Sandbox | None = None
        self.current_task: dict[str, Any] | None = None
        self.step_count = 0
        self.max_steps = max_steps
        self.cumulative_progress = 0.0
        self.files_read: set[str] = set()
        self.episode_actions: list[FlakySleuthAction] = []
        self.search_pattern_counts: dict[str, int] = {}
        self.search_context_counts: dict[str, int] = {}
        self.consecutive_searches = 0

    def reset(self) -> FlakySleuthObservation:
        if self.sandbox:
            self.sandbox.cleanup()

        self.current_task = self.loader.sample()
        self.current_task.setdefault("label", "flaky")

        self.sandbox = Sandbox(self.current_task)
        self.sandbox.setup()

        self.current_task["sandbox_root"] = self.sandbox.tmpdir or ""
        test_file = self.current_task.get("test_file", "")
        if test_file and self.sandbox.tmpdir:
            self.current_task["sandbox_test_path"] = f"{self.sandbox.tmpdir}/{test_file}"

        self.step_count = 0
        self.cumulative_progress = 0.0
        self.files_read = set()
        self.episode_actions = []
        self.search_pattern_counts = {}
        self.search_context_counts = {}
        self.consecutive_searches = 0

        return self._make_obs()

    def step(self, action: FlakySleuthAction):
        if not self.current_task or not self.sandbox:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

        self.step_count += 1
        self.episode_actions.append(action)

        tool_output: str | None = None
        reward = 0.0
        done = False
        info: dict[str, Any] = {}

        if action.action_type in TERMINAL_ACTIONS:
            terminal_score = grade_action(action, self.current_task)
            late_penalty = max(0, self.step_count - 15) * 0.05

            wrong_dir_penalty = 0.0
            if (
                action.action_type == "classify_flakiness"
                and action.argument.strip().lower() == "stable"
                and str(self.current_task.get("label", "flaky")).lower() == "flaky"
            ):
                wrong_dir_penalty = 0.2

            reward = min(
                1.0,
                max(
                    0.0,
                    self.cumulative_progress + terminal_score - late_penalty - wrong_dir_penalty,
                ),
            )
            done = True
            info = {
                "terminal_score": terminal_score,
                "progress_score": self.cumulative_progress,
                "late_penalty": late_penalty,
                "task_type": self.current_task.get("task_type"),
                "category": self.current_task.get("category"),
            }
        else:
            tool_output, progress = self._execute_exploration(action)
            self.cumulative_progress = min(0.30, max(0.0, self.cumulative_progress + progress))
            reward = progress

        if not done and self.step_count >= self.max_steps:
            done = True
            info = {
                "terminal_score": 0.0,
                "progress_score": self.cumulative_progress,
                "late_penalty": max(0, self.step_count - 15) * 0.05,
                "timeout": True,
                "task_type": self.current_task.get("task_type"),
                "category": self.current_task.get("category"),
            }

        obs = self._make_obs(tool_output)
        return obs, reward, done, info

    def state(self) -> dict[str, Any]:
        if not self.current_task:
            return {
                "repo_url": None,
                "test_name": None,
                "task_type": None,
                "step_count": self.step_count,
                "files_read": [],
                "cumulative_progress": self.cumulative_progress,
            }

        return {
            "repo_url": self.current_task.get("repo_url"),
            "test_name": self.current_task.get("test_name"),
            "task_type": self.current_task.get("task_type"),
            "step_count": self.step_count,
            "files_read": sorted(self.files_read),
            "cumulative_progress": self.cumulative_progress,
        }

    def close(self) -> None:
        if self.sandbox:
            self.sandbox.cleanup()
            self.sandbox = None

    def _execute_exploration(self, action: FlakySleuthAction) -> tuple[str, float]:
        assert self.current_task is not None
        assert self.sandbox is not None

        progress = 0.0
        output = ""
        if action.action_type != "search_code":
            self.consecutive_searches = 0

        if action.action_type == "read_file":
            content = self.sandbox.read_file(action.argument)
            if content is None:
                output = f"ERROR: File not found: {action.argument}"
                progress = -0.05
            elif action.argument in self.files_read:
                output = content
                progress = 0.0
            else:
                self.files_read.add(action.argument)
                output = content
                progress = self._file_relevance_reward(action.argument)

        elif action.action_type == "search_code":
            self.consecutive_searches += 1
            output = self.sandbox.grep(action.argument)
            base_progress = self._search_relevance_reward(action.argument)
            spam_penalty, warnings = self._search_spam_penalty(action.argument, output)
            progress = max(-0.25, base_progress - spam_penalty)
            if warnings:
                output = f"{output}\n\nWARNING: {' '.join(warnings)}"

        elif action.action_type == "run_test":
            output = self.sandbox.run_test(self.current_task.get("test_name", ""))
            category = str(self.current_task.get("category", "")).strip()
            if category not in ("OD", "OD-Brit", "OD-Vic"):
                progress = 0.05
        else:
            output = f"ERROR: Unsupported action_type {action.action_type}"
            progress = -0.05

        return output, progress

    def _file_relevance_reward(self, filepath: str) -> float:
        assert self.current_task is not None

        test_file = str(self.current_task.get("test_file", ""))
        if test_file and test_file in filepath:
            return 0.07
        if filepath.endswith(".py"):
            return 0.03
        return 0.01

    def _search_relevance_reward(self, pattern: str) -> float:
        pattern_lower = pattern.lower()
        if any(signal in pattern_lower for signal in FLAKY_SIGNAL_PATTERNS):
            return 0.04
        return 0.01

    def _search_spam_penalty(self, pattern: str, output: str) -> tuple[float, list[str]]:
        penalty = 0.0
        warnings: list[str] = []

        pattern_key = " ".join(pattern.lower().split())
        if pattern_key:
            pattern_count = self.search_pattern_counts.get(pattern_key, 0) + 1
            self.search_pattern_counts[pattern_key] = pattern_count
            if pattern_count > 1:
                repeat_penalty = min(0.02 * (pattern_count - 1), 0.12)
                penalty += repeat_penalty
                warnings.append(
                    f"Repeated search pattern ({pattern_count}x) penalty={repeat_penalty:.2f}."
                )

        context_hits = self._extract_search_hits(output)
        context_key = f"{pattern_key}::{','.join(context_hits)}"
        context_count = self.search_context_counts.get(context_key, 0) + 1
        self.search_context_counts[context_key] = context_count
        if context_count > 1:
            context_penalty = min(0.03 * (context_count - 1), 0.15)
            penalty += context_penalty
            warnings.append(
                f"Same search context repeated ({context_count}x) penalty={context_penalty:.2f}."
            )

        if self.consecutive_searches > 3:
            streak_penalty = min(0.02 * (self.consecutive_searches - 3), 0.20)
            penalty += streak_penalty
            warnings.append(
                f"Search-only streak={self.consecutive_searches} penalty={streak_penalty:.2f}."
            )

        return min(penalty, 0.35), warnings

    def _extract_search_hits(self, output: str) -> tuple[str, ...]:
        files: list[str] = []
        seen: set[str] = set()
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("No matches found") or line.startswith("Search "):
                continue
            filepath = line.split(":", 1)[0].strip()
            if filepath.startswith("./"):
                filepath = filepath[2:]
            if not filepath.endswith(".py"):
                continue
            if filepath not in seen:
                seen.add(filepath)
                files.append(filepath)
            if len(files) >= 5:
                break
        return tuple(files)

    def _make_obs(self, tool_output: str | None = None) -> FlakySleuthObservation:
        if not self.current_task:
            raise RuntimeError("No current task available")

        return FlakySleuthObservation(
            repo_url=str(self.current_task.get("repo_url", "")),
            test_name=str(self.current_task.get("test_name", "")),
            test_code=str(self.current_task.get("test_code", ""))[:2000],
            file_tree=self.sandbox.file_tree if self.sandbox else [],
            tool_output=tool_output,
            task_type=str(self.current_task.get("task_type", "classify")),
            task_description=str(self.current_task.get("task_description", "Investigate the flaky test.")),
            step_count=self.step_count,
            done=False,
            reward=None,
        )
