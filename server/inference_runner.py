from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class InferenceJob:
    job_id: str
    status: str
    started_at: float
    command: list[str]
    config: dict[str, Any]
    logs: list[str] = field(default_factory=list)
    return_code: int | None = None
    finished_at: float | None = None
    error: str | None = None
    stop_requested: bool = False
    summaries: list[dict[str, Any]] = field(default_factory=list)


class InferenceRunner:
    """Run inference.py in the background and expose live status."""

    def __init__(self, repo_root: Path):
        self._repo_root = repo_root.resolve()
        self._lock = threading.Lock()
        self._job: InferenceJob | None = None
        self._proc: subprocess.Popen[str] | None = None

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._job and self._job.status in {"starting", "running"}:
                raise RuntimeError("An inference run is already in progress.")

        dataset_rel = str(payload.get("dataset_path", "dataset/py_tasks.csv")).strip()
        episodes = int(payload.get("episodes_per_task", 1))
        max_steps = int(payload.get("max_steps", 20))
        task_types = str(payload.get("task_types", "classify,root_cause,fix_proposal")).strip()
        benchmark_name = str(payload.get("benchmark_name", "flakysleuth")).strip()

        if not dataset_rel:
            raise ValueError("dataset_path must not be empty.")
        if episodes < 1 or episodes > 100:
            raise ValueError("episodes_per_task must be between 1 and 100.")
        if max_steps < 1 or max_steps > 100:
            raise ValueError("max_steps must be between 1 and 100.")
        if not task_types:
            raise ValueError("task_types must not be empty.")
        if not benchmark_name:
            raise ValueError("benchmark_name must not be empty.")

        dataset_path = self._resolve_dataset_path(dataset_rel)
        command = [
            sys.executable,
            "inference.py",
            "--dataset-path",
            os.path.relpath(dataset_path, self._repo_root),
            "--episodes-per-task",
            str(episodes),
            "--task-types",
            task_types,
            "--max-steps",
            str(max_steps),
            "--benchmark-name",
            benchmark_name,
        ]

        job = InferenceJob(
            job_id=uuid.uuid4().hex[:12],
            status="starting",
            started_at=time.time(),
            command=command,
            config={
                "dataset_path": os.path.relpath(dataset_path, self._repo_root),
                "episodes_per_task": episodes,
                "task_types": task_types,
                "max_steps": max_steps,
                "benchmark_name": benchmark_name,
                "api_base_url": _clean_optional_text(payload.get("api_base_url")),
                "model_name": _clean_optional_text(payload.get("model_name")),
                "api_key_provided": bool(_clean_optional_text(payload.get("api_key"))),
            },
        )
        self._append_log(job, f"[UI] Starting run {job.job_id}")
        self._append_log(job, f"[UI] Command: {' '.join(command)}")

        with self._lock:
            self._job = job

        worker = threading.Thread(
            target=self._run_job,
            args=(job, payload),
            daemon=True,
        )
        worker.start()
        return self.snapshot(tail=300)

    def stop(self) -> bool:
        with self._lock:
            job = self._job
            proc = self._proc
            if not job or not proc or job.status not in {"starting", "running"}:
                return False
            job.stop_requested = True

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=8)
        return True

    def snapshot(self, tail: int = 300) -> dict[str, Any]:
        with self._lock:
            if self._job is None:
                return {
                    "has_job": False,
                    "status": "idle",
                    "logs": [],
                }

            job = self._job
            logs_tail = job.logs[-max(20, min(tail, 2000)) :]
            return {
                "has_job": True,
                "job_id": job.job_id,
                "status": job.status,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "return_code": job.return_code,
                "error": job.error,
                "config": job.config,
                "command": job.command,
                "summaries": job.summaries,
                "logs": logs_tail,
            }

    def _run_job(self, job: InferenceJob, payload: dict[str, Any]) -> None:
        env = os.environ.copy()
        api_key = _clean_optional_text(payload.get("api_key"))
        api_base_url = _clean_optional_text(payload.get("api_base_url"))
        model_name = _clean_optional_text(payload.get("model_name"))

        if api_key:
            env["API_KEY"] = api_key
        if api_base_url:
            env["API_BASE_URL"] = api_base_url
        if model_name:
            env["MODEL_NAME"] = model_name

        with self._lock:
            job.status = "running"

        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                job.command,
                cwd=self._repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            with self._lock:
                self._proc = process

            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                self._append_log(job, line)
                summary = _parse_end_line(line)
                if summary:
                    with self._lock:
                        job.summaries.append(summary)

            return_code = process.wait()
            extra_log: str | None = None
            with self._lock:
                job.return_code = return_code
                job.finished_at = time.time()
                if job.stop_requested:
                    job.status = "stopped"
                    extra_log = "[UI] Run stopped by user request."
                elif return_code == 0:
                    job.status = "completed"
                else:
                    job.status = "failed"
                    extra_log = f"[UI] Process exited with code {return_code}."
                self._proc = None
            if extra_log:
                self._append_log(job, extra_log)
        except Exception as exc:
            extra_log = f"[UI] Runner failed: {exc}"
            with self._lock:
                job.error = str(exc)
                job.finished_at = time.time()
                job.status = "failed"
                self._proc = None
            self._append_log(job, extra_log)
        finally:
            if process and process.stdout:
                process.stdout.close()

    def _append_log(self, job: InferenceJob, line: str) -> None:
        with self._lock:
            job.logs.append(line)
            if len(job.logs) > 3000:
                del job.logs[: len(job.logs) - 3000]

    def _resolve_dataset_path(self, dataset_path: str) -> Path:
        candidate = Path(dataset_path)
        if not candidate.is_absolute():
            candidate = (self._repo_root / candidate).resolve()
        else:
            candidate = candidate.resolve()

        # Keep data access bounded to the repository.
        if os.path.commonpath([str(self._repo_root), str(candidate)]) != str(self._repo_root):
            raise ValueError("dataset_path must point to a file inside the repository.")
        if not candidate.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        if not candidate.is_file():
            raise ValueError(f"dataset_path is not a file: {dataset_path}")
        return candidate


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_end_line(line: str) -> dict[str, Any] | None:
    # Example:
    # [END] success=true steps=3 score=1.00 rewards=0.00,0.20,1.00
    if not line.startswith("[END] "):
        return None

    payload: dict[str, str] = {}
    for token in line[len("[END] ") :].split(" "):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        payload[key.strip()] = value.strip()

    if "success" not in payload or "steps" not in payload or "score" not in payload:
        return None

    rewards_raw = payload.get("rewards", "")
    rewards: list[float] = []
    for token in rewards_raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            rewards.append(float(token))
        except ValueError:
            continue

    try:
        return {
            "success": payload["success"].lower() == "true",
            "steps": int(payload["steps"]),
            "score": float(payload["score"]),
            "rewards": rewards,
        }
    except Exception:
        return None
