from __future__ import annotations

from env.models import FlakySleuthAction


def grade(action: FlakySleuthAction, task: dict) -> float:
    """Binary classification: flaky or stable. Exact match only."""
    if action.action_type != "classify_flakiness":
        return 0.001

    predicted = action.argument.strip().lower()
    if predicted not in ("flaky", "stable"):
        return 0.001

    ground_truth = str(task.get("label", "flaky")).strip().lower() or "flaky"
    return 0.999 if predicted == ground_truth else 0.001
