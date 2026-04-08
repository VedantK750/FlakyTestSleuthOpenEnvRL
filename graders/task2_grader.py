from __future__ import annotations

import json
from pathlib import Path

from env.models import FlakySleuthAction

_SIM_PATH = Path(__file__).resolve().parent.parent / "dataset" / "category_similarity.json"
with _SIM_PATH.open("r", encoding="utf-8") as handle:
    _RAW_SIM = json.load(handle)

_CANONICAL = {
    "OD": "OD",
    "OD-BRIT": "OD-Brit",
    "OD-VIC": "OD-Vic",
    "NIO": "NIO",
    "NOD": "NOD",
    "UD": "UD",
    "TD": "TD",
    "TZD": "TZD",
    "ID": "ID",
    "NDOI": "NDOI",
    "NDOD": "NDOD",
    "OSD": "OSD",
}


VALID_CATEGORIES = set(_CANONICAL.values())


def _normalize_category(value: str) -> str:
    text = value.strip().replace("_", "-").replace(" ", "-")
    upper = text.upper()
    return _CANONICAL.get(upper, "")


def _get_similarity(predicted: str, truth: str) -> float:
    if predicted == truth:
        return 0.999
    key_a = f"{predicted},{truth}"
    key_b = f"{truth},{predicted}"
    return float(_RAW_SIM.get(key_a, _RAW_SIM.get(key_b, 0.0)))


def grade(action: FlakySleuthAction, task: dict) -> float:
    """Root cause category classification with matrix-based partial credit."""
    if action.action_type != "classify_root_cause":
        return 0.001

    predicted = _normalize_category(action.argument)
    if predicted not in VALID_CATEGORIES:
        return 0.001

    raw_truth = str(task.get("category", "")).split(";")[0]
    truth = _normalize_category(raw_truth)
    if truth not in VALID_CATEGORIES:
        return 0.001

    sim = _get_similarity(predicted, truth)
    return max(0.001, min(0.999, sim))
