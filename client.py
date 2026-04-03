from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from env.models import FlakySleuthAction


@dataclass
class FlakySleuthClient:
    base_url: str
    timeout_s: float = 30.0

    def reset(self) -> dict[str, Any]:
        response = requests.post(f"{self.base_url.rstrip('/')}/reset", timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()

    def step(self, action: FlakySleuthAction) -> dict[str, Any]:
        payload = {"action": action.model_dump()}
        response = requests.post(
            f"{self.base_url.rstrip('/')}/step",
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()

    def state(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url.rstrip('/')}/state", timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()
