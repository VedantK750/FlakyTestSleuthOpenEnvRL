from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI

from env.models import FlakySleuthAction

CATEGORY_DESCRIPTIONS = {
    "TD": "Time-Dependent: fails due to wall-clock time assumptions",
    "TZD": "Timezone-Dependent: fails across timezone settings",
    "NOD": "Non-Deterministic: fails due to randomness/non-determinism",
    "NIO": "Non-Idempotent-Outcome: passes first run, fails on repeated run",
    "ID": "Implementation-Dependent: fails due to runtime implementation details",
}

EXPECTED_FIX_PATTERNS = {
    "TD": ["freeze_time", "mock", "patch", "utcnow", "datetime", "monkeypatch"],
    "TZD": ["timezone", "utc", "pytz", "zoneinfo", "tzinfo", "UTC"],
    "NOD": ["seed", "mock", "patch", "deterministic", "sorted"],
    "NIO": ["setup", "teardown", "fixture", "yield", "cleanup", "autouse"],
    "ID": ["sorted(", "list(", "frozenset", "OrderedDict"],
}


def grade(action: FlakySleuthAction, task: dict) -> float:
    """Hybrid fixer grader: pattern + dry-run apply + LLM judge."""
    if action.action_type != "propose_fix":
        return 0.0

    proposed_fix = action.argument.strip()
    if not proposed_fix:
        return 0.0

    category = str(task.get("category", "")).split(";")[0].strip().upper()
    known_fix = task.get("known_fix_diff", "") or ""
    test_code = task.get("test_code", "") or ""

    patterns = EXPECTED_FIX_PATTERNS.get(category, [])
    if patterns:
        matches = sum(1 for pattern in patterns if pattern.lower() in proposed_fix.lower())
        pattern_score = min(1.0, matches / max(1, len(patterns) * 0.4))
    else:
        pattern_score = 0.5

    apply_score = _check_diff_applies(proposed_fix, task)
    judge_score = _llm_judge(proposed_fix, known_fix, category, test_code)

    total = (0.35 * pattern_score) + (0.25 * apply_score) + (0.40 * judge_score)
    return round(min(1.0, max(0.0, total)), 4)


def _check_diff_applies(diff_text: str, task: dict) -> float:
    if "+++" not in diff_text or "---" not in diff_text:
        return 0.0

    repo_root = str(task.get("sandbox_root", "")).strip()
    if not repo_root or not Path(repo_root).exists():
        return 0.3

    patch_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as handle:
            handle.write(diff_text)
            patch_path = handle.name

        result = subprocess.run(
            ["patch", "--dry-run", "-p1", "-i", patch_path],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return 1.0 if result.returncode == 0 else 0.0
    except Exception:
        return 0.3
    finally:
        if patch_path and os.path.exists(patch_path):
            os.unlink(patch_path)


def _llm_judge(proposed: str, known: str, category: str, test_code: str) -> float:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    openai_key = os.environ.get("OPENAI_API_KEY")
    api_key = (os.environ.get("API_KEY") or openai_key or hf_token or "").strip()
    if not api_key:
        return 0.5

    default_base_url = (
        "https://router.huggingface.co/v1"
        if (hf_token and not openai_key and not os.environ.get("API_KEY"))
        else "https://api.openai.com/v1"
    )
    api_base_url = os.environ.get("API_BASE_URL", default_base_url)
    client = OpenAI(api_key=api_key, base_url=api_base_url)
    model = os.environ.get(
        "MODEL_NAME",
        "openai/gpt-oss-120b:novita"
        if api_base_url.startswith("https://router.huggingface.co")
        else "gpt-4o-mini",
    )

    cat_desc = CATEGORY_DESCRIPTIONS.get(category, f"Flakiness category: {category}")
    if known:
        known_section = f"Known accepted fix (from merged PR):\n```\n{known[:800]}\n```"
    else:
        known_section = "Known fix: Not available"

    prompt = f"""You are evaluating a proposed fix for a flaky Python test.

Flakiness category: {category}
What this means: {cat_desc}

Original flaky test code:
```python
{test_code[:1000]}
```

Proposed fix (unified diff):
```
{proposed[:1000]}
```

{known_section}

Score the proposed fix from 0 to 10:
- 0-2: Fix is wrong, irrelevant, or harmful
- 3-5: Fix partially addresses the issue but misses root cause
- 6-8: Fix addresses root cause with minor issues
- 9-10: Fix is correct, minimal, and complete

Respond ONLY with JSON:
{{"score": <integer 0-10>, "reason": "<one sentence>"}}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.0,
        )
        raw = (response.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        payload = json.loads(raw)
        score = int(payload.get("score", 5))
        return max(0.0, min(10.0, score)) / 10.0
    except Exception:
        return 0.5
