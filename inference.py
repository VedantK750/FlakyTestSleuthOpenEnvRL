"""FlakySleuth baseline inference script.

Environment variables:
  Preferred (Hackathon / Hugging Face router):
    HF_TOKEN or HUGGINGFACE_HUB_TOKEN
    API_BASE_URL (optional, defaults to https://router.huggingface.co/v1 when HF token is used)
    MODEL_NAME (optional, defaults to openai/gpt-oss-120b:novita on HF router)

  Optional fallback:
    OPENAI_API_KEY
    API_BASE_URL (defaults to https://api.openai.com/v1 when OpenAI key is used)
    MODEL_NAME (defaults to gpt-4o-mini for OpenAI)
"""

from __future__ import annotations

import json
import os
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from env.environment import FlakySleuthEnv
from env.models import FlakySleuthAction, FlakySleuthObservation

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
API_KEY = os.environ.get("API_KEY") or OPENAI_API_KEY or HF_TOKEN or ""

DEFAULT_BASE_URL = (
    "https://router.huggingface.co/v1"
    if (HF_TOKEN and not OPENAI_API_KEY and not os.environ.get("API_KEY"))
    else "https://api.openai.com/v1"
)
API_BASE_URL = os.environ.get("API_BASE_URL", DEFAULT_BASE_URL)

DEFAULT_MODEL = (
    "openai/gpt-oss-120b:novita"
    if API_BASE_URL.startswith("https://router.huggingface.co")
    else "gpt-4o-mini"
)
MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
EPISODES_PER_TASK = 5
MAX_STEPS = 20

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are a flaky test detective.

Respond ONLY with a single valid JSON object.

Exploration actions:
{"action_type": "read_file", "argument": "relative/path.py"}
{"action_type": "search_code", "argument": "pattern"}
{"action_type": "run_test", "argument": ""}

Terminal actions:
{"action_type": "classify_flakiness", "argument": "flaky"}
{"action_type": "classify_flakiness", "argument": "stable"}
{"action_type": "classify_root_cause", "argument": "OD"}
{"action_type": "classify_root_cause", "argument": "OD-Brit"}
{"action_type": "classify_root_cause", "argument": "OD-Vic"}
{"action_type": "classify_root_cause", "argument": "NIO"}
{"action_type": "classify_root_cause", "argument": "NOD"}
{"action_type": "classify_root_cause", "argument": "TD"}
{"action_type": "classify_root_cause", "argument": "TZD"}
{"action_type": "classify_root_cause", "argument": "ID"}
{"action_type": "propose_fix", "argument": "--- a/file.py\\n+++ b/file.py\\n@@ ... @@\\n-old\\n+new"}

Rules:
1. Read the test file first.
2. Search for flaky signals: random, time, sleep, shared state, env vars.
3. Run the test for non-order-dependent scenarios.
4. Call one terminal action when confident.
"""


def obs_to_prompt(obs: FlakySleuthObservation) -> str:
    tree_preview = "\n".join(obs.file_tree[:40])
    return f"""TASK: {obs.task_description}

Repository: {obs.repo_url}
Test name: {obs.test_name}
Step: {obs.step_count}/{MAX_STEPS}

Test source code:
```python
{obs.test_code}
```

Repository file tree:
{tree_preview}

Last tool output:
{obs.tool_output or '(No action taken yet)'}

Return only JSON action."""


def heuristic_action(obs: FlakySleuthObservation) -> FlakySleuthAction:
    if obs.step_count == 0 and obs.file_tree:
        return FlakySleuthAction(action_type="read_file", argument=obs.file_tree[0])

    if obs.step_count < 2:
        return FlakySleuthAction(action_type="search_code", argument="random")

    if obs.task_type == "classify":
        return FlakySleuthAction(action_type="classify_flakiness", argument="flaky")
    if obs.task_type == "root_cause":
        return FlakySleuthAction(action_type="classify_root_cause", argument="NOD")
    return FlakySleuthAction(
        action_type="propose_fix",
        argument=(
            "--- a/src/math_utils.py\n"
            "+++ b/src/math_utils.py\n"
            "@@\n"
            "-def unstable_sum(values):\n"
            "-    random.shuffle(values)\n"
            "-    return values[0] + values[1]\n"
            "+def unstable_sum(values):\n"
            "+    ordered = sorted(values)\n"
            "+    return ordered[0] + ordered[1]\n"
        ),
    )


def llm_action(messages: list[dict[str, str]]) -> tuple[FlakySleuthAction | None, dict[str, Any]]:
    meta: dict[str, Any] = {
        "attempted": False,
        "raw_output": "",
        "error": "",
    }
    if not API_KEY:
        return None, meta

    meta["attempted"] = True
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=400,
        temperature=0.0,
    )
    raw = (response.choices[0].message.content or "").strip()
    meta["raw_output"] = raw
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    payload = json.loads(cleaned)
    return FlakySleuthAction.model_validate(payload), meta


def _clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    remaining = len(text) - max_chars
    return f"{text[:max_chars]}\n...[truncated {remaining} chars]"


def _trace_print(
    enabled: bool,
    message: str,
    *,
    text: str | None = None,
    max_chars: int = 0,
) -> None:
    if not enabled:
        return
    print(message)
    if text is not None:
        print(_clip_text(text, max_chars))


def run_episode(
    env: FlakySleuthEnv,
    *,
    print_terminal: bool = True,
    trace_agent: bool = False,
    trace_prompts: bool = False,
    trace_max_chars: int = 2000,
    episode_label: str = "",
) -> tuple[float, dict[str, Any]]:
    obs = env.reset()
    exploration_reward_total = 0.0
    final_episode_score = 0.0
    terminal_meta: dict[str, Any] = {}

    initial_prompt = obs_to_prompt(obs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_prompt},
    ]

    _trace_print(
        trace_agent,
        (
            f"\n[trace] {episode_label} "
            f"task={obs.task_type} repo={obs.repo_url} test={obs.test_name}"
        ).strip(),
    )
    if trace_prompts:
        _trace_print(
            trace_agent,
            "[trace] system prompt:",
            text=SYSTEM_PROMPT,
            max_chars=trace_max_chars,
        )
        _trace_print(
            trace_agent,
            "[trace] initial user prompt:",
            text=initial_prompt,
            max_chars=trace_max_chars,
        )

    for step_idx in range(MAX_STEPS):
        action: FlakySleuthAction
        action_source = "heuristic"
        llm_meta: dict[str, Any] = {"attempted": False, "raw_output": "", "error": ""}
        try:
            candidate, llm_meta = llm_action(messages)
            if candidate is not None:
                action = candidate
                action_source = "llm"
            else:
                action = heuristic_action(obs)
                if llm_meta.get("attempted"):
                    llm_meta["error"] = "Model response unavailable, using heuristic fallback."
        except Exception as exc:
            llm_meta["error"] = str(exc)
            action = heuristic_action(obs)

        if trace_agent:
            print(f"[trace] step={step_idx + 1} action_source={action_source}")
            if llm_meta.get("attempted"):
                _trace_print(
                    True,
                    "[trace] raw model output:",
                    text=str(llm_meta.get("raw_output", "")),
                    max_chars=trace_max_chars,
                )
            if llm_meta.get("error"):
                print(f"[trace] llm_error={llm_meta['error']}")
            print(f"[trace] action={action.model_dump_json()}")

        obs, reward, done, info = env.step(action)

        if trace_agent:
            print(
                f"[trace] step_result reward={reward:.3f} done={done} "
                f"step_count={obs.step_count}"
            )
            if obs.tool_output:
                _trace_print(
                    True,
                    "[trace] tool_output:",
                    text=obs.tool_output,
                    max_chars=trace_max_chars,
                )

        if done:
            # Terminal reward already includes cumulative progress + terminal score.
            final_episode_score = reward
            terminal_meta = {
                "action_type": action.action_type,
                "terminal_score": float(info.get("terminal_score", 0) or 0),
                "progress_score": float(info.get("progress_score", 0) or 0),
                "explore_sum": exploration_reward_total,
                "episode_score": final_episode_score,
            }
            if print_terminal:
                print(
                    f"  Terminal: {action.action_type}({action.argument[:40]}) "
                    f"-> terminal={info.get('terminal_score', 0):.2f} "
                    f"progress={info.get('progress_score', 0):.2f} "
                    f"explore_sum={exploration_reward_total:.3f} "
                    f"episode_score={final_episode_score:.3f}"
                )
            break

        exploration_reward_total += reward
        messages.append({"role": "assistant", "content": action.model_dump_json()})
        next_prompt = obs_to_prompt(obs)
        messages.append({"role": "user", "content": next_prompt})
        if trace_agent and trace_prompts:
            _trace_print(
                True,
                f"[trace] next user prompt (step={step_idx + 1}):",
                text=next_prompt,
                max_chars=trace_max_chars,
            )

    return final_episode_score, terminal_meta


def _looks_like_placeholder_dataset(dataset_path: str) -> bool:
    path = Path(dataset_path)
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    return "fixture://" in text


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FlakySleuth baseline inference.")
    parser.add_argument(
        "--dataset-path",
        default="dataset/py_tasks.csv",
        help="Processed task CSV used by the environment.",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=EPISODES_PER_TASK,
        help="Episodes per task type.",
    )
    parser.add_argument(
        "--task-types",
        default="classify,root_cause,fix_proposal",
        help="Comma-separated task types to run (classify,root_cause,fix_proposal).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars and print classic per-episode logs.",
    )
    parser.add_argument(
        "--trace-agent",
        action="store_true",
        help=(
            "Print detailed agent trace: model output, chosen action/tool call, and "
            "step results for every episode."
        ),
    )
    parser.add_argument(
        "--trace-prompts",
        action="store_true",
        help="When tracing, also print full prompts sent to the model.",
    )
    parser.add_argument(
        "--trace-max-chars",
        type=int,
        default=2500,
        help="Max chars per traced text block (prompt/model output/tool output).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    env = FlakySleuthEnv(dataset_path=args.dataset_path)
    allowed_task_types = {"classify", "root_cause", "fix_proposal"}
    task_types = [t.strip() for t in args.task_types.split(",") if t.strip()]
    invalid = [t for t in task_types if t not in allowed_task_types]
    if invalid:
        raise ValueError(
            f"Invalid task type(s): {invalid}. "
            "Valid values: classify,root_cause,fix_proposal."
        )
    if not task_types:
        raise ValueError("No task types selected. Pass --task-types with at least one value.")
    results: dict[str, list[float]] = defaultdict(list)

    if _looks_like_placeholder_dataset(args.dataset_path):
        print(
            "[warning] dataset appears to contain fixture rows (fixture://...). "
            "Build real dataset from py-data.csv for real evaluation."
        )

    use_progress = (tqdm is not None) and (not args.no_progress)
    if args.trace_agent and use_progress:
        print("[info] --trace-agent enabled, disabling progress bars for readable trace logs.")
        use_progress = False
    overall_bar = None
    if use_progress:
        overall_bar = tqdm(
            total=len(task_types) * args.episodes_per_task,
            desc="All tasks",
            unit="ep",
            dynamic_ncols=True,
        )

    for task_type in task_types:
        print(f"\n-- Task type: {task_type} --")
        env.loader.force_task_type(task_type)
        task_bar = None
        if use_progress:
            task_bar = tqdm(
                total=args.episodes_per_task,
                desc=f"{task_type}",
                unit="ep",
                leave=False,
                dynamic_ncols=True,
            )
        for episode in range(args.episodes_per_task):
            score, meta = run_episode(
                env,
                print_terminal=not use_progress,
                trace_agent=args.trace_agent,
                trace_prompts=args.trace_prompts,
                trace_max_chars=args.trace_max_chars,
                episode_label=f"{task_type} ep={episode + 1}/{args.episodes_per_task}",
            )
            results[task_type].append(score)
            if use_progress and task_bar is not None:
                task_bar.update(1)
                task_avg = sum(results[task_type]) / len(results[task_type])
                task_bar.set_postfix(
                    score=f"{score:.3f}",
                    avg=f"{task_avg:.3f}",
                    term=f"{meta.get('terminal_score', 0):.2f}",
                )
                if overall_bar is not None:
                    overall_bar.update(1)
                    all_scores = [s for values in results.values() for s in values]
                    overall_avg = sum(all_scores) / len(all_scores)
                    overall_bar.set_postfix(task=task_type, avg=f"{overall_avg:.3f}")
            else:
                print(f"  Episode {episode + 1}: {score:.3f}")
        if task_bar is not None:
            task_bar.close()

    if overall_bar is not None:
        overall_bar.close()

    print("\n== BASELINE RESULTS ==")
    all_scores: list[float] = []
    for task_type in task_types:
        scores = results[task_type]
        avg = sum(scores) / len(scores)
        all_scores.extend(scores)
        print(f"  {task_type:12s} avg={avg:.3f} scores={[round(s, 3) for s in scores]}")

    overall = sum(all_scores) / len(all_scores)
    print(f"  {'OVERALL':12s} avg={overall:.3f}")


if __name__ == "__main__":
    main()
