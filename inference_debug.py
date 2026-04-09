"""FlakySleuth baseline inference script.

Environment variables:
  Preferred:
    HF_TOKEN / HUGGINGFACE_HUB_TOKEN (or OPENROUTER_API_KEY / API_KEY)
    API_BASE_URL (optional, defaults to https://openrouter.ai/api/v1 for router-style keys)
    MODEL_NAME (optional, defaults to qwen/qwen3.6-plus:free on OpenRouter)

  Optional fallback:
    OPENAI_API_KEY
    API_BASE_URL (defaults to https://api.openai.com/v1 when OpenAI key is used)
    MODEL_NAME (defaults to gpt-4o-mini for OpenAI)
"""

from __future__ import annotations

import json
import os
import argparse
import time
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

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
RAW_API_KEY = os.environ.get("API_KEY")
API_KEY = RAW_API_KEY or OPENROUTER_API_KEY or OPENAI_API_KEY or HF_TOKEN or ""


def _looks_like_openrouter_key(key: str | None) -> bool:
    return bool(key and key.startswith("sk-or-"))


DEFAULT_BASE_URL = (
    "https://router.huggingface.co/v1"
    if (
        HF_TOKEN
        and not RAW_API_KEY
        and not OPENROUTER_API_KEY
        and not OPENAI_API_KEY
    )
    else (
    "https://openrouter.ai/api/v1"
    if (
        (OPENROUTER_API_KEY and not RAW_API_KEY and not OPENAI_API_KEY)
        or (_looks_like_openrouter_key(RAW_API_KEY) and not OPENAI_API_KEY)
    )
    else "https://api.openai.com/v1"
    )
)
API_BASE_URL = os.environ.get("API_BASE_URL", DEFAULT_BASE_URL)

DEFAULT_MODEL = (
    "openai/gpt-oss-120b:novita"
    if API_BASE_URL.startswith("https://router.huggingface.co")
    else (
    "qwen/qwen3.6-plus:free"
    if API_BASE_URL.startswith("https://openrouter.ai")
    else "gpt-4o-mini"
    )
)
MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
# Keep a conservative default to stay under common hackathon runtime limits.
EPISODES_PER_TASK = 2
MAX_STEPS = 20
MEMORY_MAX_CHARS = 900
LLM_MAX_RETRIES = 2

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


def _to_single_line(text: str) -> str:
    return " ".join(str(text).split())


def _short_error(text: str, max_chars: int = 220) -> str:
    one_line = _to_single_line(text)
    if len(one_line) <= max_chars:
        return one_line
    hidden = len(one_line) - max_chars
    return f"{one_line[:max_chars]}...[truncated {hidden} chars]"


class _ActionParseError(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if stripped.lower().startswith("json\n"):
        stripped = stripped[5:].strip()
    return stripped


def _extract_first_json_object(text: str) -> str | None:
    start = -1
    depth = 0
    in_string = False
    escaped = False
    for idx, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue
        if ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : idx + 1]
    return None


def _parse_action_payload(raw: str) -> tuple[FlakySleuthAction, str]:
    raw_text = (raw or "").strip()
    if not raw_text:
        raise _ActionParseError("llm_empty_output", "empty response body")

    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(value: str | None) -> None:
        if value is None:
            return
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        candidates.append(cleaned)

    add_candidate(raw_text)
    stripped = _strip_code_fences(raw_text)
    add_candidate(stripped)
    add_candidate(_extract_first_json_object(stripped))
    add_candidate(_extract_first_json_object(raw_text))

    json_errors: list[str] = []
    schema_errors: list[str] = []

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            json_errors.append(str(exc))
            continue
        if not isinstance(payload, dict):
            schema_errors.append(f"top-level JSON must be an object, got {type(payload).__name__}")
            continue
        try:
            action = FlakySleuthAction.model_validate(payload)
        except Exception as exc:
            schema_errors.append(str(exc))
            continue
        return action, candidate

    if schema_errors:
        raise _ActionParseError("llm_schema_error", _short_error(schema_errors[-1], max_chars=300))
    if json_errors:
        raise _ActionParseError("llm_json_parse_error", _short_error(json_errors[-1], max_chars=300))
    raise _ActionParseError("llm_json_parse_error", "unable to extract JSON object")


def _json_repair_prompt(error_text: str, raw_output: str) -> str:
    clipped_raw = _short_error(raw_output or "(empty)", max_chars=300)
    clipped_err = _short_error(error_text, max_chars=260)
    return (
        "Your previous response was invalid.\n"
        f"Parser error: {clipped_err}\n"
        f"Previous output (truncated): {clipped_raw}\n"
        "Respond again with ONLY one valid JSON object and no extra text.\n"
        'Required schema: {"action_type": "<one valid action>", "argument": "<string>", "metadata": {}}\n'
        'Do NOT wrap in markdown fences. Do NOT add commentary.'
    )


def _chat_completion_request(messages: list[dict[str, str]]) -> Any:
    base_kwargs = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 400,
        "temperature": 0.0,
    }
    try:
        return client.chat.completions.create(
            response_format={"type": "json_object"},
            **base_kwargs,
        )
    except Exception as json_mode_exc:
        try:
            return client.chat.completions.create(**base_kwargs)
        except Exception as plain_mode_exc:
            raise RuntimeError(
                f"json_mode_error={json_mode_exc}; plain_mode_error={plain_mode_exc}"
            ) from plain_mode_exc


def _compliance_log_start(task: str, benchmark: str, model: str) -> None:
    print(f"[START] task={task} env={benchmark} model={model}", flush=True)


def _compliance_log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: str | None,
) -> None:
    error_value = _to_single_line(error) if error else "null"
    print(
        f"[STEP] step={step} action={_to_single_line(action)} "
        f"reward={reward:.2f} done={str(bool(done)).lower()} error={error_value}",
        flush=True,
    )


def _compliance_log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_value = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(bool(success)).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_value}",
        flush=True,
    )


def obs_to_prompt(obs: FlakySleuthObservation, *, memory_hint: str | None = None) -> str:
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
{obs.tool_output or "(No action taken yet)"}

Episode memory:
{memory_hint or "(No memory yet.)"}

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


def llm_action(
    messages: list[dict[str, str]],
) -> tuple[FlakySleuthAction | None, dict[str, Any]]:
    meta: dict[str, Any] = {
        "attempted": False,
        "raw_output": "",
        "error": "",
        "reason": "",
        "attempt_count": 0,
    }
    if not API_KEY:
        meta["reason"] = "no_api_key"
        return None, meta

    work_messages = list(messages)
    last_error = ""
    for attempt in range(LLM_MAX_RETRIES + 1):
        meta["attempted"] = True
        meta["attempt_count"] = attempt + 1
        try:
            response = _chat_completion_request(work_messages)
        except Exception as exc:
            last_error = f"request_failed attempt={attempt + 1}: {exc}"
            meta["error"] = _short_error(last_error, max_chars=500)
            meta["reason"] = "llm_http_error"
            if attempt < LLM_MAX_RETRIES:
                work_messages = work_messages + [
                    {"role": "user", "content": _json_repair_prompt(last_error, "")}
                ]
                continue
            return None, meta

        raw = (response.choices[0].message.content or "").strip()
        meta["raw_output"] = raw
        try:
            action, _ = _parse_action_payload(raw)
            meta["error"] = ""
            meta["reason"] = "ok"
            return action, meta
        except _ActionParseError as exc:
            last_error = f"{exc.reason}: {exc.detail}"
            meta["error"] = _short_error(last_error, max_chars=500)
            meta["reason"] = exc.reason
            if attempt < LLM_MAX_RETRIES:
                work_messages = work_messages + [
                    {"role": "user", "content": _json_repair_prompt(last_error, raw)}
                ]
                continue
            return None, meta

    meta["error"] = _short_error(last_error or "unknown llm failure", max_chars=500)
    meta["reason"] = meta["reason"] or "llm_error"
    return None, meta


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


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    mins, secs = divmod(int(round(seconds)), 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs:d}h {mins:02d}m {secs:02d}s"
    return f"{mins:02d}m {secs:02d}s"


def _build_episode_memory(
    *,
    unique_read_files: list[str],
    zero_gain_read_files: set[str],
    search_patterns: list[str],
    blocked_duplicate_reads: int,
    no_progress_streak: int,
    max_chars: int,
) -> str:
    read_tail = ", ".join(unique_read_files[-8:]) if unique_read_files else "none"
    zero_tail = ", ".join(sorted(zero_gain_read_files)[-8:]) if zero_gain_read_files else "none"
    search_tail = ", ".join(search_patterns[-6:]) if search_patterns else "none"
    loop_warning = (
        "WARNING: Possible loop detected. Stop repeating similar exploration. "
        "Switch strategy or take a terminal action."
        if no_progress_streak >= 3 or blocked_duplicate_reads >= 2
        else "Status: exploration progress appears normal."
    )
    memory = (
        f"Read files (recent): {read_tail}\n"
        f"Zero-gain read files: {zero_tail}\n"
        f"Search patterns (recent): {search_tail}\n"
        f"Blocked duplicate reads: {blocked_duplicate_reads}\n"
        f"No-progress streak: {no_progress_streak}\n"
        f"{loop_warning}\n"
        "Guidance: Avoid rereading zero-gain files unless there is new evidence. "
        "Prefer targeted search_code or terminal action when confidence is enough."
    )
    return _clip_text(memory, max_chars=max_chars)


def _duplicate_read_replacement_pattern(obs: FlakySleuthObservation) -> str:
    test_hint = obs.test_name.split("::")[-1] if obs.test_name else "test"
    return (
        f"{test_hint}|random|sleep|time|timeout|retry|asyncio|thread|"
        "fixture|global|shared|mock|patch"
    )


def _messages_char_count(messages: list[dict[str, str]]) -> int:
    # Lightweight size heuristic to avoid unbounded context growth.
    return sum(len(str(msg.get("content", ""))) + 32 for msg in messages)


def _prune_messages_window(
    messages: list[dict[str, str]],
    *,
    step_number: int,
    prune_start_step: int,
    window_turns: int,
    max_chars: int,
) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    if len(messages) <= 2:
        return messages, None

    current_chars = _messages_char_count(messages)
    exceeds_step_threshold = step_number >= prune_start_step
    exceeds_char_budget = current_chars > max_chars
    if not exceeds_step_threshold and not exceeds_char_budget:
        return messages, None

    base = messages[:2]  # system + initial prompt
    tail = messages[2:]
    keep_tail_items = max(2, window_turns * 2)
    if len(tail) > keep_tail_items:
        tail = tail[-keep_tail_items:]
    pruned = base + tail

    reason = "step_threshold" if exceeds_step_threshold else "char_budget"
    return pruned, {
        "reason": reason,
        "before_messages": len(messages),
        "after_messages": len(pruned),
        "before_chars": current_chars,
        "after_chars": _messages_char_count(pruned),
        "step": step_number,
    }


def run_episode(
    env: FlakySleuthEnv,
    *,
    print_terminal: bool = True,
    trace_agent: bool = False,
    trace_prompts: bool = False,
    trace_max_chars: int = 2000,
    episode_label: str = "",
    compliance_stdout: bool = False,
    benchmark_name: str = "flakysleuth",
    compliance_task_name: str | None = None,
    history_prune_start_step: int = 12,
    history_window_turns: int = 4,
    history_max_chars: int = 50000,
) -> tuple[float, dict[str, Any]]:
    rewards: list[float] = []
    steps_taken = 0
    success = False
    episode_task_name = (compliance_task_name or episode_label.split(" ", 1)[0].strip() or "unknown")
    exploration_reward_total = 0.0
    final_episode_score = 0.0
    terminal_meta: dict[str, Any] = {}
    llm_steps = 0
    heuristic_steps = 0
    fallback_reasons: dict[str, int] = {}
    prune_events = 0
    read_attempt_counts: dict[str, int] = {}
    unique_read_files: list[str] = []
    zero_gain_read_files: set[str] = set()
    search_patterns: list[str] = []
    blocked_duplicate_reads = 0
    no_progress_streak = 0
    memory_hint = _build_episode_memory(
        unique_read_files=unique_read_files,
        zero_gain_read_files=zero_gain_read_files,
        search_patterns=search_patterns,
        blocked_duplicate_reads=blocked_duplicate_reads,
        no_progress_streak=no_progress_streak,
        max_chars=MEMORY_MAX_CHARS,
    )
    if compliance_stdout:
        _compliance_log_start(episode_task_name, benchmark_name, MODEL_NAME)
    try:
        obs = env.reset()

        initial_prompt = obs_to_prompt(obs, memory_hint=memory_hint)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": initial_prompt},
        ]

        if not compliance_stdout:
            _trace_print(
                trace_agent,
                (
                    f"\n[trace] {episode_label} "
                    f"task={obs.task_type} repo={obs.repo_url} test={obs.test_name}"
                ).strip(),
            )
        if trace_prompts and not compliance_stdout:
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
            messages, prune_info = _prune_messages_window(
                messages,
                step_number=step_idx + 1,
                prune_start_step=history_prune_start_step,
                window_turns=history_window_turns,
                max_chars=history_max_chars,
            )
            if prune_info:
                prune_events += 1
                if trace_agent and not compliance_stdout:
                    print(
                        "[trace] context_pruned "
                        f"reason={prune_info['reason']} "
                        f"step={prune_info['step']} "
                        f"messages={prune_info['before_messages']}->{prune_info['after_messages']} "
                        f"chars={prune_info['before_chars']}->{prune_info['after_chars']}"
                    )

            action: FlakySleuthAction
            action_source = "heuristic"
            llm_meta: dict[str, Any] = {"attempted": False, "raw_output": "", "error": ""}
            step_fallback_reason: str | None = None
            try:
                candidate, llm_meta = llm_action(messages)
                if candidate is not None:
                    action = candidate
                    action_source = "llm"
                else:
                    action = heuristic_action(obs)
                    if llm_meta.get("attempted"):
                        llm_meta["error"] = (
                            "Model response unavailable, using heuristic fallback."
                        )
            except Exception as exc:
                llm_meta["error"] = str(exc)
                action = heuristic_action(obs)

            if action.action_type == "read_file":
                prior_reads = read_attempt_counts.get(action.argument, 0)
                if prior_reads >= 1:
                    blocked_duplicate_reads += 1
                    replacement = FlakySleuthAction(
                        action_type="search_code",
                        argument=_duplicate_read_replacement_pattern(obs),
                    )
                    if trace_agent and not compliance_stdout:
                        print(
                            "[trace] action_overridden "
                            f"reason=duplicate_read file={action.argument} "
                            f"replacement={replacement.action_type}"
                        )
                    action = replacement

            if action_source == "llm":
                llm_steps += 1
            else:
                heuristic_steps += 1
                if not API_KEY:
                    reason_key = "no_api_key"
                elif llm_meta.get("reason") and llm_meta.get("reason") != "ok":
                    reason_key = str(llm_meta.get("reason"))
                elif llm_meta.get("error"):
                    reason_key = "llm_error"
                elif llm_meta.get("attempted"):
                    reason_key = "empty_or_invalid_response"
                else:
                    reason_key = "heuristic_default"
                step_fallback_reason = reason_key
                fallback_reasons[reason_key] = fallback_reasons.get(reason_key, 0) + 1

            if trace_agent and not compliance_stdout:
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
            rewards.append(reward)
            steps_taken = step_idx + 1

            if action.action_type == "read_file":
                read_attempt_counts[action.argument] = read_attempt_counts.get(action.argument, 0) + 1
                if action.argument not in unique_read_files:
                    unique_read_files.append(action.argument)
                if reward <= 0:
                    zero_gain_read_files.add(action.argument)
            elif action.action_type == "search_code":
                if action.argument not in search_patterns:
                    search_patterns.append(action.argument)

            if done:
                no_progress_streak = 0
            elif reward <= 0:
                no_progress_streak += 1
            else:
                no_progress_streak = 0

            memory_hint = _build_episode_memory(
                unique_read_files=unique_read_files,
                zero_gain_read_files=zero_gain_read_files,
                search_patterns=search_patterns,
                blocked_duplicate_reads=blocked_duplicate_reads,
                no_progress_streak=no_progress_streak,
                max_chars=MEMORY_MAX_CHARS,
            )

            step_error: str | None = None
            if isinstance(info, dict):
                raw_err = info.get("last_action_error")
                if raw_err:
                    step_error = str(raw_err)
            if not step_error and obs.tool_output and str(obs.tool_output).startswith("ERROR:"):
                step_error = str(obs.tool_output)
            if step_fallback_reason:
                fallback_detail = ""
                if llm_meta.get("error"):
                    fallback_detail = f" detail={_short_error(str(llm_meta['error']))}"
                fallback_suffix = f"llm_fallback:{step_fallback_reason}{fallback_detail}"
                if step_error:
                    step_error = f"{step_error}; {fallback_suffix}"
                else:
                    step_error = fallback_suffix

            if compliance_stdout:
                _compliance_log_step(
                    step=steps_taken,
                    action=action.model_dump_json(),
                    reward=reward,
                    done=done,
                    error=step_error,
                )

            if trace_agent and not compliance_stdout:
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
                    "llm_steps": llm_steps,
                    "heuristic_steps": heuristic_steps,
                    "fallback_reasons": dict(fallback_reasons),
                    "context_prune_events": prune_events,
                    "duplicate_read_blocks": blocked_duplicate_reads,
                }
                success = final_episode_score > 0.0
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
            next_prompt = obs_to_prompt(obs, memory_hint=memory_hint)
            messages.append({"role": "user", "content": next_prompt})
            if trace_agent and trace_prompts and not compliance_stdout:
                _trace_print(
                    True,
                    f"[trace] next user prompt (step={step_idx + 1}):",
                    text=next_prompt,
                    max_chars=trace_max_chars,
                )
    except Exception as exc:
        terminal_meta["error"] = str(exc)
        success = False
        if not compliance_stdout:
            raise
    finally:
        if compliance_stdout:
            try:
                env.close()
            except Exception:
                pass
            _compliance_log_end(
                success=success,
                steps=steps_taken,
                score=min(max(final_episode_score, 0.001), 0.999),
                rewards=rewards,
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
    parser.add_argument(
        "--compliance-stdout",
        action="store_true",
        help=(
            "Emit strict compliance logs to stdout using only [START]/[STEP]/[END] lines "
            "for each episode."
        ),
    )
    parser.add_argument(
        "--benchmark-name",
        default="flakysleuth",
        help="Benchmark name used in [START] lines when --compliance-stdout is enabled.",
    )
    parser.add_argument(
        "--history-prune-start-step",
        type=int,
        default=12,
        help="Start pruning conversation history only from this step onward.",
    )
    parser.add_argument(
        "--history-window-turns",
        type=int,
        default=4,
        help="When pruning is active, keep this many recent assistant/user turns.",
    )
    parser.add_argument(
        "--history-max-chars",
        type=int,
        default=50000,
        help="Approx max chars for messages before forced pruning by size.",
    )
    return parser.parse_args()


def main() -> None:
    run_start = time.perf_counter()
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
        raise ValueError(
            "No task types selected. Pass --task-types with at least one value."
        )
    results: dict[str, list[float]] = defaultdict(list)

    if _looks_like_placeholder_dataset(args.dataset_path) and not args.compliance_stdout:
        print(
            "[warning] dataset appears to contain fixture rows (fixture://...). "
            "Build real dataset from py-data.csv for real evaluation."
        )

    use_progress = (tqdm is not None) and (not args.no_progress) and (not args.compliance_stdout)
    if args.trace_agent and use_progress and not args.compliance_stdout:
        print(
            "[info] --trace-agent enabled, disabling progress bars for readable trace logs."
        )
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
        task_start = time.perf_counter()
        if not args.compliance_stdout:
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
                print_terminal=(not use_progress) and (not args.compliance_stdout),
                trace_agent=args.trace_agent,
                trace_prompts=args.trace_prompts,
                trace_max_chars=args.trace_max_chars,
                episode_label=f"{task_type} ep={episode + 1}/{args.episodes_per_task}",
                compliance_stdout=args.compliance_stdout,
                benchmark_name=args.benchmark_name,
                compliance_task_name=task_type,
                history_prune_start_step=args.history_prune_start_step,
                history_window_turns=args.history_window_turns,
                history_max_chars=args.history_max_chars,
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
            elif not args.compliance_stdout:
                print(f"  Episode {episode + 1}: {score:.3f}")
        if task_bar is not None:
            task_bar.close()
        task_elapsed = time.perf_counter() - task_start
        if not args.compliance_stdout:
            avg_task = sum(results[task_type]) / max(1, len(results[task_type]))
            print(
                f"  [time] task={task_type} elapsed={_format_duration(task_elapsed)} "
                f"avg_ep={task_elapsed / max(1, args.episodes_per_task):.2f}s "
                f"avg_score={avg_task:.3f}"
            )

    if overall_bar is not None:
        overall_bar.close()

    if args.compliance_stdout:
        return

    total_elapsed = time.perf_counter() - run_start
    print("\n== BASELINE RESULTS ==")
    all_scores: list[float] = []
    for task_type in task_types:
        scores = results[task_type]
        avg = sum(scores) / len(scores)
        all_scores.extend(scores)
        print(f"  {task_type:12s} avg={avg:.3f} scores={[round(s, 3) for s in scores]}")

    overall = sum(all_scores) / len(all_scores)
    print(f"  {'OVERALL':12s} avg={overall:.3f}")
    print(
        f"  {'RUNTIME':12s} total={_format_duration(total_elapsed)} "
        f"episodes={len(all_scores)} "
        f"avg_ep={(total_elapsed / max(1, len(all_scores))):.2f}s"
    )


if __name__ == "__main__":
    main()
