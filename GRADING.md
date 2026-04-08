# FlakySleuth Grading: Exact Scoring Formulas

This document describes the **exact scoring logic implemented in code** for:
- Task 1: `classify` (`classify_flakiness`)
- Task 2: `root_cause` (`classify_root_cause`)
- Task 3: `fix_proposal` (`propose_fix`)

It also explains how per-step rewards are combined inside the environment.

## Source of Truth

- `env/environment.py`
- `graders/__init__.py`
- `graders/task1_grader.py`
- `graders/task2_grader.py`
- `graders/task3_grader.py`
- `dataset/category_similarity.json`

## 1) Dispatch: Which grader is used?

`graders/grade_action()` selects grader by `task["task_type"]`:
- `classify` -> Task 1 grader
- `root_cause` -> Task 2 grader
- `fix_proposal` -> Task 3 grader
- anything else -> `0.0`

## 2) Environment reward pipeline (applies to all tasks)

At each `env.step(action)`:

1. If action is terminal (`classify_flakiness`, `classify_root_cause`, `propose_fix`):
   - compute `terminal_score = grade_action(action, task)`
   - compute penalties
   - final step reward:

```text
reward = clamp(
    cumulative_progress + terminal_score - late_penalty - wrong_dir_penalty,
    0.0,
    1.0
)
```

Where:
- `late_penalty = max(0, step_count - 15) * 0.05`
- `wrong_dir_penalty = 0.2` only when:
  - action is `classify_flakiness`
  - predicted argument is `"stable"`
  - ground-truth label is `"flaky"`
- `done = True`

2. If action is non-terminal (exploration):
   - compute `progress` from exploration action
   - update cumulative progress:

```text
cumulative_progress = clamp(cumulative_progress + progress, 0.0, 0.30)
reward = progress
```

3. Timeout rule:
   - if not already done and `step_count >= max_steps`, set `done = True`
   - no additional terminal score is applied at timeout.

## 3) Exploration progress rewards (exact values)

### `read_file`
- file missing/unsafe -> `progress = -0.05`
- file already read in this episode -> `progress = 0.0`
- new file:
  - if file path contains `task["test_file"]` -> `0.07`
  - else if file ends with `.py` -> `0.03`
  - else -> `0.01`

### `search_code`
- base reward:
  - if query contains any flaky-signal tokens (`sleep`, `random`, `time`, `datetime`, `thread`, `asyncio`, `fixture`, `setup`, `teardown`, `global`, `shared`, `singleton`, `os.environ`, `socket`, `timeout`, `retry`, `mock`, `patch`) -> `0.04`
  - otherwise -> `0.01`
- spam penalties (all apply, then summed and capped):
  - repeated same normalized search pattern in episode:
    - `repeat_penalty = min(0.02 * (pattern_count - 1), 0.12)` for `pattern_count > 1`
  - repeated same search context (same normalized pattern + same extracted top `.py` hit files):
    - `context_penalty = min(0.03 * (context_count - 1), 0.15)` for `context_count > 1`
  - long search-only streak:
    - `streak_penalty = min(0.02 * (consecutive_searches - 3), 0.20)` for `consecutive_searches > 3`
  - total spam penalty cap: `min(sum_penalties, 0.35)`
- final `search_code` progress:

```text
progress = max(-0.25, base_reward - spam_penalty)
```

- environment appends `WARNING:` text to tool output when penalties fire.
- `consecutive_searches` resets on any non-`search_code` action.

### `run_test`
- if category is **not** one of `OD`, `OD-Brit`, `OD-Vic` -> `0.05`
- if category is order-dependent (`OD`, `OD-Brit`, `OD-Vic`) -> `0.0`

### unsupported action type
- `progress = -0.05`

## 4) Task 1 scorer (`classify_flakiness`)

Binary exact-match scorer:

```text
if action_type != "classify_flakiness": return 0.0
if predicted not in {"flaky","stable"}: return 0.0
truth = task["label"] (default "flaky")
terminal_score = 1.0 if predicted == truth else 0.0
```

Notes:
- In current dataset builder, rows are written with `label = "flaky"` by default.
- Predicting `"stable"` on flaky truth also triggers environment `wrong_dir_penalty = 0.2`.

## 5) Task 2 scorer (`classify_root_cause`)

Matrix-based similarity scorer.

### 5.1 Category normalization

Prediction and truth are normalized by:
- trim
- replace `_` with `-`
- replace spaces with `-`
- uppercase and map through canonical aliases:
  - `OD-BRIT` -> `OD-Brit`
  - `OD-VIC` -> `OD-Vic`
  - etc.

If normalized value is not in valid set, score is `0.0`.

Truth category is the **first** category if semicolon-separated:

```text
raw_truth = str(task["category"]).split(";")[0]
```

### 5.2 Similarity scoring

```text
if predicted == truth: return 1.0
else return similarity[predicted,truth] or similarity[truth,predicted] or 0.0
```

The similarity matrix is loaded from `dataset/category_similarity.json`.

Current non-identity similarity entries:
- `OD,OD-Brit`: `0.7`
- `OD,OD-Vic`: `0.7`
- `OD-Brit,OD-Vic`: `0.8`
- `OD,NIO`: `0.4`
- `OD,NDOI`: `0.3`
- `NOD,TD`: `0.6`
- `NOD,TZD`: `0.5`
- `NOD,NDOI`: `0.5`
- `TD,TZD`: `0.7`
- `NOD,ID`: `0.3`
- `UD,OD`: `0.2`
- `UD,NOD`: `0.2`
- `UD,NIO`: `0.2`
- `UD,TD`: `0.2`
- `UD,ID`: `0.2`

Any missing pair defaults to `0.0`.

## 6) Task 3 scorer (`propose_fix`)

Hybrid weighted scorer:

```text
if action_type != "propose_fix": return 0.0
if proposed_fix is empty: return 0.0

total = 0.35 * pattern_score + 0.25 * apply_score + 0.40 * judge_score
terminal_score = round(clamp(total, 0.0, 1.0), 4)
```

### 6.1 `pattern_score`

Category-specific keyword patterns are checked against the proposed diff.

For category with pattern list:

```text
matches = number of patterns found (case-insensitive substring)
pattern_score = min(1.0, matches / max(1, len(patterns) * 0.4))
```

If category has no pattern list:
- `pattern_score = 0.5`

Current pattern lists:
- `TD`: `freeze_time`, `mock`, `patch`, `utcnow`, `datetime`, `monkeypatch`
- `TZD`: `timezone`, `utc`, `pytz`, `zoneinfo`, `tzinfo`, `UTC`
- `NOD`: `seed`, `mock`, `patch`, `deterministic`, `sorted`
- `NIO`: `setup`, `teardown`, `fixture`, `yield`, `cleanup`, `autouse`
- `ID`: `sorted(`, `list(`, `frozenset`, `OrderedDict`

### 6.2 `apply_score` (`_check_diff_applies`)

```text
if diff does not contain both '---' and '+++': return 0.0
if sandbox_root missing or not existing: return 0.3
else run: patch --dry-run -p1 -i <temp_patch>
  return 1.0 if patch exit code == 0
  return 0.0 otherwise
on exception: return 0.3
```

### 6.3 `judge_score` (`_llm_judge`)

LLM judge behavior:
- If no API key available -> `judge_score = 0.5`
- Else sends a judge prompt asking for JSON `{"score": 0..10, "reason": ...}`
- Parses integer score, clamps to `[0,10]`, then scales to `[0,1]`:

```text
judge_score = clamp(int_score, 0, 10) / 10
```

- On any judge exception / parse failure -> `judge_score = 0.5`

API/model resolution in judge:
- API key preference: `API_KEY` -> `OPENROUTER_API_KEY` -> `OPENAI_API_KEY`
- Base URL:
  - OpenRouter inferred -> `https://openrouter.ai/api/v1`
  - else -> `https://api.openai.com/v1`
- Model default:
  - OpenRouter base URL -> `qwen/qwen3.6-plus:free`
  - else -> `gpt-4o-mini`

## 7) Worked examples

### Example A: Task 1 correct classify early

- `cumulative_progress = 0.05`
- `terminal_score = 1.0`
- `late_penalty = 0.0`
- `wrong_dir_penalty = 0.0`

```text
reward = clamp(0.05 + 1.0 - 0 - 0, 0, 1) = 1.0
```

### Example B: Task 2 wrong category but some exploration

- `cumulative_progress = 0.05`
- `terminal_score = 0.0` (no similarity match)
- penalties = `0`

```text
reward = clamp(0.05 + 0.0, 0, 1) = 0.05
```

### Example C: Task 3 with weak fix and no API key

- `judge_score = 0.5` fallback
- `apply_score` and `pattern_score` depend on diff contents
- final weighted sum then clamped and rounded to 4 decimals.

## 8) Important implementation notes

- `cumulative_progress` is capped at `0.30` and never below `0.0`.
- Terminal reward can be reduced by late penalty after step 15.
- Timeout does not invoke grader; it only ends the episode.
- Dataset construction choices (especially `label` and category quality) heavily influence observed score behavior.

## 9) Inference-side controls (not grader formulas)

`inference.py` now includes policy/runtime controls that do not change grader math directly but change agent behavior:

- episode memory injected into every prompt (recent files, search patterns, no-progress streak)
- explicit loop warning prompt when no-progress/duplicate patterns are detected
- duplicate `read_file` attempts are overridden to targeted `search_code`
- conversation compaction controls:
  - `--history-prune-start-step` (default `12`)
  - `--history-window-turns` (default `4`)
  - `--history-max-chars` (default `50000`)
- detailed tracing options (`--trace-agent`, `--trace-prompts`) for audit/debug
