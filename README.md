---
title: FlakySleuth Environment Server
emoji: "🔍"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# FlakySleuth Environment

OpenEnv-compatible RL environment for flaky-test investigation in real Python repos.

Flaky tests are dangerous because they make CI results untrustworthy: real regressions can be ignored as "just flaky," while healthy code can fail randomly and block releases, wasting engineering time and eroding confidence in test signals. We are building this Gym-style RL environment so agents can practice flaky-test triage in realistic repositories, learn to separate true failures from nondeterministic noise, and generate faster, more reliable debugging and fix strategies at scale.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build Dataset

Input: raw IDoFT CSV (e.g. `py-data.csv`)  
Output: processed task CSV (`dataset/py_tasks.csv`)

```bash
python dataset/build_dataset.py --input py-data.csv --output dataset/py_tasks.csv
```

### `dataset/build_dataset.py` flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--input` | `str` | `idoft/py-data.csv` | Path to raw IDoFT CSV |
| `--output` | `str` | `dataset/py_tasks.csv` | Output processed task CSV |
| `--validate-only` | bool | `False` | Validate schema + print summary only (no clone/fetch) |
| `--limit` | `int` | `None` | Process first N rows only |

Notes:
- Uses live GitHub fetch at exact SHAs.
- Optional `GITHUB_TOKEN` improves PR diff fetching/rate limits.

## Run Server

```bash
python -m server.app
```

Quick check:
```bash
curl -s http://localhost:8000/health
```

## Run Inference

Recommended (HF Router/OpenRouter/OpenAI compatible):

```bash
export HF_TOKEN=your_hf_token
# optional:
# export API_BASE_URL=https://router.huggingface.co/v1
# export MODEL_NAME=openai/gpt-oss-120b:novita

python inference.py --dataset-path dataset/py_tasks.csv --episodes-per-task 2
```

### Run Inference From Space UI

When deployed, the Space homepage serves a UI at `/` (also `/web`) that starts
`inference.py` in the background and streams logs live.

UI defaults:
- `episodes_per_task=1`
- slider range up to `100`
- live ETA estimator: `selected_tasks × episodes_per_task × 180s`
- warning when ETA may exceed 20 minutes (hackathon guidance)

### `inference.py` flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--dataset-path` | `str` | `dataset/py_tasks.csv` | Processed task CSV used by env |
| `--episodes-per-task` | `int` | `2` | Episodes per selected task type |
| `--task-types` | `str` | `classify,root_cause,fix_proposal` | Comma-separated task types |
| `--max-steps` | `int` | `20` | Max steps per episode |
| `--no-progress` | flag | `False` | Disable progress bars in non-compliance mode |
| `--trace-agent` | flag | `False` | Print detailed action/model/tool trace |
| `--trace-prompts` | flag | `False` | Include full prompts in trace |
| `--trace-max-chars` | `int` | `2500` | Clip size for traced prompt/output blocks |
| `--compliance-stdout` | flag | `True` | Strict `[START]/[STEP]/[END]` logs (default on) |
| `--no-compliance-stdout` | flag | `False` | Switch to baseline summary/progress output |
| `--benchmark-name` | `str` | `flakysleuth` | Label printed in `[START]` logs |
| `--history-prune-start-step` | `int` | `12` | Start compacting history from this step |
| `--history-window-turns` | `int` | `4` | Keep this many recent assistant/user turns on prune |
| `--history-max-chars` | `int` | `50000` | Force prune when message history exceeds this size |

Detailed trace to log:
```bash
python inference.py \
  --dataset-path dataset/py_tasks.csv \
  --episodes-per-task 1 \
  --task-types classify,root_cause \
  --no-compliance-stdout \
  --trace-agent \
  --history-prune-start-step 12 \
  --history-window-turns 4 > agent_trace.log 2>&1
```

## OpenEnv CLI

```bash
openenv/bin/openenv validate --json
openenv/bin/openenv build
openenv/bin/openenv push
```
