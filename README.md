# FlakySleuth Environment

OpenEnv-compatible RL environment for flaky-test investigation in real Python repos.

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

Recommended (HF router):

```bash
export HF_TOKEN=your_hf_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=openai/gpt-oss-120b:novita

python inference.py --dataset-path dataset/py_tasks.csv --episodes-per-task 5
```

### `inference.py` flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--dataset-path` | `str` | `dataset/py_tasks.csv` | Processed task CSV used by env |
| `--episodes-per-task` | `int` | `5` | Episodes per selected task type |
| `--task-types` | `str` | `classify,root_cause,fix_proposal` | Comma-separated task types |
| `--no-progress` | bool | `False` | Disable progress bars |
| `--trace-agent` | bool | `False` | Print model output, action/tool call, and step results |
| `--trace-prompts` | bool | `False` | Also print prompts sent to the model |
| `--trace-max-chars` | `int` | `2500` | Max chars per traced block |

Trace to log:
```bash
python inference.py \
  --dataset-path dataset/py_tasks.csv \
  --episodes-per-task 5 \
  --task-types classify,root_cause \
  --trace-agent --trace-prompts > agent_trace.log 2>&1
```

## Known Issues

| Issue | Why it happens | What to do |
|---|---|---|
| `git checkout <sha>` fails during inference/dataset build | Repo history changed, SHA unavailable, or fetch issue | Rebuild dataset, skip bad rows with `--limit` for debugging, and check build summary fail reasons |
| Dataset build looks “stuck” after schema summary | It is cloning/fetching many repos and test files | Wait for the progress bar; test quickly with `--limit 2` first |
| Build ends with `kept: 0` and fetch failures | Some repos/SHAs/files are unavailable at source | Use the printed failure counters/examples; rerun later or filter problematic rows |
| Inference seems frozen when tracing to file | Large trace output + network model calls; stdout is redirected | Run fewer episodes first, reduce trace with `--trace-max-chars`, and tail logs in another terminal |
| Agent falls back to heuristic/root-cause guess | Model call failed/rate-limited/invalid JSON | Check `--trace-agent` logs, verify `HF_TOKEN`, and retry |
| All episodes show same score | Usually heuristic fallback + fixed step/reward pattern | Enable trace flags to confirm model output and actions |
| `rg` not found in sandbox | `ripgrep` isn’t installed in that runtime | The sandbox falls back to `grep` automatically |
| Episode cap at 20 steps | `MAX_STEPS` is currently fixed in code | Edit `MAX_STEPS` in `inference.py` if you need a larger limit |

## OpenEnv CLI

```bash
openenv/bin/openenv validate --json
openenv/bin/openenv build
openenv/bin/openenv push
```
