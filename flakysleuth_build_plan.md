# FlakySleuth — Comprehensive Round 1 Build Plan
## Meta × PyTorch × Scaler OpenEnv Hackathon

---

## 0. What You Are Building (One Paragraph for Clarity)

You are building an **OpenEnv-compliant RL environment** called `FlakySleuthEnv`. It simulates a real software engineering task: investigating flaky tests in real Python GitHub repositories. An LLM agent is dropped into a sandboxed repo at a specific commit, given a test that is known to be flaky (sourced from the IDoFT dataset), and must use tool calls (read files, grep code, run tests) to investigate and produce a verdict. The environment scores the agent's verdict using deterministic graders (Tasks 1 and 2) and a hybrid programmatic + LLM judge grader (Task 3). You are NOT training any model. The submitted artifact is the environment itself — its graders, reward logic, OpenEnv spec compliance, Docker container, and a baseline `inference.py` script that proves it works.

---

## 1. Repository Structure

```
flaky-sleuth-env/
│
├── inference.py                  ← REQUIRED: must be named exactly this, in root
├── openenv.yaml                  ← REQUIRED: OpenEnv spec metadata
├── Dockerfile                    ← REQUIRED: must build and run
├── requirements.txt
├── README.md
│
├── server.py                     ← FastAPI HTTP server (OpenEnv endpoints)
│
├── env/
│   ├── __init__.py
│   ├── models.py                 ← All Pydantic models (Observation, Action, Reward)
│   ├── environment.py            ← FlakySleuthEnv core class
│   ├── sandbox.py                ← Git clone, file read, grep, run_test
│   └── task_loader.py            ← Loads tasks from dataset CSV
│
├── graders/
│   ├── __init__.py               ← grade_action() dispatcher
│   ├── task1_grader.py           ← Binary flaky/stable
│   ├── task2_grader.py           ← Root cause category + similarity matrix
│   └── task3_grader.py           ← Fix proposal: pattern + diff + LLM judge
│
├── dataset/
│   ├── build_dataset.py          ← OFFLINE SCRIPT: preprocess IDoFT → py_tasks.csv
│   ├── py_tasks.csv              ← Final preprocessed task bank (committed to repo)
│   └── category_similarity.json  ← Similarity matrix for Task 2 partial credit
│
└── tests/
    └── test_compliance.py        ← openenv validate compliance checks
```

---

## 2. Data Pipeline (Do This First, Offline)

### 2.1 Download the Raw Dataset

```bash
git clone https://github.com/TestingResearchIllinois/idoft
# The file you need:
# idoft/py-data.csv
```

### 2.2 Understand the CSV Columns

The `py-data.csv` has these columns:
```
Project URL | SHA Detected | Pytest Test Name | Category | Status | PR Link | Notes
```

- **Project URL**: GitHub repo to clone
- **SHA Detected**: Exact commit to clone at (this is where the test IS flaky)
- **Pytest Test Name**: Format is `path/to/test_file.py::TestClass::test_method` or `path/to/test_file.py::test_method`
- **Category**: One of OD, OD-Brit, OD-Vic, NIO, NOD, UD, TD, TZD, ID, NDOI, NDOD, OSD (may be semicolon-separated for multiple)
- **Status**: Blank, Opened, Accepted, Rejected, etc.
- **PR Link**: Format `owner/repo#number` — only present when Status is Opened/Accepted

### 2.3 Filter Rules Per Task

```python
# Task 1 (classify): Use these categories — they have clear static signals
TASK1_CATEGORIES = ["NOD", "TD", "TZD", "NIO", "ID", "OD", "OD-Brit", "OD-Vic"]

# Task 2 (root cause): Same categories — agent must identify which one
TASK2_CATEGORIES = ["NOD", "TD", "TZD", "NIO", "ID", "OD", "OD-Brit", "OD-Vic"]
# Exclude "UD" (unknown — no ground truth to grade against)

# Task 3 (fix proposal): ONLY rows where a fix was accepted AND category is gradeable
TASK3_CATEGORIES = ["TD", "TZD", "NOD", "NIO", "ID"]
# Exclude: OD, OD-Brit, OD-Vic (cannot verify fix without multi-order execution)
# Exclude: UD (unknown cause = cannot score fix)
# Require: Status == "Accepted" AND PR Link is not empty
```

### 2.4 Build `py_tasks.csv` (the `build_dataset.py` script)

This script runs ONCE offline. It:
1. Reads `idoft/py-data.csv`
2. For each row, fetches the test source code by cloning the repo at SHA (or using GitHub raw API)
3. For Task 3 rows (Status=Accepted), fetches the PR diff from GitHub API
4. Outputs `dataset/py_tasks.csv`

```python
# dataset/build_dataset.py

import pandas as pd
import requests
import subprocess
import tempfile
import os

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]  # set this before running

def fetch_test_code(repo_url: str, sha: str, pytest_test_name: str) -> str:
    """
    Clone repo at SHA, extract the test function source code.
    pytest_test_name format: path/to/test.py::TestClass::test_method
    """
    test_file = pytest_test_name.split("::")[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run([
            "git", "clone", "--depth=1", repo_url, tmpdir
        ], capture_output=True)
        subprocess.run([
            "git", "checkout", sha
        ], cwd=tmpdir, capture_output=True)
        filepath = os.path.join(tmpdir, test_file)
        if not os.path.exists(filepath):
            return ""
        with open(filepath) as f:
            return f.read()[:5000]  # cap at 5000 chars


def fetch_pr_diff(pr_link: str) -> str:
    """
    pr_link format: "owner/repo#number"
    Returns unified diff string of the PR.
    """
    if not pr_link or "#" not in pr_link:
        return ""
    repo, number = pr_link.strip().split("#")
    url = f"https://api.github.com/repos/{repo}/pulls/{number}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.diff"
    }
    resp = requests.get(url, headers=headers, timeout=10)
    if resp.status_code == 200:
        return resp.text[:3000]  # cap diff size
    return ""


def build():
    df = pd.read_csv("idoft/py-data.csv")
    
    # Rename columns for clarity
    df.columns = [c.strip() for c in df.columns]
    
    rows = []
    for _, row in df.iterrows():
        repo_url = str(row.get("Project URL", "")).strip()
        sha = str(row.get("SHA Detected", "")).strip()
        test_name = str(row.get("Pytest Test Name", "")).strip()
        category_raw = str(row.get("Category", "")).strip()
        status = str(row.get("Status", "")).strip()
        pr_link = str(row.get("PR Link", "")).strip()
        
        # Skip rows with missing essentials
        if not repo_url or not sha or not test_name or not category_raw:
            continue
        
        # Take primary category (first if semicolon-separated)
        category = category_raw.split(";")[0].strip()
        
        # Skip UD for Task 2 (no ground truth)
        if category == "UD":
            continue
        
        # Determine task types this row is eligible for
        task_types = []
        if category in ["NOD", "TD", "TZD", "NIO", "ID", "OD", "OD-Brit", "OD-Vic"]:
            task_types.append("classify")
            task_types.append("root_cause")
        if (category in ["TD", "TZD", "NOD", "NIO", "ID"]
                and status == "Accepted"
                and pr_link and pr_link != "nan"):
            task_types.append("fix_proposal")
        
        if not task_types:
            continue
        
        # Fetch test source code
        test_code = fetch_test_code(repo_url, sha, test_name)
        if not test_code:
            continue
        
        # Fetch fix diff for Task 3 eligible rows
        known_fix_diff = ""
        if "fix_proposal" in task_types:
            known_fix_diff = fetch_pr_diff(pr_link)
        
        rows.append({
            "repo_url": repo_url,
            "sha": sha,
            "test_name": test_name,
            "test_file": test_name.split("::")[0],
            "category": category,
            "status": status,
            "pr_link": pr_link,
            "task_types": ";".join(task_types),
            "test_code": test_code,
            "known_fix_diff": known_fix_diff,
        })
    
    out = pd.DataFrame(rows)
    out.to_csv("dataset/py_tasks.csv", index=False)
    print(f"Built {len(out)} task rows")
    print(out["category"].value_counts())
    print(out["task_types"].value_counts())

if __name__ == "__main__":
    build()
```

### 2.5 Build `category_similarity.json`

```json
{
  "OD,OD-Brit": 0.7,
  "OD,OD-Vic": 0.7,
  "OD-Brit,OD-Vic": 0.8,
  "OD,NIO": 0.4,
  "OD,NDOI": 0.3,
  "NOD,TD": 0.6,
  "NOD,TZD": 0.5,
  "NOD,NDOI": 0.5,
  "TD,TZD": 0.7,
  "NOD,ID": 0.3,
  "UD,OD": 0.2,
  "UD,NOD": 0.2,
  "UD,NIO": 0.2,
  "UD,TD": 0.2,
  "UD,ID": 0.2
}
```

---

## 3. Pydantic Models (`env/models.py`)

```python
from pydantic import BaseModel
from typing import Literal, Optional, List

class FlakySleuthObservation(BaseModel):
    repo_url: str
    test_name: str
    test_code: str
    file_tree: List[str]
    tool_output: Optional[str] = None
    task_type: Literal["classify", "root_cause", "fix_proposal"]
    task_description: str
    step_count: int

class FlakySleuthAction(BaseModel):
    action_type: Literal[
        "read_file",
        "search_code",
        "run_test",
        "classify_flakiness",
        "classify_root_cause",
        "propose_fix",
    ]
    argument: str

class FlakySleuthReward(BaseModel):
    score: float
    breakdown: dict
    explanation: str
```

---

## 4. Sandbox (`env/sandbox.py`)

The sandbox wraps a cloned git repo. It handles all filesystem operations.

```python
import subprocess
import tempfile
import os
import shutil
from typing import Optional, List

class Sandbox:
    def __init__(self, task: dict):
        self.task = task
        self.tmpdir: Optional[str] = None
        self.file_tree: List[str] = []

    def setup(self):
        """Clone repo at the specific SHA. Called by env.reset()."""
        self.tmpdir = tempfile.mkdtemp(prefix="flakysleuth_")
        try:
            # Shallow clone for speed
            subprocess.run([
                "git", "clone", "--depth=50",
                self.task["repo_url"],
                self.tmpdir
            ], capture_output=True, timeout=60, check=True)

            # Checkout exact SHA where flakiness was detected
            subprocess.run([
                "git", "checkout", self.task["sha"]
            ], cwd=self.tmpdir, capture_output=True, timeout=30, check=True)

            self.file_tree = self._build_file_tree()
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Sandbox setup failed: {e}")

    def read_file(self, relative_path: str) -> Optional[str]:
        """Read a file relative to repo root. Returns None if not found."""
        full_path = os.path.normpath(os.path.join(self.tmpdir, relative_path))
        # Security: ensure path stays inside tmpdir
        if not full_path.startswith(self.tmpdir):
            return None
        if not os.path.isfile(full_path):
            return None
        try:
            with open(full_path, "r", errors="replace") as f:
                return f.read()[:4000]  # cap to avoid huge files
        except Exception:
            return None

    def grep(self, pattern: str) -> str:
        """Grep for pattern across all .py files in the repo."""
        if not self.tmpdir:
            return "ERROR: Sandbox not initialized"
        try:
            result = subprocess.run(
                ["grep", "-rn", "--include=*.py", pattern, "."],
                cwd=self.tmpdir,
                capture_output=True,
                text=True,
                timeout=10
            )
            output = result.stdout[:2000]
            return output if output else f"No matches found for: {pattern}"
        except subprocess.TimeoutExpired:
            return "Search timed out"
        except Exception as e:
            return f"Search error: {e}"

    def run_test(self, pytest_test_name: str) -> str:
        """
        Run the specific test via pytest.
        ONLY called for non-OD tasks.
        """
        if self.task["category"] in ("OD", "OD-Brit", "OD-Vic"):
            return (
                "Test execution skipped for order-dependent tests. "
                "Use read_file and search_code to analyze static code structure instead. "
                "Look for: shared state, missing setUp/tearDown, module-scoped fixtures, global mutations."
            )
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", pytest_test_name,
                 "--tb=short", "-x", "--timeout=30", "-q"],
                cwd=self.tmpdir,
                capture_output=True,
                text=True,
                timeout=60
            )
            output = (result.stdout + result.stderr)[:2000]
            return output if output else "Test completed with no output"
        except subprocess.TimeoutExpired:
            return "Test execution timed out (>60s)"
        except Exception as e:
            return f"Test execution error: {e}"

    def cleanup(self):
        """Remove temp directory. Called after episode ends."""
        if self.tmpdir and os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir, ignore_errors=True)
        self.tmpdir = None
        self.file_tree = []

    def _build_file_tree(self) -> List[str]:
        """Return top-2-level file paths relative to repo root."""
        result = []
        for root, dirs, files in os.walk(self.tmpdir):
            # Skip hidden dirs and common noise
            dirs[:] = [d for d in dirs if not d.startswith(".")
                       and d not in ("node_modules", "__pycache__", ".git", "venv", ".tox")]
            depth = root.replace(self.tmpdir, "").count(os.sep)
            if depth <= 2:
                for f in files:
                    rel = os.path.relpath(os.path.join(root, f), self.tmpdir)
                    result.append(rel)
            if len(result) > 100:
                break
        return result[:100]
```

---

## 5. Task Loader (`env/task_loader.py`)

```python
import pandas as pd
import random
from typing import Optional

class TaskLoader:
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        # Expand task_types column into individual rows
        rows = []
        for _, row in df.iterrows():
            for tt in str(row["task_types"]).split(";"):
                r = row.to_dict()
                r["task_type"] = tt.strip()
                rows.append(r)
        self.tasks = rows
        self._forced_type: Optional[str] = None

    def sample(self) -> dict:
        """Sample a random task, optionally filtered by type."""
        pool = self.tasks
        if self._forced_type:
            pool = [t for t in self.tasks if t["task_type"] == self._forced_type]
        task = random.choice(pool).copy()
        task["task_description"] = self._make_description(task)
        return task

    def force_task_type(self, task_type: str):
        """Force next sample() calls to return a specific task type."""
        self._forced_type = task_type

    def _make_description(self, task: dict) -> str:
        tt = task["task_type"]
        if tt == "classify":
            return (
                "Investigate the given test and determine whether it is FLAKY or STABLE. "
                "Use read_file and search_code to gather evidence. "
                "When confident, call classify_flakiness with argument 'flaky' or 'stable'."
            )
        elif tt == "root_cause":
            return (
                f"This test is confirmed flaky. Identify its root cause category. "
                f"Valid categories: OD, OD-Brit, OD-Vic, NIO, NOD, TD, TZD, ID, NDOI. "
                f"Use read_file and search_code to find evidence. "
                f"Call classify_root_cause with the category code when confident."
            )
        elif tt == "fix_proposal":
            return (
                f"This test is confirmed flaky with root cause: {task['category']}. "
                f"Propose a concrete fix as a unified diff. "
                f"Use read_file and search_code to understand the code. "
                f"Call propose_fix with a valid unified diff string."
            )
        return "Investigate the flaky test."
```

---

## 6. Core Environment (`env/environment.py`)

```python
import random
from env.models import FlakySleuthObservation, FlakySleuthAction
from env.sandbox import Sandbox
from env.task_loader import TaskLoader
from graders import grade_action

FLAKY_SIGNAL_PATTERNS = [
    "sleep", "random", "time", "datetime", "thread", "asyncio",
    "fixture", "setUp", "tearDown", "global", "shared", "singleton",
    "os.environ", "socket", "timeout", "retry", "mock", "patch"
]

class FlakySleuthEnv:
    def __init__(self, dataset_path: str = "dataset/py_tasks.csv"):
        self.loader = TaskLoader(dataset_path)
        self.sandbox: Sandbox = None
        self.current_task: dict = None
        self.step_count: int = 0
        self.cumulative_progress: float = 0.0
        self.files_read: set = set()
        self.episode_actions: list = []

    def reset(self) -> FlakySleuthObservation:
        # Cleanup previous episode
        if self.sandbox:
            self.sandbox.cleanup()
        
        # Sample new task
        self.current_task = self.loader.sample()
        self.sandbox = Sandbox(self.current_task)
        self.sandbox.setup()
        
        # Reset episode state
        self.step_count = 0
        self.cumulative_progress = 0.0
        self.files_read = set()
        self.episode_actions = []
        
        return self._make_obs()

    def step(self, action: FlakySleuthAction):
        self.step_count += 1
        self.episode_actions.append(action)
        tool_output = None
        reward = 0.0
        done = False
        info = {}

        TERMINAL_ACTIONS = ("classify_flakiness", "classify_root_cause", "propose_fix")

        if action.action_type in TERMINAL_ACTIONS:
            # Grade terminal action
            terminal_score = grade_action(action, self.current_task)
            
            # Late step penalty: -0.05 per step beyond 15
            late_penalty = max(0, (self.step_count - 15)) * 0.05
            
            # Wrong-direction penalty for T1
            wrong_dir_penalty = 0.0
            if (action.action_type == "classify_flakiness"
                    and action.argument.lower() == "stable"
                    and self.current_task.get("label") == "flaky"):
                wrong_dir_penalty = 0.2
            
            reward = min(0.999, max(0.001,
                self.cumulative_progress + terminal_score
                - late_penalty - wrong_dir_penalty
            ))
            done = True
            info = {
                "terminal_score": terminal_score,
                "progress_score": self.cumulative_progress,
                "late_penalty": late_penalty,
                "task_type": self.current_task["task_type"],
                "category": self.current_task["category"],
            }

        else:
            # Exploratory action
            tool_output, progress = self._execute_exploration(action)
            self.cumulative_progress = min(0.30, self.cumulative_progress + progress)
            reward = progress

        obs = self._make_obs(tool_output)
        return obs, reward, done, info

    def state(self) -> dict:
        return {
            "repo_url": self.current_task["repo_url"] if self.current_task else None,
            "test_name": self.current_task["test_name"] if self.current_task else None,
            "task_type": self.current_task["task_type"] if self.current_task else None,
            "step_count": self.step_count,
            "files_read": list(self.files_read),
            "cumulative_progress": self.cumulative_progress,
        }

    def _execute_exploration(self, action: FlakySleuthAction):
        progress = 0.0
        output = ""

        if action.action_type == "read_file":
            content = self.sandbox.read_file(action.argument)
            if content is None:
                output = f"ERROR: File not found: {action.argument}"
                progress = -0.05  # hallucination penalty
            elif action.argument in self.files_read:
                output = content
                progress = 0.0   # no reward for re-read
            else:
                self.files_read.add(action.argument)
                output = content
                progress = self._file_relevance_reward(action.argument)

        elif action.action_type == "search_code":
            output = self.sandbox.grep(action.argument)
            progress = self._search_relevance_reward(action.argument)

        elif action.action_type == "run_test":
            output = self.sandbox.run_test(self.current_task["test_name"])
            # Reward for actually running the test (shows initiative)
            # But 0 if OD task (sandbox returns static message)
            if self.current_task["category"] not in ("OD", "OD-Brit", "OD-Vic"):
                progress = 0.05

        return output, progress

    def _file_relevance_reward(self, filepath: str) -> float:
        task = self.current_task
        test_file = task.get("test_file", "")
        
        if test_file and test_file in filepath:
            return 0.0017   # reading the actual test file
        if any(filepath.endswith(ext) for ext in (".py",)):
            return 0.0013   # any python file
        return 0.0011       # non-python file (requirements, config, etc.)

    def _search_relevance_reward(self, pattern: str) -> float:
        pattern_lower = pattern.lower()
        if any(sig in pattern_lower for sig in FLAKY_SIGNAL_PATTERNS):
            return 0.0014   # searching for known flakiness signals
        return 0.0011       # generic search

    def _make_obs(self, tool_output=None) -> FlakySleuthObservation:
        task = self.current_task
        return FlakySleuthObservation(
            repo_url=task["repo_url"],
            test_name=task["test_name"],
            test_code=task.get("test_code", "")[:2000],
            file_tree=self.sandbox.file_tree if self.sandbox else [],
            tool_output=tool_output,
            task_type=task["task_type"],
            task_description=task["task_description"],
            step_count=self.step_count,
        )
```

---

## 7. Graders

### 7.1 Dispatcher (`graders/__init__.py`)

```python
from env.models import FlakySleuthAction
from graders.task1_grader import grade as grade_t1
from graders.task2_grader import grade as grade_t2
from graders.task3_grader import grade as grade_t3

def grade_action(action: FlakySleuthAction, task: dict) -> float:
    tt = task["task_type"]
    if tt == "classify":
        return grade_t1(action, task)
    elif tt == "root_cause":
        return grade_t2(action, task)
    elif tt == "fix_proposal":
        return grade_t3(action, task)
    return 0.001
```

### 7.2 Task 1 Grader (`graders/task1_grader.py`)

```python
from env.models import FlakySleuthAction

def grade(action: FlakySleuthAction, task: dict) -> float:
    """Binary classification: flaky or stable. Exact match only."""
    if action.action_type != "classify_flakiness":
        return 0.001
    
    predicted = action.argument.strip().lower()
    if predicted not in ("flaky", "stable"):
        return 0.001
    
    # All IDoFT rows are flaky; stable examples are synthetically added
    # with label="stable" during dataset construction
    ground_truth = task.get("label", "flaky")
    return 0.999 if predicted == ground_truth else 0.0
```

### 7.3 Task 2 Grader (`graders/task2_grader.py`)

```python
import json
import os
from env.models import FlakySleuthAction

# Load similarity matrix once at module level
_SIM_PATH = os.path.join(os.path.dirname(__file__), 
                          "..", "dataset", "category_similarity.json")
with open(_SIM_PATH) as f:
    _RAW_SIM = json.load(f)

def _get_similarity(pred: str, true: str) -> float:
    if pred == true:
        return 0.999
    key1 = f"{pred},{true}"
    key2 = f"{true},{pred}"
    return _RAW_SIM.get(key1, _RAW_SIM.get(key2, 0.0))

VALID_CATEGORIES = {
    "OD", "OD-Brit", "OD-Vic", "NIO", "NOD",
    "UD", "TD", "TZD", "ID", "NDOI", "NDOD", "OSD"
}

def grade(action: FlakySleuthAction, task: dict) -> float:
    """
    Root cause category classification.
    Exact match = 1.0
    Related category = partial credit via similarity matrix
    Wrong family = 0.0
    """
    if action.action_type != "classify_root_cause":
        return 0.001
    
    predicted = action.argument.strip().upper()
    
    # Handle common variations
    predicted = predicted.replace(" ", "-")  # "OD Brit" → "OD-Brit"
    
    if predicted not in VALID_CATEGORIES:
        return 0.001   # invalid category string
    
    # Take primary category from dataset (first if semicolon-separated)
    true_category = str(task.get("category", "")).split(";")[0].strip().upper()
    
    return _get_similarity(predicted, true_category)
```

### 7.4 Task 3 Grader (`graders/task3_grader.py`)

```python
import subprocess
import tempfile
import os
import json
from openai import OpenAI
from env.models import FlakySleuthAction

CATEGORY_DESCRIPTIONS = {
    "TD":   "Time-Dependent: test fails due to reliance on wall-clock time",
    "TZD":  "Timezone-Dependent: test fails in different timezones",
    "NOD":  "Non-Deterministic: test fails due to randomness or non-determinism",
    "NIO":  "Non-Idempotent-Outcome: test passes first run but fails on second run",
    "ID":   "Implementation-Dependent: test fails due to language/runtime non-determinism (e.g. dict ordering)",
}

EXPECTED_FIX_PATTERNS = {
    "TD":   ["freeze_time", "mock", "patch", "utcnow", "datetime", "monkeypatch"],
    "TZD":  ["timezone", "utc", "pytz", "zoneinfo", "tzinfo", "UTC"],
    "NOD":  ["seed", "mock", "patch", "deterministic", "sorted"],
    "NIO":  ["setUp", "tearDown", "fixture", "yield", "cleanup", "autouse"],
    "ID":   ["sorted(", "list(", "frozenset", "OrderedDict"],
}

def grade(action: FlakySleuthAction, task: dict) -> float:
    """
    Fix proposal grader.
    Component A: Pattern check     — 0.35 weight
    Component B: Diff applies      — 0.25 weight  
    Component C: LLM judge         — 0.40 weight
    """
    if action.action_type != "propose_fix":
        return 0.001
    
    proposed_fix = action.argument.strip()
    if not proposed_fix:
        return 0.001
    
    category = str(task.get("category", "")).split(";")[0].strip().upper()
    known_fix = task.get("known_fix_diff", "") or ""
    test_code = task.get("test_code", "") or ""
    
    # ── Component A: Pattern check ────────────────────────────────
    patterns = EXPECTED_FIX_PATTERNS.get(category, [])
    if patterns:
        matches = sum(1 for p in patterns if p in proposed_fix)
        pattern_score = min(0.999, matches / max(1, len(patterns) * 0.4))
    else:
        pattern_score = 0.5
    
    # ── Component B: Diff applies cleanly ─────────────────────────
    apply_score = _check_diff_applies(proposed_fix, task)
    
    # ── Component C: LLM judge ────────────────────────────────────
    judge_score = _llm_judge(proposed_fix, known_fix, category, test_code)
    
    total = (0.35 * pattern_score) + (0.25 * apply_score) + (0.40 * judge_score)
    return round(min(0.999, max(0.001, total)), 4)


def _check_diff_applies(fix: str, task: dict) -> float:
    """Try a dry-run patch application against the test file in a temp copy."""
    try:
        test_file = task.get("test_file", "")
        sandbox_path = task.get("sandbox_test_path", "")
        
        if not sandbox_path or not os.path.exists(sandbox_path):
            return 0.3  # can't verify, neutral-ish
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(fix)
            patch_path = f.name
        
        result = subprocess.run(
            ["patch", "--dry-run", "-p1", sandbox_path, patch_path],
            capture_output=True, text=True, timeout=10
        )
        os.unlink(patch_path)
        return 0.999 if result.returncode == 0 else 0.0
    except Exception:
        return 0.3  # can't verify, neutral


def _llm_judge(proposed: str, known: str, category: str, test_code: str) -> float:
    """Call the LLM judge via OpenAI-compatible API."""
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
    )
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    
    cat_desc = CATEGORY_DESCRIPTIONS.get(category, f"Flakiness category: {category}")
    known_section = f"Known accepted fix (from merged PR):\n```\n{known[:800]}\n```" if known else "Known fix: Not available"
    
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
- 0–2: Fix is wrong, irrelevant, or makes things worse
- 3–5: Fix partially addresses the issue but misses root cause
- 6–8: Fix correctly addresses root cause with minor issues
- 9–10: Fix is correct, clean, minimal, and addresses root cause completely

Respond ONLY with a JSON object and nothing else:
{{"score": <integer 0-10>, "reason": "<one sentence explanation>"}}"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        score = int(data["score"])
        return max(0.0, min(10.0, score)) / 10.0
    except Exception:
        return 0.5  # fallback neutral on any failure
```

---

## 8. OpenEnv HTTP Server (`server.py`)

```python
from fastapi import FastAPI, HTTPException
from env.models import FlakySleuthObservation, FlakySleuthAction
from env.environment import FlakySleuthEnv

app = FastAPI(title="FlakySleuth Environment")
env = FlakySleuthEnv()

@app.post("/reset")
def reset() -> FlakySleuthObservation:
    return env.reset()

@app.post("/step")
def step(action: FlakySleuthAction):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info,
    }

@app.get("/state")
def state():
    return env.state()

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
```

---

## 9. `openenv.yaml`

```yaml
name: flaky-sleuth-env
version: 0.1.0
description: >
  An RL environment where an LLM agent investigates flaky tests in real
  Python GitHub repositories. The agent uses tool calls to read code,
  search for patterns, and run tests — then produces a verdict (classify,
  root cause, or fix). Tasks range from binary flakiness classification
  to proposing concrete code fixes verified by a hybrid grader.

observation_type: FlakySleuthObservation
action_type: FlakySleuthAction
reward_range: (0.001, 0.999)

tasks:
  - id: task1_classify
    name: "Flaky vs. Stable Classification"
    difficulty: easy
    description: >
      Given a test from a real Python repo, classify it as flaky or stable.
      Agent must call classify_flakiness with argument 'flaky' or 'stable'.

  - id: task2_root_cause
    name: "Root Cause Category Identification"
    difficulty: medium
    description: >
      Given a confirmed flaky test, identify the root cause category
      (OD, NOD, TD, TZD, NIO, ID, etc.) via static code analysis.

  - id: task3_fix_proposal
    name: "Fix Proposal"
    difficulty: hard
    description: >
      Given a confirmed flaky test and its root cause, propose a concrete
      fix as a unified diff. Evaluated by pattern matching + LLM judge.

episode_max_steps: 20
baseline_script: inference.py

infra:
  vcpu: 2
  memory_gb: 8
  max_inference_minutes: 20
```

---

## 10. Baseline Inference Script (`inference.py`)

**CRITICAL:** Must be named exactly `inference.py` in the root directory. Must use OpenAI client. Must read `API_BASE_URL`, `MODEL_NAME`, `OPENAI_API_KEY` from environment variables.

```python
"""
FlakySleuth baseline inference script.

Required environment variables:
  OPENAI_API_KEY  — API key
  API_BASE_URL    — LLM endpoint (default: https://api.openai.com/v1)
  MODEL_NAME      — Model identifier (default: gpt-4o-mini)

Runs 5 episodes × 3 task types = 15 total episodes.
Prints average score per task type.
Must complete in under 20 minutes on vcpu=2, 8GB RAM.
"""

import os
import json
from openai import OpenAI
from env.environment import FlakySleuthEnv
from env.models import FlakySleuthAction

# ── Configuration ──────────────────────────────────────────────────
API_KEY      = os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
EPISODES_PER_TASK = 5

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ── System prompt (teaches the model your tool interface) ──────────
SYSTEM_PROMPT = """You are a flaky test detective. You investigate Python tests in real GitHub repositories.

At each step, respond ONLY with a single valid JSON object — no explanation, no markdown, no extra text.

Available actions:

EXPLORATORY (use these to gather evidence):
{"action_type": "read_file", "argument": "relative/path/to/file.py"}
{"action_type": "search_code", "argument": "pattern_to_grep_for"}
{"action_type": "run_test", "argument": ""}

TERMINAL (use exactly one of these to end the episode):
{"action_type": "classify_flakiness", "argument": "flaky"}
{"action_type": "classify_flakiness", "argument": "stable"}
{"action_type": "classify_root_cause", "argument": "OD"}
{"action_type": "classify_root_cause", "argument": "NOD"}
{"action_type": "classify_root_cause", "argument": "TD"}
{"action_type": "classify_root_cause", "argument": "TZD"}
{"action_type": "classify_root_cause", "argument": "NIO"}
{"action_type": "classify_root_cause", "argument": "ID"}
{"action_type": "classify_root_cause", "argument": "OD-Brit"}
{"action_type": "classify_root_cause", "argument": "OD-Vic"}
{"action_type": "propose_fix", "argument": "--- a/path\\n+++ b/path\\n@@ ... @@\\n-old line\\n+new line"}

RULES:
1. Always read the test file first before making a terminal decision.
2. Search for flakiness signals: sleep, random, time, datetime, thread, os.environ, shared state.
3. For order-dependent (OD) tests, run_test is disabled — use static analysis only.
4. Call a terminal action only when you have enough evidence.
5. Respond with ONLY valid JSON. Nothing else."""


def obs_to_prompt(obs) -> str:
    return f"""TASK: {obs.task_description}

Repository: {obs.repo_url}
Test name: {obs.test_name}
Step: {obs.step_count}/20

Test source code:
```python
{obs.test_code}
```

Repository file tree (top-level):
{chr(10).join(obs.file_tree[:40])}

Result of your last action:
{obs.tool_output or "(No action taken yet — this is the start of the episode)"}

What is your next action? Respond with JSON only."""


def run_episode(env: FlakySleuthEnv) -> float:
    obs = env.reset()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": obs_to_prompt(obs)},
    ]
    total_reward = 0.0

    for step in range(20):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=400,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": raw})

            # Parse action
            clean = raw.replace("```json", "").replace("```", "").strip()
            action_dict = json.loads(clean)
            action = FlakySleuthAction(**action_dict)

        except json.JSONDecodeError:
            # Model produced non-JSON — inject correction message
            messages.append({
                "role": "user",
                "content": "ERROR: Your response was not valid JSON. "
                           "Respond ONLY with a JSON object as specified."
            })
            continue
        except Exception as e:
            print(f"  Step {step} error: {e}")
            break

        obs, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            print(f"  Terminal: {action.action_type}({action.argument[:50]}) "
                  f"→ terminal={info.get('terminal_score', 0):.2f} "
                  f"progress={info.get('progress_score', 0):.2f} "
                  f"total={total_reward:.2f}")
            break

        messages.append({"role": "user", "content": obs_to_prompt(obs)})

    return total_reward


def main():
    env = FlakySleuthEnv()
    results = {"classify": [], "root_cause": [], "fix_proposal": []}

    for task_type in results.keys():
        print(f"\n── Task type: {task_type} ──")
        env.loader.force_task_type(task_type)
        for ep in range(EPISODES_PER_TASK):
            score = run_episode(env)
            results[task_type].append(score)
            print(f"  Episode {ep+1}: {score:.3f}")

    print("\n══ BASELINE RESULTS ══")
    for task_type, scores in results.items():
        avg = sum(scores) / len(scores)
        print(f"  {task_type:15s}: avg={avg:.3f}  scores={[round(s,3) for s in scores]}")
    
    overall = sum(s for scores in results.values() for s in scores)
    overall /= sum(len(v) for v in results.values())
    print(f"  {'OVERALL':15s}: avg={overall:.3f}")


if __name__ == "__main__":
    main()
```

---

## 11. Dockerfile

```dockerfile
FROM python:3.11-slim

# Install git and patch (needed for sandbox)
RUN apt-get update && apt-get install -y \
    git \
    patch \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

# Expose port for HF Spaces
EXPOSE 7860

# Start FastAPI server
CMD ["python", "server.py"]
```

---

## 12. `requirements.txt`

```
fastapi>=0.110.0
uvicorn>=0.27.0
pydantic>=2.0.0
openai>=1.0.0
pandas>=2.0.0
gitpython>=3.1.0
pytest>=7.0.0
pytest-timeout>=2.0.0
requests>=2.31.0
```

---

## 13. Build Order (Day-by-Day Sprint)

```
DAY 1 — Data Foundation
────────────────────────
□ Clone idoft repo, inspect py-data.csv manually
□ Run build_dataset.py offline (set GITHUB_TOKEN)
□ Verify py_tasks.csv has rows for all 3 task types
□ Manually inspect 5-10 rows to sanity check test_code and known_fix_diff
□ Build category_similarity.json

DAY 2 — Core Environment
──────────────────────────
□ Implement env/models.py (Pydantic models)
□ Implement env/sandbox.py (clone, read_file, grep, run_test)
□ Test sandbox.py manually on 2-3 real repos
□ Implement env/task_loader.py
□ Implement env/environment.py (reset, step, state)
□ Write a quick smoke test: reset() → 3 steps → terminal action

DAY 3 — Graders
────────────────
□ Implement graders/task1_grader.py
□ Implement graders/task2_grader.py + verify similarity matrix
□ Implement graders/task3_grader.py (pattern + diff + LLM judge)
□ Unit test all 3 graders with hardcoded inputs
□ Verify scores are always in (0.001, 0.999)

DAY 4 — Server + Spec Compliance
──────────────────────────────────
□ Implement server.py (FastAPI: /reset, /step, /state, /health)
□ Write openenv.yaml
□ Run openenv validate — fix any errors
□ Build Dockerfile locally: docker build . && docker run -p 7860:7860
□ Test endpoints with curl

DAY 5 — Inference Script + Deploy
────────────────────────────────────
□ Implement inference.py (ReAct loop, OpenAI client)
□ Run inference.py locally against real API
□ Verify it completes in <20 min, produces scores for all 3 task types
□ Deploy to Hugging Face Spaces
□ Verify HF Space returns 200 on health check and responds to reset()
□ Run pre-submission validation script

DAY 6 — Polish + Submit
─────────────────────────
□ Write README (env description, observation/action spaces, setup)
□ Run full baseline one more time, record scores
□ Submit HF Space URL before April 8 11:59 PM IST
```

---

## 14. Pre-Submission Checklist (from Official Spec)

```
□ HF Space deploys and returns 200 on automated ping
□ reset() responds correctly
□ openenv validate passes (openenv.yaml + typed models + step/reset/state)
□ docker build succeeds on submitted repo
□ inference.py runs without error and produces scores
□ 3 tasks with graders, all scores in 0.0–1.0
□ API_BASE_URL, MODEL_NAME, OPENROUTER_API_KEY env vars defined
□ Inference script is named exactly inference.py in root directory
□ All LLM calls use OpenAI client with those env vars
□ Runtime < 20 min on vcpu=2, 8GB RAM
```

---

## 15. Key Design Decisions Summary (for context)

| Decision | Choice | Reason |
|---|---|---|
| Language | Python only | Fast sandboxing, clean IDoFT data, no JVM overhead |
| Dataset | IDoFT py-data.csv + category codes | Real repos, ground truth categories, PR-linked fixes |
| OD tests in T3 | Excluded | Cannot verify fix without multi-order test execution |
| OD tests in T1/T2 | Included | Static code analysis is a valid proxy |
| T2 grader | Similarity matrix | Some wrong answers are more wrong than others |
| T3 grader | Hybrid (pattern + diff + LLM judge) | Pure string match unfair; pure LLM judge non-deterministic |
| Reward shaping | Step-level progress rewards | Prevents sparse reward, rewards good investigative behavior |
| Max steps | 20 | Balances exploration depth vs infra time constraints |
| Progress reward cap | 0.30 | Terminal score (0.70 max) dominates; exploration is supporting signal |
