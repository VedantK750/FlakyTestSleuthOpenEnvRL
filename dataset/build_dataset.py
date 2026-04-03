"""Offline dataset builder for FlakySleuth.

Examples:
  # Validate schema and show category/status summary only
  python dataset/build_dataset.py --input py-data.csv --validate-only

  # Build full task CSV (requires network access for repo cloning)
  export GITHUB_TOKEN=...
  python dataset/build_dataset.py --input py-data.csv --output dataset/py_tasks.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

TASK12_CATEGORIES = ["NOD", "TD", "TZD", "NIO", "ID", "OD", "OD-Brit", "OD-Vic"]
TASK3_CATEGORIES = ["TD", "TZD", "NOD", "NIO", "ID"]

PROJECT_URL_COL = "Project URL"
SHA_COL = "SHA Detected"
CATEGORY_COL = "Category"
STATUS_COL = "Status"
PR_LINK_COL = "PR Link"
NOTES_COL = "Notes"
TEST_NAME_ALIASES = [
    "Pytest Test Name",
    "Pytest Test Name (PathToFile::TestClass::TestMethod or PathToFile::TestMethod)",
]


def _normalize_header(text: str) -> str:
    return " ".join(str(text).strip().split())


def _resolve_test_name_column(columns: list[str]) -> str:
    normalized = {_normalize_header(c): c for c in columns}
    for alias in TEST_NAME_ALIASES:
        key = _normalize_header(alias)
        if key in normalized:
            return normalized[key]
    raise KeyError(
        "Could not find pytest test-name column. Expected one of: "
        + ", ".join(TEST_NAME_ALIASES)
    )


def _parse_pr_link(pr_link: str) -> tuple[str, str] | None:
    """Return (owner/repo, number) from URL or owner/repo#number."""
    value = (pr_link or "").strip()
    if not value or value.lower() == "nan":
        return None

    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        parts = [p for p in parsed.path.split("/") if p]
        # Expected: /owner/repo/pull/number
        if len(parts) >= 4 and parts[2] == "pull" and parts[3].isdigit():
            return f"{parts[0]}/{parts[1]}", parts[3]
        return None

    if "#" in value:
        repo, number = value.split("#", 1)
        if repo.strip() and number.strip().isdigit():
            return repo.strip(), number.strip()
    return None


def _is_accepted_status(status: str) -> bool:
    value = (status or "").strip().lower()
    return value in {"accepted", "merged", "fixed"}


def fetch_test_code(repo_url: str, sha: str, pytest_test_name: str) -> tuple[str, str, str]:
    test_file = pytest_test_name.split("::")[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            init = subprocess.run(
                ["git", "init", tmpdir],
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
            )
            if init.returncode != 0:
                return "", "git_init_failed", (init.stderr or init.stdout or "").strip()[:200]

            remote = subprocess.run(
                ["git", "-C", tmpdir, "remote", "add", "origin", repo_url],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if remote.returncode != 0:
                return "", "git_remote_add_failed", (remote.stderr or remote.stdout or "").strip()[:200]

            # Fetch only the requested commit for speed and correctness.
            fetch = subprocess.run(
                ["git", "-C", tmpdir, "fetch", "--depth=1", "origin", sha],
                capture_output=True,
                text=True,
                check=False,
                timeout=90,
            )
            if fetch.returncode != 0:
                return "", "git_fetch_sha_failed", (fetch.stderr or fetch.stdout or "").strip()[:200]

            checkout = subprocess.run(
                ["git", "-C", tmpdir, "checkout", "--detach", "FETCH_HEAD"],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            if checkout.returncode != 0:
                return "", "git_checkout_failed", (checkout.stderr or checkout.stdout or "").strip()[:200]
        except subprocess.TimeoutExpired:
            return "", "git_timeout", "timeout"

        file_path = Path(tmpdir) / test_file
        if not file_path.exists():
            return "", "test_file_missing_at_sha", test_file
        return file_path.read_text(encoding="utf-8", errors="replace")[:5000], "", ""


def fetch_pr_diff(pr_link: str, github_token: str) -> str:
    parsed = _parse_pr_link(pr_link)
    if not parsed:
        return ""

    repo, number = parsed
    url = f"https://api.github.com/repos/{repo}/pulls/{number}"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.diff",
    }
    response = requests.get(url, headers=headers, timeout=15)
    if response.status_code == 200:
        return response.text[:3000]
    return ""


def _validate_schema(input_csv: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(input_csv)
    df.columns = [_normalize_header(col) for col in df.columns]

    missing = []
    for required in [PROJECT_URL_COL, SHA_COL, CATEGORY_COL, STATUS_COL, PR_LINK_COL]:
        if required not in df.columns:
            missing.append(required)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    test_name_col = _resolve_test_name_column(list(df.columns))
    return df, test_name_col


def _print_input_summary(df: pd.DataFrame, test_name_col: str) -> None:
    print("Input schema check: OK")
    print(f"Rows: {len(df)}")
    print(f"Using test-name column: {test_name_col}")
    print("Columns:", list(df.columns))
    print("\nCategory distribution (top 20):")
    print(df[CATEGORY_COL].fillna("").astype(str).value_counts().head(20))
    print("\nStatus distribution:")
    print(df[STATUS_COL].fillna("").astype(str).value_counts().head(20))


def build(
    input_csv: str,
    output_csv: str,
    github_token: str,
    *,
    validate_only: bool = False,
    limit: int | None = None,
) -> None:
    df, test_name_col = _validate_schema(input_csv)
    _print_input_summary(df, test_name_col)
    if validate_only:
        return

    total_rows = min(len(df), limit) if limit is not None else len(df)
    print(
        f"\nStarting build over {total_rows} rows "
        f"(this can take a while: cloning repos + reading files + optional PR diff fetch)"
    )

    stats: dict[str, int] = {
        "kept": 0,
        "skipped_missing_core_fields": 0,
        "skipped_ud": 0,
        "skipped_no_task_types": 0,
        "skipped_test_code_fetch_failed": 0,
        "skipped_test_code_fetch_git_fail": 0,
        "skipped_test_code_fetch_file_missing": 0,
        "fix_diff_fetched": 0,
    }
    fetch_fail_examples: list[dict[str, str]] = []

    rows = []
    iterator = df.iterrows()
    if tqdm is not None:
        iterator = tqdm(iterator, total=total_rows, desc="Building tasks", unit="row")

    processed = 0
    for idx, (_, row) in enumerate(iterator, start=1):
        if idx > total_rows:
            break
        processed = idx

        repo_url = str(row.get(PROJECT_URL_COL, "")).strip()
        sha = str(row.get(SHA_COL, "")).strip()
        test_name = str(row.get(test_name_col, "")).strip()
        category_raw = str(row.get(CATEGORY_COL, "")).strip()
        status = str(row.get(STATUS_COL, "")).strip()
        pr_link = str(row.get(PR_LINK_COL, "")).strip()

        if not repo_url or not sha or not test_name or not category_raw:
            stats["skipped_missing_core_fields"] += 1
            _update_progress(iterator, tqdm, stats)
            continue

        category = category_raw.split(";")[0].strip()
        if category == "UD":
            stats["skipped_ud"] += 1
            _update_progress(iterator, tqdm, stats)
            continue

        task_types: list[str] = []
        if category in TASK12_CATEGORIES:
            task_types.extend(["classify", "root_cause"])
        if category in TASK3_CATEGORIES and _is_accepted_status(status) and _parse_pr_link(pr_link):
            task_types.append("fix_proposal")

        if not task_types:
            stats["skipped_no_task_types"] += 1
            _update_progress(iterator, tqdm, stats)
            continue

        test_code, fetch_reason, fetch_detail = fetch_test_code(repo_url, sha, test_name)
        if not test_code:
            stats["skipped_test_code_fetch_failed"] += 1
            if fetch_reason in {
                "git_init_failed",
                "git_remote_add_failed",
                "git_fetch_sha_failed",
                "git_checkout_failed",
                "git_timeout",
            }:
                stats["skipped_test_code_fetch_git_fail"] += 1
            if fetch_reason == "test_file_missing_at_sha":
                stats["skipped_test_code_fetch_file_missing"] += 1
            if len(fetch_fail_examples) < 10:
                fetch_fail_examples.append(
                    {
                        "repo_url": repo_url,
                        "sha": sha,
                        "test_name": test_name,
                        "reason": fetch_reason,
                        "detail": fetch_detail,
                    }
                )
            _update_progress(iterator, tqdm, stats)
            continue

        known_fix_diff = ""
        if "fix_proposal" in task_types and github_token:
            known_fix_diff = fetch_pr_diff(pr_link, github_token)
            if known_fix_diff:
                stats["fix_diff_fetched"] += 1

        rows.append(
            {
                "repo_url": repo_url,
                "sha": sha,
                "test_name": test_name,
                "test_file": test_name.split("::")[0],
                "category": category,
                "label": "flaky",
                "status": status,
                "pr_link": pr_link,
                "task_types": ";".join(task_types),
                "test_code": test_code,
                "known_fix_diff": known_fix_diff,
            }
        )
        stats["kept"] += 1
        _update_progress(iterator, tqdm, stats, processed, total_rows)

    out = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    if tqdm is None:
        print()

    print("\nBuild summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"Built {len(out)} task rows -> {output_csv}")
    if fetch_fail_examples:
        print("\nSample fetch failures (first 10):")
        for i, sample in enumerate(fetch_fail_examples, start=1):
            print(
                f"  {i}. reason={sample['reason']} "
                f"repo={sample['repo_url']} sha={sample['sha']} "
                f"test={sample['test_name']} detail={sample['detail']}"
            )
    if len(out):
        print(out["category"].value_counts())
        print(out["task_types"].value_counts())


def _update_progress(
    iterator,
    tqdm_mod,
    stats: dict[str, int],
    processed: int | None = None,
    total_rows: int | None = None,
) -> None:
    if tqdm_mod is not None and hasattr(iterator, "set_postfix"):
        iterator.set_postfix(
            kept=stats["kept"],
            miss=stats["skipped_missing_core_fields"],
            ud=stats["skipped_ud"],
            no_task=stats["skipped_no_task_types"],
            fetch_fail=stats["skipped_test_code_fetch_failed"],
        )
        return

    if processed is None or total_rows is None:
        return
    if processed == 1 or processed % 20 == 0 or processed == total_rows:
        print(
            f"\r[{processed}/{total_rows}] "
            f"kept={stats['kept']} "
            f"fetch_fail={stats['skipped_test_code_fetch_failed']} "
            f"no_task={stats['skipped_no_task_types']}",
            end="",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FlakySleuth task dataset")
    parser.add_argument("--input", default="idoft/py-data.csv", help="Path to IDoFT py-data.csv")
    parser.add_argument("--output", default="dataset/py_tasks.csv", help="Output CSV path")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate input schema and print summary, without cloning/fetching.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max input rows to process (useful for quick sanity checks).",
    )
    args = parser.parse_args()

    github_token = os.environ.get("GITHUB_TOKEN", "")
    build(
        args.input,
        args.output,
        github_token,
        validate_only=args.validate_only,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
