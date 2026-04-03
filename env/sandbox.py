from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


class Sandbox:
    def __init__(self, task: dict):
        self.task = task
        self.tmpdir: str | None = None
        self.file_tree: list[str] = []

    def setup(self) -> None:
        """Prepare a working copy of the repository for the episode."""
        self.tmpdir = tempfile.mkdtemp(prefix="flakysleuth_")
        repo_url = str(self.task.get("repo_url", "")).strip()
        sha = str(self.task.get("sha", "")).strip()

        try:
            if repo_url.startswith("fixture://"):
                self._copy_fixture_repo(repo_url)
            else:
                self._clone_repo(repo_url, sha)

            self.file_tree = self._build_file_tree()
        except Exception as exc:
            self.cleanup()
            raise RuntimeError(f"Sandbox setup failed: {exc}") from exc

    def read_file(self, relative_path: str) -> str | None:
        """Read a file relative to sandbox root. Returns None when not found/unsafe."""
        if not self.tmpdir:
            return None

        root = os.path.abspath(self.tmpdir)
        full_path = os.path.abspath(os.path.join(root, relative_path))

        # Path traversal guard.
        if os.path.commonpath([root, full_path]) != root:
            return None
        if not os.path.isfile(full_path):
            return None

        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as handle:
                return handle.read()[:4000]
        except Exception:
            return None

    def grep(self, pattern: str) -> str:
        """Search .py files in repo, preferring ripgrep and falling back to grep."""
        if not self.tmpdir:
            return "ERROR: Sandbox not initialized"

        rg_cmd = ["rg", "-n", "--glob", "*.py", pattern, "."]
        grep_cmd = ["grep", "-RIn", "--include=*.py", pattern, "."]

        try:
            result = subprocess.run(
                rg_cmd,
                cwd=self.tmpdir,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except FileNotFoundError:
            # ripgrep not installed in runtime; fall back to POSIX grep.
            try:
                result = subprocess.run(
                    grep_cmd,
                    cwd=self.tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except FileNotFoundError:
                return (
                    "Search error: neither 'rg' (ripgrep) nor 'grep' is installed in the "
                    "runtime."
                )
            except subprocess.TimeoutExpired:
                return "Search timed out"
            except Exception as exc:
                return f"Search error: {exc}"
        except subprocess.TimeoutExpired:
            return "Search timed out"
        except Exception as exc:
            return f"Search error: {exc}"

        try:
            output = (result.stdout + result.stderr).strip()[:2000]
            if output:
                return output
            return f"No matches found for: {pattern}"
        except Exception as exc:
            return f"Search error: {exc}"

    def run_test(self, pytest_test_name: str) -> str:
        """Run a test for non-order-dependent categories."""
        if not self.tmpdir:
            return "ERROR: Sandbox not initialized"

        category = str(self.task.get("category", "")).strip()
        if category in ("OD", "OD-Brit", "OD-Vic"):
            return (
                "Test execution skipped for order-dependent tests. "
                "Use read_file and search_code for static analysis. "
                "Look for shared state, missing cleanup, or global mutations."
            )

        try:
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "pytest",
                    pytest_test_name,
                    "--tb=short",
                    "-x",
                    "--timeout=30",
                    "-q",
                ],
                cwd=self.tmpdir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = (result.stdout + result.stderr).strip()[:2000]
            return output or "Test completed with no output"
        except subprocess.TimeoutExpired:
            return "Test execution timed out (>60s)"
        except Exception as exc:
            return f"Test execution error: {exc}"

    def cleanup(self) -> None:
        if self.tmpdir and os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir, ignore_errors=True)
        self.tmpdir = None
        self.file_tree = []

    def _clone_repo(self, repo_url: str, sha: str) -> None:
        if not repo_url:
            raise ValueError("Missing repo_url")
        assert self.tmpdir is not None

        sha = (sha or "").strip()
        # Robust path: fetch the exact commit directly (works even when not in shallow branch history).
        if sha and sha.lower() != "nan":
            init = subprocess.run(
                ["git", "init", self.tmpdir],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if init.returncode != 0:
                raise RuntimeError(f"git init failed: {init.stderr.strip()}")

            remote = subprocess.run(
                ["git", "-C", self.tmpdir, "remote", "add", "origin", repo_url],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if remote.returncode != 0:
                raise RuntimeError(f"git remote add failed: {remote.stderr.strip()}")

            fetch = subprocess.run(
                ["git", "-C", self.tmpdir, "fetch", "--depth=1", "origin", sha],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if fetch.returncode != 0:
                raise RuntimeError(
                    "git fetch exact sha failed: "
                    + (fetch.stderr.strip() or fetch.stdout.strip())
                )

            checkout = subprocess.run(
                ["git", "-C", self.tmpdir, "checkout", "--detach", "FETCH_HEAD"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if checkout.returncode != 0:
                raise RuntimeError(
                    "git checkout fetched sha failed: "
                    + (checkout.stderr.strip() or checkout.stdout.strip())
                )
            return

        # Fallback for rows without a SHA.
        clone = subprocess.run(
            ["git", "clone", "--depth=50", repo_url, self.tmpdir],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if clone.returncode != 0:
            raise RuntimeError(
                "git clone failed: " + (clone.stderr.strip() or clone.stdout.strip())
            )

    def _copy_fixture_repo(self, repo_url: str) -> None:
        fixture_name = repo_url.replace("fixture://", "", 1).strip("/")
        if not fixture_name:
            raise ValueError("Fixture name missing in repo_url")

        fixture_dir = (
            Path(__file__).resolve().parent.parent
            / "dataset"
            / "fixtures"
            / fixture_name
        )
        if not fixture_dir.exists():
            raise FileNotFoundError(f"Fixture repo not found: {fixture_dir}")

        assert self.tmpdir is not None
        shutil.copytree(fixture_dir, self.tmpdir, dirs_exist_ok=True)

    def _build_file_tree(self) -> list[str]:
        assert self.tmpdir is not None
        result: list[str] = []
        for root, dirs, files in os.walk(self.tmpdir):
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in ("node_modules", "__pycache__", ".git", "venv", ".tox")
            ]
            depth = root.replace(self.tmpdir, "").count(os.sep)
            if depth <= 2:
                for file_name in files:
                    rel_path = os.path.relpath(os.path.join(root, file_name), self.tmpdir)
                    result.append(rel_path)
            if len(result) > 100:
                break
        return result[:100]
