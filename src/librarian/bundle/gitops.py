"""Git operations on the bundle: auto-commit per mutation, rollback on lint failure."""
from __future__ import annotations

import subprocess
from pathlib import Path


class GitOps:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            capture_output=True, text=True, check=check,
        )

    def ensure_repo(self) -> None:
        if not (self.root / ".git").exists():
            self._git("init", "-b", "main")
            self._git("add", "-A")
            self._git("commit", "-m", "init: empty memory bundle", check=False)

    def head(self) -> str:
        r = self._git("rev-parse", "HEAD", check=False)
        return r.stdout.strip() if r.returncode == 0 else ""

    def is_dirty(self) -> bool:
        return bool(self._git("status", "--porcelain").stdout.strip())

    def commit_all(self, message: str) -> str:
        """Stage and commit everything. Returns the new HEAD ('' if there is nothing to commit)."""
        self._git("add", "-A")
        r = self._git("commit", "-m", message, check=False)
        if r.returncode != 0:
            if "nothing to commit" in (r.stdout + r.stderr):
                return ""
            raise RuntimeError(f"git commit failed: {r.stderr or r.stdout}")
        return self.head()

    def rollback_uncommitted(self) -> None:
        """Hard-reset the working tree to HEAD (safety net after lint failure)."""
        self._git("reset", "--hard", "HEAD")
        self._git("clean", "-fd")

    def tag(self, name: str) -> None:
        self._git("tag", "-f", name)
