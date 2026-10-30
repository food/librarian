"""BundleStore: sandboxed IO on the memory bundle.

Security: all paths are resolved against the bundle root and checked
(no escapes, no symlinks). Writes are atomic (tmp + rename).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from . import okf
from .okf import Concept, RESERVED_FILES

# Top-level sections of the library; inbox = deterministic fallback
SECTIONS = ["people", "projects", "infrastructure", "preferences", "knowledge", "events", "inbox"]


class BundleStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    # ---------- Path sandbox ----------

    def resolve(self, rel_path: str) -> Path:
        """Safely resolve a bundle-relative path; raises on escape/symlink."""
        rel_path = rel_path.lstrip("/").replace("\\", "/")
        p = (self.root / rel_path).resolve()
        if not p.is_relative_to(self.root):
            raise PermissionError(f"Path leaves the bundle: {rel_path}")
        # reject symlinks in the path (including intermediate levels)
        cur = p
        while cur != self.root:
            if cur.is_symlink():
                raise PermissionError(f"Symlink not allowed in bundle: {cur}")
            cur = cur.parent
        return p

    def rel(self, p: Path) -> str:
        return p.relative_to(self.root).as_posix()

    # ---------- Reading ----------

    def exists(self, rel_path: str) -> bool:
        return self.resolve(rel_path).is_file()

    def read_text(self, rel_path: str) -> str:
        return self.resolve(rel_path).read_text(encoding="utf-8")

    def read_concept(self, rel_path: str) -> Concept:
        return okf.parse(rel_path.lstrip("/"), self.read_text(rel_path))

    def list_concept_paths(self) -> list[str]:
        """All concept files (excluding index.md/log.md), bundle-relative, sorted."""
        out: list[str] = []
        for p in sorted(self.root.rglob("*.md")):
            if ".git" in p.parts:
                continue
            if p.name in RESERVED_FILES:
                continue
            out.append(self.rel(p))
        return out

    def iter_concepts(self):
        for path in self.list_concept_paths():
            try:
                yield self.read_concept(path)
            except ValueError:
                # broken file: the linter reports it; iteration continues
                continue

    # ---------- Writing (atomic) ----------

    def write_text(self, rel_path: str, text: str) -> None:
        target = self.resolve(rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            # mkstemp creates 0600 - the memory should be readable by the user
            # (Samba/editor on the host), not only by the container user
            os.chmod(tmp, 0o644)
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def write_concept(self, concept: Concept) -> None:
        self.write_text(concept.path, okf.serialize(concept))

    def delete(self, rel_path: str) -> None:
        p = self.resolve(rel_path)
        if p.is_file():
            p.unlink()

    # ---------- Setup ----------

    def init_layout(self) -> None:
        """Create directories + root index.md/log.md if missing."""
        self.root.mkdir(parents=True, exist_ok=True)
        for section in SECTIONS:
            (self.root / section).mkdir(exist_ok=True)
        if not (self.root / "log.md").exists():
            self.write_text("log.md", "# Log\n")
        if not (self.root / "index.md").exists():
            self.write_text("index.md", "# Index\n")
