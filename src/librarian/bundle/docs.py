"""Reference shelf: read-only docs mirror next to the memory bundle.

The librarian may READ these files (FTS hits, query loop) but never write
them - they are not memory, they are reference docs. The source of truth
stays the docs repo; the mirror is refreshed on deploy/backup.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter

DOC_PREFIX = "doc:"
# Claude Code auto-memory and secrets do not belong in the index
EXCLUDED_DIRS = {"memory", "secrets", ".git"}
# Templates declare example dependencies - don't count them as real services
DEP_EXCLUDED_DIRS = {"vorlagen"}


@dataclass
class DocDep:
    dst_name: str
    label: str = ""


@dataclass
class DocService:
    doc_path: str            # doc:team/services/x.md
    name: str                # short name (frontmatter service: or file stem)
    deps: list[DocDep] = field(default_factory=list)


def is_doc(path: str) -> bool:
    return path.startswith(DOC_PREFIX)


class DocsStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def available(self) -> bool:
        return self.root.is_dir()

    def list_paths(self) -> list[str]:
        """All markdown files, bundle-style relative paths, with doc: prefix."""
        if not self.available():
            return []
        out = []
        for p in sorted(self.root.rglob("*.md")):
            rel = p.relative_to(self.root)
            if any(part in EXCLUDED_DIRS for part in rel.parts):
                continue
            out.append(DOC_PREFIX + rel.as_posix())
        return out

    def resolve(self, doc_path: str) -> Path:
        rel = doc_path.removeprefix(DOC_PREFIX).lstrip("/")
        p = (self.root / rel).resolve()
        if not p.is_relative_to(self.root.resolve()):
            raise PermissionError(f"Path leaves the reference shelf: {doc_path}")
        return p

    def exists(self, doc_path: str) -> bool:
        try:
            return self.resolve(doc_path).is_file()
        except PermissionError:
            return False

    def read_text(self, doc_path: str) -> str:
        return self.resolve(doc_path).read_text(encoding="utf-8", errors="replace")

    def read_split(self, doc_path: str):
        """(frontmatter dict, body without frontmatter). No frontmatter: ({}, full text)."""
        try:
            post = frontmatter.loads(self.read_text(doc_path))
            return dict(post.metadata or {}), post.content
        except Exception:  # noqa: BLE001 - broken YAML: treat as full text
            return {}, self.read_text(doc_path)

    def title_of(self, doc_path: str) -> str:
        """Frontmatter title > first '# ' heading > file name."""
        try:
            meta, body = self.read_split(doc_path)
            if meta.get("title"):
                return str(meta["title"])
            for line in body.splitlines()[:10]:
                if line.startswith("# "):
                    return line[2:].strip()
        except OSError:
            pass
        return doc_path.removeprefix(DOC_PREFIX).rsplit("/", 1)[-1]

    # ---------- Service dependencies (deterministic, from frontmatter) ----------

    def service_of(self, doc_path: str) -> DocService | None:
        """Service declaration of a doc (None if there is no service: frontmatter)."""
        rel = doc_path.removeprefix(DOC_PREFIX)
        if any(part in DEP_EXCLUDED_DIRS for part in rel.split("/")):
            return None
        meta, _ = self.read_split(doc_path)
        if not meta.get("service"):
            return None
        deps: list[DocDep] = []
        for raw in meta.get("dependencies") or []:
            if isinstance(raw, str):
                deps.append(DocDep(dst_name=raw.strip()))
            elif isinstance(raw, dict) and raw.get("service"):
                deps.append(DocDep(dst_name=str(raw["service"]).strip(),
                                   label=str(raw.get("art", "")).strip()))
        return DocService(doc_path=doc_path, name=str(meta["service"]).strip(), deps=deps)

    def all_services(self) -> list[DocService]:
        return [s for p in self.list_paths() if (s := self.service_of(p))]

    def fingerprint(self) -> str:
        """Change fingerprint (path+size+mtime) for the drift check."""
        h = hashlib.sha256()
        for doc_path in self.list_paths():
            p = self.resolve(doc_path)
            stat = p.stat()
            h.update(f"{doc_path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        return h.hexdigest()[:16]
