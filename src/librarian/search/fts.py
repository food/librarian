"""SQLite FTS5 index over the bundle. Deterministic prefilter for the librarian.

Write-through: mutations update the index in the same code path.
Drift detection: meta.indexed_commit vs. git HEAD -> full reindex.
The DB can be fully rebuilt from the bundle (no backup needed).
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from ..bundle.okf import Concept
from ..bundle.store import BundleStore

SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS concepts USING fts5(
    path UNINDEXED, title, type, tags, description, aliases, body
);
CREATE TABLE IF NOT EXISTS links (
    src TEXT NOT NULL, dst TEXT NOT NULL, PRIMARY KEY (src, dst)
);
CREATE TABLE IF NOT EXISTS hot_cache (
    path TEXT PRIMARY KEY, summary TEXT, added_at TEXT, last_access TEXT, hits INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, client TEXT, op TEXT,
    query TEXT, source TEXT, detail TEXT
);
CREATE TABLE IF NOT EXISTS review_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT, detail TEXT, status TEXT DEFAULT 'open'
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS doc_deps (
    src_name TEXT NOT NULL, src_path TEXT NOT NULL,
    dst_name TEXT NOT NULL, dst_path TEXT,          -- NULL = ghost (undocumented)
    label TEXT DEFAULT '',
    PRIMARY KEY (src_name, dst_name, label)
);
-- All documented services (including those without deps / without references),
-- so deps_graph also shows isolated nodes instead of only those from edges.
CREATE TABLE IF NOT EXISTS doc_services (
    name TEXT PRIMARY KEY, doc_path TEXT NOT NULL
);
"""


@dataclass
class Hit:
    path: str
    title: str
    type: str
    description: str
    score: float


def _fts_query(user_query: str) -> str:
    """Translate user input into a safe FTS5 OR query (no syntax crashes)."""
    words = re.findall(r"\w+", user_query, flags=re.UNICODE)
    if not words:
        return '""'
    return " OR ".join(f'"{w}"' for w in words[:12])


class FtsIndex:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.db.commit()

    # ---------- write-through ----------

    def upsert(self, concept: Concept, link_targets: list[str]) -> None:
        self.remove(concept.path)
        self.db.execute(
            "INSERT INTO concepts (path, title, type, tags, description, aliases, body) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (concept.path, concept.meta.title, concept.meta.type,
             " ".join(concept.meta.tags), concept.meta.description,
             " ".join(concept.meta.aliases), concept.body),
        )
        for dst in link_targets:
            self.db.execute("INSERT OR IGNORE INTO links (src, dst) VALUES (?, ?)",
                            (concept.path, dst))
        self.db.commit()

    def remove(self, path: str) -> None:
        self.db.execute("DELETE FROM concepts WHERE path = ?", (path,))
        self.db.execute("DELETE FROM links WHERE src = ?", (path,))
        self.db.commit()

    # ---------- Search ----------

    def search(self, query: str, limit: int = 8, memory_only: bool = False) -> list[Hit]:
        # Docs share the table with the concepts and vastly outnumber them
        # (currently 1220 : 70). Anyone looking for memory candidates must exclude them
        # BEFORE the LIMIT - filtering afterwards often yields an empty
        # list even though good targets exist.
        doc_filter = " AND path NOT LIKE 'doc:%'" if memory_only else ""
        rows = self.db.execute(
            "SELECT path, title, type, description, bm25(concepts) AS score "
            f"FROM concepts WHERE concepts MATCH ?{doc_filter} ORDER BY score LIMIT ?",
            (_fts_query(query), limit),
        ).fetchall()
        return [Hit(path=r[0], title=r[1], type=r[2], description=r[3], score=r[4])
                for r in rows]

    def count(self) -> int:
        return self.db.execute(
            "SELECT count(*) FROM concepts WHERE path NOT LIKE 'doc:%'").fetchone()[0]

    def docs_count(self) -> int:
        return self.db.execute(
            "SELECT count(*) FROM concepts WHERE path LIKE 'doc:%'").fetchone()[0]

    # ---------- Reindex / Drift ----------

    def get_meta(self, key: str) -> str:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else ""

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
        self.db.commit()

    def reindex(self, store: BundleStore, head: str) -> int:
        """Reindex bundle concepts (docs entries stay untouched)."""
        from ..bundle import links as linkmod
        self.db.execute("DELETE FROM concepts WHERE path NOT LIKE 'doc:%'")
        self.db.execute("DELETE FROM links")
        n = 0
        for concept in store.iter_concepts():
            self.upsert(concept, linkmod.extract_links(concept.body, concept.path))
            n += 1
        # Concepts can also vanish outside the server (merge, cleanup
        # directly in the bundle repo). Orphaned hot-cache rows would otherwise sit
        # in the seed of EVERY session until the TTL and point at nothing.
        self.db.execute("DELETE FROM hot_cache WHERE path NOT IN (SELECT path FROM concepts)")
        self.db.commit()
        self.set_meta("indexed_commit", head)
        return n

    def ensure_fresh(self, store: BundleStore, head: str) -> bool:
        """Reindex if the bundle HEAD differs from the indexed state. True = reindexed."""
        if head and self.get_meta("indexed_commit") == head:
            return False
        self.reindex(store, head)
        return True

    # ---------- Reference shelf (read-only docs, kind via doc: prefix) ----------

    def reindex_docs(self, docs) -> int:
        """Reindex the docs mirror (entries with doc: prefix in the path)."""
        self.db.execute("DELETE FROM concepts WHERE path LIKE 'doc:%'")
        n = 0
        for doc_path in docs.list_paths():
            try:
                body = docs.read_text(doc_path)
            except OSError:
                continue
            self.db.execute(
                "INSERT INTO concepts (path, title, type, tags, description, aliases, body) "
                "VALUES (?, ?, 'doc', '', ?, '', ?)",
                (doc_path, docs.title_of(doc_path),
                 "Reference docs (docs mirror, read-only)", body))
            n += 1
        self.db.commit()
        self._reindex_doc_deps(docs)
        self.set_meta("indexed_docs_fingerprint", docs.fingerprint())
        return n

    def _reindex_doc_deps(self, docs) -> None:
        """Rebuild service dependencies from doc frontmatter (deterministic)."""
        self.db.execute("DELETE FROM doc_deps")
        self.db.execute("DELETE FROM doc_services")
        services = docs.all_services()
        by_name = {s.name: s.doc_path for s in services}
        for s in services:
            self.db.execute(
                "INSERT OR REPLACE INTO doc_services (name, doc_path) VALUES (?, ?)",
                (s.name, s.doc_path))
            for dep in s.deps:
                self.db.execute(
                    "INSERT OR REPLACE INTO doc_deps "
                    "(src_name, src_path, dst_name, dst_path, label) VALUES (?, ?, ?, ?, ?)",
                    (s.name, s.doc_path, dep.dst_name,
                     by_name.get(dep.dst_name), dep.label))
        self.db.commit()

    def deps_graph(self) -> dict:
        """Service dependency graph: nodes (incl. ghosts) + directed edges."""
        nodes: dict[str, dict] = {}
        # Base: ALL documented services - including isolated ones without edges
        for name, doc_path in self.db.execute(
                "SELECT name, doc_path FROM doc_services").fetchall():
            nodes[name] = {"name": name, "doc": doc_path, "ghost": False}
        rows = self.db.execute(
            "SELECT src_name, src_path, dst_name, dst_path, label FROM doc_deps").fetchall()
        for src_name, src_path, dst_name, dst_path, _ in rows:
            nodes.setdefault(src_name, {"name": src_name, "doc": src_path, "ghost": False})
            node = nodes.setdefault(dst_name, {"name": dst_name, "doc": dst_path,
                                               "ghost": dst_path is None})
            if dst_path:  # documented after the fact -> clear ghost status
                node.update(doc=dst_path, ghost=False)
        edges = [{"src": r[0], "dst": r[2], "label": r[4]} for r in rows]
        # "did you mean ...?": ghost whose name (differing only by separators ./-/_)
        # unambiguously matches a documented service - exposes naming
        # inconsistencies in new docs instead of silently turning them into ghosts.
        norm = lambda s: re.sub(r"[-._]", "", s.lower())
        by_norm: dict[str, list[str]] = {}
        for n in nodes.values():
            if not n["ghost"]:
                by_norm.setdefault(norm(n["name"]), []).append(n["name"])
        for n in nodes.values():
            if n["ghost"]:
                cand = by_norm.get(norm(n["name"]))
                if cand and len(cand) == 1:
                    n["suggest"] = cand[0]
        return {"nodes": list(nodes.values()), "edges": edges}

    def ensure_docs_fresh(self, docs) -> bool:
        """Docs reindex when the fingerprint changed. True = reindexed."""
        if not docs.available():
            return False
        if self.get_meta("indexed_docs_fingerprint") == docs.fingerprint():
            return False
        self.reindex_docs(docs)
        return True

    # ---------- Web UI data ----------

    def graph(self) -> dict:
        """Nodes + edges for the graph visualizer (existing targets only)."""
        nodes = [
            {"path": r[0], "title": r[1] or r[0].split("/")[-1].removesuffix(".md"),
             "type": r[2], "section": r[0].split("/")[0], "description": r[3] or ""}
            for r in self.db.execute(
                "SELECT path, title, type, description FROM concepts "
                "WHERE path NOT LIKE 'doc:%'").fetchall()
        ]
        paths = {n["path"] for n in nodes}
        edges = [
            {"src": r[0], "dst": r[1]}
            for r in self.db.execute("SELECT src, dst FROM links").fetchall()
            if r[0] in paths and r[1] in paths
        ]
        degree: dict[str, int] = {}
        for e in edges:
            degree[e["src"]] = degree.get(e["src"], 0) + 1
            degree[e["dst"]] = degree.get(e["dst"], 0) + 1
        for n in nodes:
            n["degree"] = degree.get(n["path"], 0)
        return {"nodes": nodes, "edges": edges}

    def traces_recent(self, limit: int = 50) -> list[dict]:
        rows = self.db.execute(
            "SELECT ts, client, op, query, source, detail FROM traces "
            "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"ts": r[0], "client": r[1], "op": r[2], "query": r[3],
                 "source": r[4], "detail": r[5]} for r in rows]

    def review_set_status(self, review_id: int, status: str) -> bool:
        cur = self.db.execute(
            "UPDATE review_queue SET status = ? WHERE id = ? AND status = 'open'",
            (status, review_id))
        self.db.commit()
        return cur.rowcount > 0

    # ---------- Review-Queue ----------

    def queue_review(self, kind: str, detail: str) -> None:
        """Destructive proposals land here instead of being executed (deduplicated)."""
        row = self.db.execute(
            "SELECT 1 FROM review_queue WHERE kind = ? AND detail = ? AND status = 'open'",
            (kind, detail)).fetchone()
        if not row:
            from datetime import datetime
            self.db.execute(
                "INSERT INTO review_queue (ts, kind, detail) VALUES (?, ?, ?)",
                (datetime.now().astimezone().isoformat(timespec="seconds"), kind, detail))
            self.db.commit()

    def open_reviews(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT id, ts, kind, detail FROM review_queue WHERE status = 'open' "
            "ORDER BY id").fetchall()
        return [{"id": r[0], "ts": r[1], "kind": r[2], "detail": r[3]} for r in rows]

    # ---------- Traces ----------

    def trace(self, ts: str, client: str, op: str, query: str, source: str, detail: str) -> None:
        self.db.execute(
            "INSERT INTO traces (ts, client, op, query, source, detail) VALUES (?, ?, ?, ?, ?, ?)",
            (ts, client, op, query, source, detail))
        self.db.commit()
