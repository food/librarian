"""Hot cache: the most recently stored/used concepts, immediately at hand.

Entries are added on every mutation and every query hit. Eviction: TTL without
access OR LRU beyond max_entries. The query shortcut answers unambiguous
hits straight from the file - no LLM, no GPU (~ms instead of ~s).
"""
from __future__ import annotations

from datetime import datetime, timedelta

from .search.fts import FtsIndex, Hit


def _now() -> str:
    # Microseconds: several bumps within the same second must be LRU-sortable
    return datetime.now().astimezone().isoformat(timespec="microseconds")


class HotCache:
    def __init__(self, fts: FtsIndex, ttl_days: int, max_entries: int):
        self.db = fts.db
        self.ttl_days = ttl_days
        self.max_entries = max_entries

    def bump(self, path: str, summary: str = "") -> None:
        """Register access/creation (upsert + hit++), then evict."""
        now = _now()
        cur = self.db.execute(
            "UPDATE hot_cache SET last_access = ?, hits = hits + 1, "
            "summary = CASE WHEN ? != '' THEN ? ELSE summary END WHERE path = ?",
            (now, summary, summary, path))
        if cur.rowcount == 0:
            self.db.execute(
                "INSERT INTO hot_cache (path, summary, added_at, last_access, hits) "
                "VALUES (?, ?, ?, ?, 1)", (path, summary, now, now))
        self._evict()
        self.db.commit()

    def _evict(self) -> None:
        cutoff = (datetime.now().astimezone()
                  - timedelta(days=self.ttl_days)).isoformat(timespec="seconds")
        self.db.execute("DELETE FROM hot_cache WHERE last_access < ?", (cutoff,))
        # rowid as tie-break: the Windows clock is coarse enough that several bumps
        # get the same timestamp - without a tie-break SQLite evicts arbitrarily
        self.db.execute(
            "DELETE FROM hot_cache WHERE path NOT IN "
            "(SELECT path FROM hot_cache ORDER BY last_access DESC, rowid DESC LIMIT ?)",
            (self.max_entries,))

    def remove(self, path: str) -> None:
        self.db.execute("DELETE FROM hot_cache WHERE path = ?", (path,))
        self.db.commit()

    def paths(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT path FROM hot_cache").fetchall()}

    def recent(self, limit: int = 8) -> list[dict]:
        rows = self.db.execute(
            "SELECT path, summary, last_access, hits FROM hot_cache "
            "ORDER BY last_access DESC, rowid DESC LIMIT ?", (limit,)).fetchall()
        return [{"path": r[0], "summary": r[1], "last_access": r[2], "hits": r[3]}
                for r in rows]

    def shortcut(self, hits: list[Hit]) -> str | None:
        """Unambiguous hot hit? Then return the path (answer without LLM).

        'Unambiguous' = top hit is in the hot cache AND clearly better than
        rank 2 (bm25: lower = better) or the only hit.
        """
        if not hits or hits[0].path not in self.paths():
            return None
        if len(hits) == 1 or hits[0].score < hits[1].score - 1.5:
            return hits[0].path
        return None
