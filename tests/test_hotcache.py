"""M3: hot cache - bump/evict, query shortcut without LLM, seed hot list."""
import pytest

from librarian.bundle.okf import Concept, ConceptMeta
from librarian.hotcache import HotCache
from librarian.ops import MemoryOps
from librarian.seed import render_seed
from tests.test_ops import FakeLlm


@pytest.fixture
def hot(fts):
    return HotCache(fts, ttl_days=7, max_entries=3)


@pytest.fixture
def ops(settings, store, git, fts, hot):
    return MemoryOps(settings, store, git, fts, FakeLlm(), hot=hot)


def put(store, path, title, body):
    store.write_concept(Concept(path=path, meta=ConceptMeta(type="fact", title=title),
                                body=body))


def test_bump_and_lru_eviction(hot):
    for i in range(5):
        hot.bump(f"knowledge/k{i}.md", f"Fact {i}")
    assert len(hot.paths()) == 3                       # max_entries applies (LRU)
    assert "knowledge/k4.md" in hot.paths()
    assert "knowledge/k0.md" not in hot.paths()
    hot.bump("knowledge/k4.md")
    assert hot.recent(1)[0]["hits"] == 2


def test_query_shortcut_without_llm(ops, store, git, fts):
    put(store, "knowledge/wifi.md", "WiFi", "The WiFi password is in the hallway cabinet safe.\n")
    git.commit_all("seed")
    fts.reindex(store, git.head())
    ops.llm = FakeLlm(fail=True)                       # LLM dead -> shortcut must work without it
    ops.hot.bump("knowledge/wifi.md", "WiFi info")
    r = ops.query("Where is the WiFi password?", client="test")
    assert r.get("via") == "hot_cache"
    assert "hallway cabinet" in r["answer"]
    assert ops.llm.calls == []                         # really NO LLM call


def test_no_shortcut_for_cold_or_ambiguous(ops, store, git, fts):
    put(store, "knowledge/a.md", "Server Alpha", "Server Alpha is in the basement.\n")
    put(store, "knowledge/b.md", "Server Beta", "Server Beta is in the hallway.\n")
    git.commit_all("seed")
    fts.reindex(store, git.head())
    # cold: no shortcut
    assert ops.hot.shortcut(fts.search("Server", 5)) is None
    # both hot + query matches both equally well: no shortcut (ambiguous)
    ops.hot.bump("knowledge/a.md")
    ops.hot.bump("knowledge/b.md")
    assert ops.hot.shortcut(fts.search("Where is the server?", 5)) is None


def test_add_bumps_hot_and_seed_shows_recent(ops, store):
    ops.add("The printer is in the basement.", client="test")
    assert ops.hot.paths()                             # mutation lands in the hot cache
    seed = render_seed(store, ops.hot)
    assert "Recently stored/used:" in seed


def test_stale_hot_entry_removed_on_read_failure(ops, store, git, fts):
    put(store, "knowledge/tmp.md", "Temp", "Short-lived content Xyzzy.\n")
    git.commit_all("seed")
    fts.reindex(store, git.head())
    ops.hot.bump("knowledge/tmp.md")
    store.delete("knowledge/tmp.md")                   # file gone, FTS still stale
    r = ops.query("Xyzzy?", client="test", synthesize=False)
    assert "knowledge/tmp.md" not in ops.hot.paths()   # dead cache entry removed


def test_reindex_prunes_hot_entries_of_deleted_concepts(ops, store, git, fts):
    """If deleted outside the server, no dead entry may remain in the seed."""
    put(store, "knowledge/gone.md", "Gone", "Will be deleted shortly.\n")
    git.commit_all("seed")
    fts.reindex(store, git.head())
    ops.hot.bump("knowledge/gone.md", "Dead entry")
    store.delete("knowledge/gone.md")                  # e.g. git rm in the bundle repo
    git.commit_all("cleanup")
    fts.reindex(store, git.head())
    assert "knowledge/gone.md" not in ops.hot.paths()
    assert "Dead entry" not in render_seed(store, ops.hot)
