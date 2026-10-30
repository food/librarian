"""M4: dreaming (dry-run/apply, inbox filing, budget) + merge."""
import pytest

from librarian import dream
from librarian.bundle.lint import run_lint
from librarian.bundle.okf import Concept, ConceptMeta
from librarian.ops import ClassifyResult, EnrichPatch, EnrichResult, MemoryOps, RelatedPaths
from tests.test_ops import FakeLlm


def put(store, path, title, body, type_="fact", **kw):
    store.write_concept(Concept(path=path,
                                meta=ConceptMeta(type=type_, title=title, **kw),
                                body=body))


@pytest.fixture
def ops(settings, store, git, fts):
    return MemoryOps(settings, store, git, fts, FakeLlm())


# ---------- dream ----------

def test_file_inbox_item_enrich(ops, store, git, fts):
    put(store, "people/anna.md", "Anna", "Lives in Springfield.\n", type_="person")
    put(store, "inbox/2026-07-16-anna-dog.md", "Anna has a dog",
        "Anna has a dog named Rex.\n")
    git.commit_all("seed")
    fts.reindex(store, git.head())
    ops.llm = FakeLlm(responses=[
        ClassifyResult(action="enrich", target_path="people/anna.md"),
        EnrichPatch(index=-1, paragraph="Has a dog named Rex."),
    ])
    result = dream.file_inbox_item(ops, "inbox/2026-07-16-anna-dog.md")
    assert "enriched" in result
    assert not store.exists("inbox/2026-07-16-anna-dog.md")   # inbox cleared
    assert "Rex" in store.read_concept("people/anna.md").body
    # deletion + enrich in the same commit
    assert not git.is_dirty()


def test_file_inbox_item_skips_when_llm_down(ops, store, git, fts):
    put(store, "inbox/2026-07-16-x.md", "X", "Fact X.\n")
    git.commit_all("seed")
    fts.reindex(store, git.head())
    ops.llm = FakeLlm(fail=True)
    assert dream.file_inbox_item(ops, "inbox/2026-07-16-x.md") is None
    assert store.exists("inbox/2026-07-16-x.md")               # stays in place


def test_dream_dry_run_changes_nothing(settings, store, git, fts, monkeypatch):
    put(store, "inbox/2026-07-16-y.md", "Y", "Fact Y.\n")
    put(store, "knowledge/solo.md", "Solo", "Orphaned content.\n")
    git.commit_all("seed")
    fts.reindex(store, git.head())
    head_before = git.head()
    monkeypatch.setattr(dream, "build_ops",
                        lambda s: MemoryOps(settings, store, git, fts, FakeLlm()))
    monkeypatch.setattr(dream, "get_settings", lambda *a, **k: settings)
    result = dream.run(apply=False)
    assert result["mode"] == "dry-run"
    assert any("inbox" in w for w in result["would_do"])
    assert any("orphan" in w for w in result["would_do"])
    assert git.head() == head_before                            # nothing committed


def test_dream_apply_files_inbox_and_respects_budget(settings, store, git, fts,
                                                     monkeypatch):
    for i in range(3):
        put(store, f"inbox/2026-07-16-f{i}.md", f"F{i}", f"Fact number {i}.\n")
    git.commit_all("seed")
    fts.reindex(store, git.head())
    llm = FakeLlm()  # default: classify -> create, orphan -> no related
    monkeypatch.setattr(dream, "build_ops",
                        lambda s: MemoryOps(settings, store, git, fts, llm))
    monkeypatch.setattr(dream, "get_settings", lambda *a, **k: settings)
    monkeypatch.setenv("DREAM_MAX_OPS", "2")
    result = dream.run(apply=True)
    assert result["mode"] == "apply"
    filed = [a for a in result["actions"] if a.startswith("inbox ")]
    assert len(filed) == 2                                      # budget applies
    assert any("Budget exhausted" in a for a in result["actions"])
    remaining = [p for p in store.list_concept_paths() if p.startswith("inbox/")]
    assert len(remaining) == 1


# ---------- merge ----------

def test_merge_moves_content_redirects_links_and_deletes_source(ops, store, git, fts):
    put(store, "people/anna.md", "Anna", "Original. Likes coffee.\n", type_="person")
    put(store, "people/anna-2.md", "Anna S.", "Duplicate. Lives in Springfield.\n",
        type_="person", aliases=["Anna Smith"])
    put(store, "projects/garden.md", "Garden",
        "Helps: [Anna S.](/people/anna-2.md).\n", type_="project")
    git.commit_all("seed")
    fts.reindex(store, git.head())
    ops.llm = FakeLlm(responses=[
        EnrichResult(body="Original. Likes coffee. Lives in Springfield.\n"),
    ])
    r = ops.merge("people/anna-2.md", "people/anna.md", client="test")
    assert r["result"] == "merged"
    assert not store.exists("people/anna-2.md")
    anna = store.read_concept("people/anna.md")
    assert "Springfield" in anna.body
    assert "Anna S." in anna.meta.aliases and "Anna Smith" in anna.meta.aliases
    # reference in the garden project was redirected
    assert "/people/anna.md" in store.read_concept("projects/garden.md").body
    assert "/people/anna-2.md" not in store.read_concept("projects/garden.md").body
    assert run_lint(store).ok


def test_merge_fallback_appends_when_llm_down(ops, store, git, fts):
    put(store, "knowledge/a.md", "A", "Content A with many details.\n")
    put(store, "knowledge/a-2.md", "A dup", "Extra info Xyzzy.\n")
    git.commit_all("seed")
    fts.reindex(store, git.head())
    ops.llm = FakeLlm(fail=True)
    r = ops.merge("knowledge/a-2.md", "knowledge/a.md", client="test")
    body = store.read_concept("knowledge/a.md").body
    assert "Xyzzy" in body and "Merged from" in body         # nothing lost


def test_merge_rejects_missing_paths(ops):
    with pytest.raises(ValueError):
        ops.merge("people/nothing.md", "people/also-nothing.md", client="test")
