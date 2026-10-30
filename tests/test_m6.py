"""M6: reference shelf - read-only docs in FTS, query reads them, write paths stay doc-free."""
import pytest

from librarian.agent.loop import QueryStep, run_query_loop
from librarian.bundle.docs import DocsStore
from librarian.bundle.okf import Concept, ConceptMeta
from librarian.ops import ClassifyResult, MemoryOps
from tests.test_m2 import put
from tests.test_ops import FakeLlm


@pytest.fixture
def docs(tmp_path):
    root = tmp_path / "docs"
    (root / "server" / "gateway").mkdir(parents=True)
    (root / "server" / "gateway" / "litellm.md").write_text(
        "# LiteLLM\n\nCopilot login: docker exec -it litellm python3 relogin.\n",
        encoding="utf-8")
    (root / "memory").mkdir()
    (root / "memory" / "secret.md").write_text("# Auto-Memory\n", encoding="utf-8")
    return DocsStore(root)


@pytest.fixture
def ops(settings, store, git, fts, docs):
    return MemoryOps(settings, store, git, fts, FakeLlm(), docs=docs)


def test_docs_store_lists_and_excludes(docs):
    paths = docs.list_paths()
    assert "doc:server/gateway/litellm.md" in paths
    assert not any("memory" in p for p in paths)        # auto-memory excluded
    assert docs.title_of("doc:server/gateway/litellm.md") == "LiteLLM"
    with pytest.raises(PermissionError):
        docs.resolve("doc:../escape.md")


def test_docs_indexed_and_searchable(fts, store, git, docs):
    git.commit_all("init")
    fts.reindex(store, git.head())
    assert fts.ensure_docs_fresh(docs)
    assert fts.docs_count() == 1
    hits = fts.search("How do I log in to Copilot again?")
    assert any(h.path == "doc:server/gateway/litellm.md" for h in hits)
    # fingerprint unchanged -> no reindex; bundle reindex does NOT delete docs
    assert not fts.ensure_docs_fresh(docs)
    fts.reindex(store, git.head())
    assert fts.docs_count() == 1


def test_graph_excludes_docs(fts, store, git, docs):
    store.write_concept(Concept(path="knowledge/a.md",
                                meta=ConceptMeta(type="fact", title="A"), body="x\n"))
    git.commit_all("seed")
    fts.reindex(store, git.head())
    fts.reindex_docs(docs)
    g = fts.graph()
    assert all(not n["path"].startswith("doc:") for n in g["nodes"])


def test_query_loop_reads_doc_and_cites_it(settings, store, git, fts, docs):
    git.commit_all("init")
    fts.reindex(store, git.head())
    fts.reindex_docs(docs)
    llm = FakeLlm(responses=[
        QueryStep(action="answer",
                  answer="docker exec -it litellm python3 relogin",
                  sources=["doc:server/gateway/litellm.md"]),
    ])
    r = run_query_loop(settings, store, fts, llm, "How to log in to Copilot again?",
                       docs=docs)
    assert "relogin" in r.answer
    assert r.sources == ["doc:server/gateway/litellm.md"]


def test_add_never_targets_docs(ops, store, git, fts, docs):
    git.commit_all("init")
    fts.reindex(store, git.head())
    fts.reindex_docs(docs)
    # LLM hallucinates a doc: as the enrich target -> hard validation -> create
    ops.llm = FakeLlm(responses=[
        ClassifyResult(action="enrich", target_path="doc:server/gateway/litellm.md",
                       section="knowledge", title="X", slug="x", type="fact"),
    ])
    r = ops.add("LiteLLM has a new timeout.", client="test")
    assert r["result"] == "created"
    assert not r["path"].startswith("doc:")
    # doc file untouched
    assert "relogin" in docs.read_text("doc:server/gateway/litellm.md")


def test_update_never_rewrites_docs(ops, store, git, fts, docs):
    """The marker is ALSO in the docs - only the bundle gets rewritten."""
    store.write_concept(Concept(
        path="knowledge/copilot.md",
        meta=ConceptMeta(type="fact", title="Copilot"),
        body="Copilot login works via the relogin script.\n"))
    git.commit_all("seed")
    fts.reindex(store, git.head())
    fts.reindex_docs(docs)
    from librarian.ops import EnrichResult, UpdateTarget
    ops.llm = FakeLlm(responses=[
        UpdateTarget(target_path="knowledge/copilot.md",
                     obsolete_markers=["relogin"],
                     new_fact="Copilot login now works via the OAuth portal."),
        EnrichResult(body="Copilot login now works via the OAuth portal.\n"),
    ])
    r = ops.update("Copilot login is now the OAuth portal instead of relogin.", client="test")
    assert r["result"] == "updated"
    assert r["rewritten"] == ["knowledge/copilot.md"]   # ONLY the bundle
    assert "relogin" in docs.read_text("doc:server/gateway/litellm.md")  # docs untouched


def test_maintain_orphan_candidates_exclude_docs(ops, store, git, fts, docs):
    """Doc hits must never be offered to the LLM as wiring candidates."""
    put(store, "projects/litellm-tool.md", "LiteLLM Tool",
        "Uses Copilot relogin via LiteLLM.\n", type_="project")   # orphan, matches the docs
    put(store, "infrastructure/gateway.md", "Gateway",
        "LiteLLM gateway relogin infrastructure.\n")               # orphan, real concept
    git.commit_all("seed")
    fts.reindex(store, git.head())
    fts.reindex_docs(docs)
    seen_candidates = []

    class SpyLlm(FakeLlm):
        def chat_json(self, system, user, schema):
            if schema.__name__ == "RelatedPaths":
                seen_candidates.append(user)
            return super().chat_json(system, user, schema)

    ops.llm = SpyLlm()
    ops.maintain(client="test")
    assert seen_candidates, "orphan wiring was not attempted at all"
    assert all("doc:" not in u for u in seen_candidates)
