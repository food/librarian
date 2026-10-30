from librarian.bundle.okf import Concept, ConceptMeta


def _concept(path, title, body, **kw):
    return Concept(path=path, meta=ConceptMeta(type="fact", title=title, **kw), body=body)


def test_fts_roundtrip(fts):
    fts.upsert(_concept("knowledge/pv.md", "PV system",
                        "The Enphase Enlighten cloud API provides the daily values.",
                        tags=["solar"]), [])
    fts.upsert(_concept("people/anna.md", "Anna", "Lives in Springfield."), [])
    hits = fts.search("How does the PV data from Enphase arrive?")
    assert hits and hits[0].path == "knowledge/pv.md"
    # a query with FTS special characters must not crash
    assert fts.search('what is "this" AND (here)?') is not None
    fts.remove("knowledge/pv.md")
    assert all(h.path != "knowledge/pv.md" for h in fts.search("Enphase"))


def test_search_memory_only_excludes_docs(fts, tmp_path):
    """Docs outnumber concepts - memory_only must drop them BEFORE the LIMIT."""
    from librarian.bundle.docs import DocsStore
    docs_root = tmp_path / "docs" / "acme"
    docs_root.mkdir(parents=True)
    for i in range(9):
        (docs_root / f"s{i}.md").write_text(
            f"# Service {i}\nThis is something about the coffee machine.\n",
            encoding="utf-8")
    fts.reindex_docs(DocsStore(tmp_path / "docs"))
    fts.upsert(_concept("infrastructure/coffee.md", "Coffee machine",
                        "The coffee machine is in the kitchen."), [])

    mixed = fts.search("coffee machine", 8)
    assert any(h.path.startswith("doc:") for h in mixed)        # docs share the top-N
    only = fts.search("coffee machine", 8, memory_only=True)
    assert [h.path for h in only] == ["infrastructure/coffee.md"]


def test_fts_reindex_and_drift(fts, store, git):
    from librarian.bundle.okf import Concept, ConceptMeta
    store.write_concept(Concept(path="knowledge/a.md",
                                meta=ConceptMeta(type="fact", title="Alpha"),
                                body="Alpha content.\n"))
    git.commit_all("add a")
    n = fts.reindex(store, git.head())
    assert n == 1
    assert not fts.ensure_fresh(store, git.head())      # fresh -> no reindex
    store.write_concept(Concept(path="knowledge/b.md",
                                meta=ConceptMeta(type="fact", title="Beta"),
                                body="Beta content.\n"))
    git.commit_all("add b")
    assert fts.ensure_fresh(store, git.head())          # drift -> reindex
    assert fts.count() == 2
