import pytest

from librarian.bundle import links as linkmod
from librarian.bundle import okf
from librarian.bundle.indexer import append_log, regenerate_indexes
from librarian.bundle.lint import run_lint
from librarian.bundle.okf import Concept, ConceptMeta


def make(store, path, type_="fact", title="", body="Content.\n", **extra):
    c = Concept(path=path, meta=ConceptMeta(type=type_, title=title, **extra), body=body)
    store.write_concept(c)
    return c


def test_okf_roundtrip(store):
    make(store, "people/anna.md", "person", "Anna", "Lives in Springfield.\n",
         description="Test person", tags=["family"], aliases=["A."])
    c = store.read_concept("people/anna.md")
    assert c.meta.type == "person"
    assert c.meta.title == "Anna"
    assert c.meta.tags == ["family"]
    assert "Springfield" in c.body
    # unknown frontmatter keys survive the roundtrip
    text = store.read_text("people/anna.md").replace("type: person", "type: person\ncustom_key: 42")
    store.write_text("people/anna.md", text)
    c2 = store.read_concept("people/anna.md")
    assert (c2.meta.model_extra or {}).get("custom_key") == 42
    store.write_concept(c2)
    assert "custom_key" in store.read_text("people/anna.md")


def test_slugify():
    # non-ASCII input on purpose: umlauts and sharp s must be transliterated
    assert okf.slugify("Müller & Söhne GmbH") == "muller-sohne-gmbh"
    assert okf.slugify("  Straße 1  ") == "strasse-1"
    assert okf.slugify("!!!") == "unnamed"


def test_store_sandbox(store):
    with pytest.raises(PermissionError):
        store.resolve("../escape.md")
    with pytest.raises(PermissionError):
        store.resolve("/../../etc/passwd")


def test_reserved_files_not_concepts(store):
    make(store, "knowledge/fact.md")
    paths = store.list_concept_paths()
    assert "knowledge/fact.md" in paths
    assert all(not p.endswith(("index.md", "log.md")) for p in paths)


def test_link_extraction():
    body = ("See [Anna](/people/anna.md) and [rel](../projects/x.md) "
            "and [[wiki-target]] and [external](https://example.com).")
    targets = linkmod.extract_links(body, "knowledge/note.md")
    assert "people/anna.md" in targets
    assert "projects/x.md" in targets
    assert "wiki-target.md" in targets
    assert not any("example.com" in t for t in targets)


def test_backlink_idempotent():
    body = "Text.\n"
    once = linkmod.append_backlink(body, "people/anna.md", "Anna")
    twice = linkmod.append_backlink(once, "people/anna.md", "Anna")
    assert once == twice
    assert "## Related" in once


def test_indexes_and_log(store):
    make(store, "people/anna.md", "person", "Anna", description="Test person")
    regenerate_indexes(store)
    root = store.read_text("index.md")
    assert "People" in root and "1 concepts" in root
    section = store.read_text("people/index.md")
    assert "[Anna](/people/anna.md) - Test person" in section
    append_log(store, "first entry")
    append_log(store, "second entry")
    log = store.read_text("log.md")
    assert log.index("second") < log.index("first")  # newest first


def test_lint_broken_link_and_orphan(store):
    make(store, "people/anna.md", "person", "Anna",
         body="Knows [Bob](/people/bob.md).\n")
    report = run_lint(store)
    checks = {(f.check, f.path) for f in report.findings}
    assert ("broken-link", "people/anna.md") in checks
    assert ("orphan", "people/anna.md") in checks
    assert not report.ok  # broken-link is an error


def test_lint_clean_when_linked(store):
    make(store, "people/anna.md", "person", "Anna", body="Knows [Bob](/people/bob.md).\n")
    make(store, "people/bob.md", "person", "Bob", body="Knows [Anna](/people/anna.md).\n")
    report = run_lint(store)
    assert report.ok
    assert not [f for f in report.findings if f.check == "orphan"]
