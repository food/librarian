"""M7: service dependencies from doc frontmatter (deterministic) + deps graph."""
import pytest

from librarian.bundle.docs import DocsStore
from librarian.search.fts import FtsIndex


@pytest.fixture
def svc_docs(tmp_path):
    root = tmp_path / "docs"
    (root / "acme" / "services").mkdir(parents=True)
    # "vorlagen" is the template directory name excluded by src/librarian/bundle/docs.py
    (root / "acme" / "vorlagen").mkdir(parents=True)
    # "art" is the frontmatter key for the dependency type, parsed by DocsStore
    (root / "acme" / "services" / "backend.md").write_text(
        "---\nservice: backend\ndependencies:\n"
        "  - service: postgres\n    art: DB\n"
        "  - service: frontend\n    art: REST\n"
        "  - keycloak\n"                                   # short form without art
        "---\n\n# Backend\n\nContent.\n", encoding="utf-8")
    (root / "acme" / "services" / "frontend.md").write_text(
        "---\nservice: frontend\ndependencies:\n"
        "  - service: backend\n    art: REST\n"
        "---\n\n# Frontend\n\nContent.\n", encoding="utf-8")
    (root / "acme" / "services" / "no-deps.md").write_text(
        "# Prose only\n\nNo frontmatter.\n", encoding="utf-8")
    (root / "acme" / "vorlagen" / "service-template.md").write_text(
        "---\nservice: example-service\ndependencies:\n  - example-database\n---\n"
        "\n# Template\n", encoding="utf-8")
    return DocsStore(root)


def test_service_parsing_and_template_excluded(svc_docs):
    services = {s.name: s for s in svc_docs.all_services()}
    assert set(services) == {"backend", "frontend"}        # template + prose doc excluded
    deps = {(d.dst_name, d.label) for d in services["backend"].deps}
    assert deps == {("postgres", "DB"), ("frontend", "REST"), ("keycloak", "")}


def test_frontmatter_stripped_from_title_and_split(svc_docs):
    assert svc_docs.title_of("doc:acme/services/backend.md") == "Backend"
    meta, body = svc_docs.read_split("doc:acme/services/backend.md")
    assert meta["service"] == "backend"
    assert body.startswith("# Backend")


def test_deps_graph_with_ghosts(tmp_path, svc_docs):
    fts = FtsIndex(tmp_path / "db" / "librarian.db")
    fts.reindex_docs(svc_docs)
    g = fts.deps_graph()
    by_name = {n["name"]: n for n in g["nodes"]}
    assert not by_name["backend"]["ghost"]
    assert not by_name["frontend"]["ghost"]                # documented + referenced
    assert by_name["postgres"]["ghost"]                    # only referenced -> ghost
    assert by_name["keycloak"]["ghost"]
    assert {"src": "backend", "dst": "postgres", "label": "DB"} in g["edges"]
    assert len(g["edges"]) == 4
    # reindex is idempotent (no duplicate edges)
    fts.reindex_docs(svc_docs)
    assert len(fts.deps_graph()["edges"]) == 4


def test_ghost_resolves_after_doc_added(tmp_path, svc_docs):
    fts = FtsIndex(tmp_path / "db" / "librarian.db")
    fts.reindex_docs(svc_docs)
    (svc_docs.root / "acme" / "services" / "postgres.md").write_text(
        "---\nservice: postgres\n---\n\n# Postgres\n", encoding="utf-8")
    fts.reindex_docs(svc_docs)
    g = fts.deps_graph()
    by_name = {n["name"]: n for n in g["nodes"]}
    assert not by_name["postgres"]["ghost"]                # ghost gone
    assert by_name["postgres"]["doc"] == "doc:acme/services/postgres.md"
