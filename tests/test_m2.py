"""M2: full linter + fix, query loop, update (contradiction), maintain."""
import pytest

from librarian.agent.llm import LlmUnavailable
from librarian.agent.loop import QueryStep, run_query_loop
from librarian.bundle.lint import fix_deterministic, run_lint
from librarian.bundle.okf import Concept, ConceptMeta
from librarian.ops import (ClassifyResult, EnrichResult, MemoryOps, RelatedPaths,
                           UpdateTarget)
from tests.test_ops import FakeLlm


def put(store, path, title, body, type_="fact", **kw):
    store.write_concept(Concept(path=path,
                                meta=ConceptMeta(type=type_, title=title, **kw),
                                body=body))


@pytest.fixture
def ops(settings, store, git, fts):
    return MemoryOps(settings, store, git, fts, FakeLlm())


# ---------- full linter ----------

def test_lint_fixture_bundle_with_intentional_errors(store):
    """Fixture with deliberate errors: every check fires."""
    put(store, "people/anna.md", "Anna",
        "Knows [Bob](/people/missing.md).\n")                   # broken-link + orphan
    put(store, "people/carl.md", "Carl",
        "Knows [Anna](/people/anna.md).\n")                     # anna asym (no backlink)
    put(store, "people/anna-2.md", "Anna", "x\n")               # duplicate title + stub
    store.write_text("knowledge/index.md",
                     "# Knowledge\n\n- [Ghost](/knowledge/ghost.md)\n")  # index-ghost
    store.write_text("log.md", "no header\n")                   # log-format

    checks = {f.check for f in run_lint(store).findings}
    assert {"broken-link", "orphan", "backlink-asym", "duplicate",
            "stub", "index-ghost", "log-format"} <= checks


def test_lint_fix_deterministic_and_idempotent(store):
    put(store, "people/anna.md", "Anna", "Standalone text.\n")
    put(store, "people/carl.md", "Carl", "Knows [Anna](/people/anna.md).\n")
    # broken-link that is resolvable (a unique basename exists):
    put(store, "projects/house.md", "House",
        "Belongs to [Anna](/people/anna-old.md).\n")  # not resolvable -> stays an error
    put(store, "knowledge/note.md", "Note",
        "See [Carl](/persons/carl.md).\n")            # wrong directory -> resolvable

    report = run_lint(store)
    fixes = fix_deterministic(store, report)
    assert any("carl" in f for f in fixes)
    # asymmetric backlink Anna->Carl was added
    assert "/people/carl.md" in store.read_concept("people/anna.md").body
    # resolvable broken-link repaired
    assert "/people/carl.md" in store.read_concept("knowledge/note.md").body
    # 2nd run: no new deterministic fixes (idempotent)
    fixes2 = fix_deterministic(store, run_lint(store))
    assert not [f for f in fixes2 if "backlink" in f or "broken-link" in f]
    # unresolvable broken-link remains as an error
    assert any(f.check == "broken-link" for f in run_lint(store).errors)


# ---------- query loop ----------

def test_query_loop_answers_with_sources(settings, store, git, fts):
    put(store, "infrastructure/server.md", "Server", "The server has the IP 10.0.0.5.\n")
    git.commit_all("seed")
    fts.reindex(store, git.head())
    llm = FakeLlm(responses=[
        QueryStep(action="answer", answer="The IP is 10.0.0.5.",
                  sources=["infrastructure/server.md"]),
    ])
    r = run_query_loop(settings, store, fts, llm, "What IP does the server have?")
    assert "10.0.0.5" in r.answer
    assert r.sources == ["infrastructure/server.md"]
    assert r.trace[0]["step"] == "fts"


def test_query_loop_reads_linked_file_then_answers(settings, store, git, fts):
    # plan.md does NOT match the question (no common word) -> it is not
    # pre-read, the LLM has to follow the link from construction.md
    put(store, "projects/construction.md", "Construction",
        "See [Plan](/knowledge/plan.md) for details.\n")
    put(store, "knowledge/plan.md", "Plan", "Total: 50000 euros.\n")
    git.commit_all("seed")
    fts.reindex(store, git.head())
    llm = FakeLlm(responses=[
        QueryStep(action="read", paths=["knowledge/plan.md"]),
        QueryStep(action="answer", answer="The budget is 50000 euros.",
                  sources=["knowledge/plan.md"]),
    ])
    r = run_query_loop(settings, store, fts, llm, "What is the budget for the construction?")
    assert "50000" in r.answer


def test_query_loop_budget_exhausted_raises(settings, store, git, fts):
    put(store, "knowledge/a.md", "A", "Content A.\n")
    git.commit_all("seed")
    fts.reindex(store, git.head())
    llm = FakeLlm(responses=[QueryStep(action="read", paths=["knowledge/nothing.md"])] * 4)
    with pytest.raises(LlmUnavailable):
        run_query_loop(settings, store, fts, llm, "A?")


def test_query_falls_back_to_fts(ops, store, git, fts):
    put(store, "knowledge/wifi.md", "WiFi", "The WiFi password is in the safe.\n")
    git.commit_all("seed")
    fts.reindex(store, git.head())
    ops.llm = FakeLlm(fail=True)
    r = ops.query("Where is the WiFi password?", client="test")
    assert r["matches"] and "safe" in r["matches"][0]["content"]
    assert "FTS-Fallback" in r["note"]


# ---------- update: contradiction pipeline ----------

def test_update_removes_old_fact_everywhere(ops, store, git, fts):
    """Grep proof: the stale-fact marker disappears from ALL files."""
    put(store, "people/alex.md", "Alex", "Lives in Hyderabad.\n", type_="person")
    put(store, "projects/move.md", "Move",
        "Packing boxes for Hyderabad. Concerns [Alex](/people/alex.md).\n",
        type_="project")
    git.commit_all("seed")
    fts.reindex(store, git.head())
    ops.llm = FakeLlm(responses=[
        UpdateTarget(target_path="people/alex.md", obsolete_markers=["Hyderabad"],
                     new_fact="Alex lives in Bangalore."),
        EnrichResult(body="Lives in Bangalore.\n"),
        EnrichResult(body="Packing boxes for Bangalore. Concerns [Alex](/people/alex.md).\n"),
    ])
    r = ops.update("Alex moved from Hyderabad to Bangalore.", client="test")
    assert r["result"] == "updated"
    assert set(r["rewritten"]) == {"people/alex.md", "projects/move.md"}
    for concept in store.iter_concepts():
        assert "hyderabad" not in concept.body.lower()          # the grep proof
    assert "Bangalore" in store.read_concept("people/alex.md").body


def test_update_rolls_back_if_marker_survives(ops, store, git, fts):
    put(store, "people/alex.md", "Alex", "Lives in Hyderabad.\n", type_="person")
    git.commit_all("seed")
    fts.reindex(store, git.head())
    ops.llm = FakeLlm(responses=[
        UpdateTarget(target_path="people/alex.md", obsolete_markers=["Hyderabad"],
                     new_fact="Alex lives in Bangalore."),
        EnrichResult(body="Lives in Bangalore (formerly Hyderabad).\n"),  # marker survives!
    ])
    r = ops.update("Alex now lives in Bangalore.", client="test")
    # original untouched, correction saved as inbox fallback
    assert "Hyderabad" in store.read_concept("people/alex.md").body
    assert r["path"].startswith("inbox/")


def test_update_rewrites_only_the_paragraphs_with_a_marker(ops, store, git, fts):
    """Only affected paragraphs go to the model, the rest stays byte-identical.
    A 25 KB body as a single JSON string came back broken from the model."""
    intro = "Project house.\nSecond line stays."
    tail = "## Notes\n\n- Point A\n- Point B"
    put(store, "projects/house.md", "House",
        f"{intro}\n\nThe server is in Hyderabad.\n\n{tail}\n\nBackup is in Hyderabad.\n",
        type_="project")
    git.commit_all("seed")
    fts.reindex(store, git.head())
    ops.llm = FakeLlm(responses=[
        UpdateTarget(target_path="projects/house.md", obsolete_markers=["Hyderabad"],
                     new_fact="Server and backup are now in Bangalore."),
        EnrichResult(body="The server is in Bangalore."),
        EnrichResult(body="Backup is in Bangalore.\n"),
    ])
    r = ops.update("Server and backup moved from Hyderabad to Bangalore.", client="test")
    assert r["result"] == "updated"
    assert store.read_concept("projects/house.md").body == (
        f"{intro}\n\nThe server is in Bangalore.\n\n{tail}\n\nBackup is in Bangalore.\n")
    sent = [user for schema, user in ops.llm.calls if schema == "EnrichResult"]
    assert len(sent) == 2
    # "Paragraph from" is the prompt label produced by src/librarian/agent/prompts.py
    excerpts = [user.split("Paragraph from projects/house.md:\n")[1] for user in sent]
    assert excerpts == ["---\nThe server is in Hyderabad.\n---\n",
                        "---\nBackup is in Hyderabad.\n\n---\n"]


def test_update_rejects_a_paragraph_that_shrinks_below_half(ops, store, git, fts):
    """A rewrite may replace the stale fact, but not swallow half the paragraph."""
    long = ("The server is in Hyderabad. It has been running since 2019, is backed up "
            "every night at 03:15 and synced by timer every 30 minutes.\n")
    put(store, "infrastructure/server.md", "Server", long, type_="infrastructure")
    git.commit_all("seed")
    fts.reindex(store, git.head())
    ops.llm = FakeLlm(responses=[
        UpdateTarget(target_path="infrastructure/server.md", obsolete_markers=["Hyderabad"],
                     new_fact="The server is in Bangalore."),
        EnrichResult(body="The server is in Bangalore."),   # rest swallowed
    ])
    r = ops.update("The server moved from Hyderabad to Bangalore.", client="test")
    assert r["path"].startswith("inbox/")                        # fallback instead of data loss
    assert store.read_concept("infrastructure/server.md").body == long


def test_update_without_matches_becomes_add(ops, store, git, fts):
    ops.llm = FakeLlm(responses=[
        UpdateTarget(target_path="", obsolete_markers=["Xyzzy"],
                     new_fact="The router is now called gateway2."),
        ClassifyResult(action="create", section="infrastructure", title="Router",
                       slug="router", type="infrastructure"),
    ])
    r = ops.update("The router is now called gateway2.", client="test")
    assert r["result"] == "created"


# ---------- maintain ----------

def test_maintain_fixes_and_wires_orphan(ops, store, git, fts):
    put(store, "people/anna.md", "Anna", "Likes gardens.\n", type_="person")
    put(store, "projects/garden.md", "Garden project",
        "A garden full of flowers. [Anna](/people/anna.md) helps.\n", type_="project")
    git.commit_all("seed")
    fts.reindex(store, git.head())
    # anna: incoming link present but no backlink (asym); garden: orphan
    ops.llm = FakeLlm(responses=[
        RelatedPaths(related_paths=["people/anna.md"]),  # orphan garden -> anna
    ])
    r = ops.maintain(client="test")
    assert any("backlink" in a or "orphan" in a for a in r["actions"])
    # after maintain: no orphan/asym findings left (except inbox)
    left = [f for f in run_lint(store).findings if f.check in ("orphan", "backlink-asym")]
    assert not left


def test_maintain_queues_duplicates(ops, store, git, fts):
    put(store, "people/anna.md", "Anna", "Original [x](/people/anna-2.md).\n", type_="person")
    put(store, "people/anna-2.md", "Anna", "Duplicate [x](/people/anna.md).\n", type_="person")
    git.commit_all("seed")
    fts.reindex(store, git.head())
    r = ops.maintain(client="test", wire_orphans=False)
    assert r["review_open"]
    assert r["review_open"][0]["kind"] == "duplicate"
    # second run deduplicates the queue
    r2 = ops.maintain(client="test", wire_orphans=False)
    assert len(r2["review_open"]) == len(r["review_open"])


def test_update_marker_prefix_of_new_term_newsletter(ops, store, git, fts):
    """Regression: stale marker 'NEWS' is a prefix of the new 'NEWSLETTER' - the
    rewrite must NOT be rejected as 'marker survives' and dumped into the inbox
    (real digest-newsletter case)."""
    put(store, "infrastructure/digest.md", "Digest",
        "The digest newsletters are in the NEWS block of the daily report.\n",
        type_="infrastructure")
    git.commit_all("seed")
    fts.reindex(store, git.head())
    ops.llm = FakeLlm(responses=[
        UpdateTarget(target_path="infrastructure/digest.md",
                     obsolete_markers=["NEWS"],
                     new_fact="Digest newsletters are now in the NEWSLETTER block."),
        EnrichResult(body="The digest newsletters are in the NEWSLETTER block of the daily report.\n"),
    ])
    r = ops.update("Digest newsletters are now in the NEWSLETTER block instead of the NEWS block.",
                   client="test")
    assert r["result"] == "updated"
    assert "NEWSLETTER block" in store.read_concept("infrastructure/digest.md").body


def test_update_marker_prefix_of_new_term_singular_plural(ops, store, git, fts):
    """Regression: stale marker 'upload_id' is a prefix of 'upload_ids' - no
    false-positive 'marker survives' (real marketplace-analysis case)."""
    put(store, "projects/marketplace.md", "Marketplace",
        "Analysis via POST /api/analyze with upload_id (one photo).\n",
        type_="project")
    git.commit_all("seed")
    fts.reindex(store, git.head())
    ops.llm = FakeLlm(responses=[
        UpdateTarget(target_path="projects/marketplace.md",
                     obsolete_markers=["upload_id"],
                     new_fact="Analysis now accepts an upload_ids list (several photos)."),
        EnrichResult(body="Analysis via POST /api/analyze with upload_ids list (several photos).\n"),
    ])
    r = ops.update("Analysis now takes an upload_ids list instead of upload_id.", client="test")
    assert r["result"] == "updated"
    assert "upload_ids" in store.read_concept("projects/marketplace.md").body


def test_update_matches_markers_across_path_separators(ops, store, git, fts):
    r"""C:/path and C:\path must find each other as markers."""
    put(store, "projects/app.md", "App",
        "Code lives in C:\\Users\\user\\Documents\\offers in the repo.\n",
        type_="project")
    git.commit_all("seed")
    fts.reindex(store, git.head())
    ops.llm = FakeLlm(responses=[
        UpdateTarget(target_path="projects/app.md",
                     obsolete_markers=["C:/Users/user/Documents/offers"],
                     new_fact="Code lives in C:/Users/user/Documents/projects/offers."),
        EnrichResult(body="Code lives in C:/Users/user/Documents/projects/offers in the repo.\n"),
    ])
    r = ops.update("The repo moved to Documents/projects/offers.", client="test")
    assert r["result"] == "updated"
    assert "projects/offers" in store.read_concept("projects/app.md").body
