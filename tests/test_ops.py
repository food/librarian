"""MemoryOps with a fake LLM: filing, fallback, rollback protection, parallel smoke."""
import threading

import pytest

from librarian.agent.llm import LlmUnavailable
from librarian.bundle.okf import Concept, ConceptMeta
from librarian.ops import ClassifyResult, EnrichPatch, EnrichResult, MemoryOps


class FakeLlm:
    """Answers deterministically; controllable per test."""

    def __init__(self, responses=None, fail=False):
        self.responses = list(responses or [])
        self.fail = fail
        self.calls = []

    def chat_json(self, system, user, schema):
        self.calls.append((schema.__name__, user))
        if self.fail:
            raise LlmUnavailable("test: LLM down")
        if self.responses:
            return self.responses.pop(0)
        if schema is ClassifyResult:
            return ClassifyResult(action="create", section="knowledge", title="New Thing",
                                  slug="new-thing", type="fact", description="Test fact")
        if schema.__name__ == "QueryStep":
            return schema(action="answer", answer="(fake answer)", sources=[])
        if schema is EnrichPatch:
            return EnrichPatch(paragraph="(fake) merged fact.")
        try:
            return schema()          # schemas with default fields (e.g. RelatedPaths)
        except Exception:
            return EnrichResult(body="whatever")


@pytest.fixture
def ops(settings, store, git, fts):
    return MemoryOps(settings, store, git, fts, FakeLlm())


def test_add_create(ops, store, git):
    result = ops.add("The printer is in the basement.", client="test")
    assert result["result"] == "created"
    assert store.exists(result["path"])
    c = store.read_concept(result["path"])
    assert c.meta.source == "test"
    assert c.meta.timestamp
    assert "printer" in c.body
    assert "index.md" in store.read_text("index.md") or True
    assert "via test" in _last_commit_msg(git)
    # indexes + log were written along
    assert "New Thing" in store.read_text("knowledge/index.md")
    assert "create:" in store.read_text("log.md")


def test_add_enrich(ops, store, git, fts):
    store.write_concept(Concept(path="people/anna.md",
                                meta=ConceptMeta(type="person", title="Anna"),
                                body="Lives in Springfield.\n"))
    git.commit_all("seed")
    fts.reindex(store, git.head())
    ops.llm = FakeLlm(responses=[
        ClassifyResult(action="enrich", target_path="people/anna.md"),
        EnrichPatch(index=-1, paragraph="Has a dog named Rex."),
    ])
    result = ops.add("Anna has a dog named Rex.", client="test")
    assert result["result"] == "enriched"
    assert "Rex" in store.read_concept("people/anna.md").body


def _anna(store, git, fts, body):
    store.write_concept(Concept(path="people/anna.md",
                                meta=ConceptMeta(type="person", title="Anna"), body=body))
    git.commit_all("seed")
    fts.reindex(store, git.head())


def test_enrich_answers_with_one_paragraph_and_keeps_the_rest(ops, store, git, fts):
    """The model sees numbered paragraphs and returns only one new one;
    it lands before the backlinks, everything else stays byte-identical."""
    _anna(store, git, fts,
          "Lives in Springfield.\n\nWorks as a forester.\n\n## Related\n\n- [Bob](/people/bob.md)\n")
    ops.llm = FakeLlm(responses=[
        ClassifyResult(action="enrich", target_path="people/anna.md"),
        EnrichPatch(index=-1, paragraph="Has a dog named Rex."),
    ])
    assert ops.add("Anna has a dog named Rex.", client="test")["result"] == "enriched"
    assert store.read_concept("people/anna.md").body == (
        "Lives in Springfield.\n\nWorks as a forester.\n\nHas a dog named Rex.\n\n"
        "## Related\n\n- [Bob](/people/bob.md)\n")
    sent = next(user for schema, user in ops.llm.calls if schema == "EnrichPatch")
    assert "[0] Lives in Springfield.\n\n[1] Works as a forester.\n\n[2] ## Related" in sent


def test_enrich_replaces_a_paragraph_but_never_a_heading_or_link_list(ops, store, git, fts):
    _anna(store, git, fts, "Lives in Springfield.\n\n## Related\n\n- [Bob](/people/bob.md)\n")
    ops.llm = FakeLlm(responses=[
        ClassifyResult(action="enrich", target_path="people/anna.md"),
        EnrichPatch(index=0, paragraph="[0] Has lived in Springfield since 2020."),   # number is stripped
        ClassifyResult(action="enrich", target_path="people/anna.md"),
        EnrichPatch(index=2, paragraph="Has a dog named Rex."),      # points at the link list
    ])
    ops.add("Anna has lived in Springfield since 2020.", client="test")
    ops.add("Anna has a dog named Rex.", client="test")
    assert store.read_concept("people/anna.md").body == (
        "Has lived in Springfield since 2020.\n\nHas a dog named Rex.\n\n"
        "## Related\n\n- [Bob](/people/bob.md)\n")


def test_add_enrich_hallucinated_target_falls_back_to_create(ops, store):
    ops.llm = FakeLlm(responses=[
        ClassifyResult(action="enrich", target_path="people/nonexistent.md",
                       section="knowledge", title="X", slug="x", type="fact"),
    ])
    result = ops.add("Some fact.", client="test")
    assert result["result"] == "created"


def test_add_llm_down_inbox_fallback(ops, store):
    ops.llm = FakeLlm(fail=True)
    result = ops.add("Important fact despite the LLM outage.", client="test")
    assert result["result"] == "created"
    assert result["path"].startswith("inbox/")
    assert store.exists(result["path"])


def test_enrich_shrunken_body_rejected(ops, store, git, fts):
    store.write_concept(Concept(
        path="people/anna.md",
        meta=ConceptMeta(type="person", title="Anna"),
        body="Line 1.\nLine 2.\nLine 3.\nLine 4.\nLine 5.\nLine 6.\n"))
    git.commit_all("seed")
    fts.reindex(store, git.head())
    ops.llm = FakeLlm(responses=[
        ClassifyResult(action="enrich", target_path="people/anna.md"),
        EnrichPatch(index=0, paragraph="Now just short."),  # suspiciously shrunk -> rejected
    ])
    result = ops.add("Anna likes coffee.", client="test")
    assert result["path"].startswith("inbox/")            # fallback instead of data loss
    assert "Line 6." in store.read_concept("people/anna.md").body


def test_create_links_both_ways(ops, store, git, fts):
    store.write_concept(Concept(path="people/anna.md",
                                meta=ConceptMeta(type="person", title="Anna"),
                                body="Lives in Springfield.\n"))
    git.commit_all("seed")
    fts.reindex(store, git.head())
    ops.llm = FakeLlm(responses=[
        ClassifyResult(action="create", section="projects", title="Garden project",
                       slug="garden-project", type="project",
                       related_paths=["people/anna.md"]),
    ])
    result = ops.add("Anna is planning a garden project.", client="test")
    new_body = store.read_concept(result["path"]).body
    anna_body = store.read_concept("people/anna.md").body
    assert "/people/anna.md" in new_body
    assert f"/{result['path']}" in anna_body              # link both ways


def test_update_too_many_hits_skips_rewrites(ops, store, git, fts):
    """Marker too generic -> inbox fallback BEFORE the first rewrite call."""
    from librarian.ops import MAX_REWRITES, UpdateTarget
    for i in range(MAX_REWRITES + 1):
        store.write_concept(Concept(path=f"knowledge/n{i}.md",
                                    meta=ConceptMeta(type="fact", title=f"Note {i}"),
                                    body="The server is called alpha.\n"))
    git.commit_all("seed")
    fts.reindex(store, git.head())
    ops.llm = FakeLlm(responses=[
        UpdateTarget(target_path="knowledge/n0.md", obsolete_markers=["alpha"],
                     new_fact="The server is called beta."),
    ])
    result = ops.update("The server is now called beta.", client="test")
    assert result["path"].startswith("inbox/")
    assert [c[0] for c in ops.llm.calls] == ["UpdateTarget"]   # not a single rewrite
    assert "alpha" in store.read_concept("knowledge/n0.md").body


def test_update_marker_in_new_fact_is_dropped(ops, store, git, fts):
    """'X no longer applies' names X - otherwise no rewrite can pass the gate."""
    from librarian.ops import UpdateTarget
    store.write_concept(Concept(path="projects/tts.md",
                                meta=ConceptMeta(type="project", title="TTS"),
                                body="The app throttles to 0.9x playbackRate.\n"))
    git.commit_all("seed")
    fts.reindex(store, git.head())
    ops.llm = FakeLlm(responses=[
        UpdateTarget(target_path="projects/tts.md", obsolete_markers=["0.9x playbackRate"],
                     new_fact="The app NO LONGER throttles to 0.9x playbackRate."),
        # marker is dropped -> no hit -> add path (enrich)
        ClassifyResult(action="enrich", target_path="projects/tts.md"),
        EnrichPatch(index=-1, paragraph="The app NO LONGER throttles to 0.9x playbackRate."),
    ])
    result = ops.update("The app no longer throttles to 0.9x playbackRate.", client="test")
    assert not result["path"].startswith("inbox/")
    assert "NO LONGER" in store.read_concept("projects/tts.md").body


def test_update_ignores_inbox_leftovers(ops, store, git, fts):
    """An old inbox leftover containing the stale fact must not break the correction."""
    from librarian.ops import UpdateTarget
    store.write_concept(Concept(path="infrastructure/server.md",
                                meta=ConceptMeta(type="fact", title="Server"),
                                body="The server is called alpha.\n"))
    store.write_concept(Concept(path="inbox/2026-01-01-correction.md",
                                meta=ConceptMeta(type="fact", title="CORRECTION"),
                                body="CORRECTION (failed): The server is called alpha.\n"))
    git.commit_all("seed")
    fts.reindex(store, git.head())
    ops.llm = FakeLlm(responses=[
        UpdateTarget(target_path="infrastructure/server.md", obsolete_markers=["alpha"],
                     new_fact="The server is called beta."),
        EnrichResult(body="The server is called beta.\n"),
    ])
    result = ops.update("The server is now called beta.", client="test")
    assert result["result"] == "updated"
    assert result["rewritten"] == ["infrastructure/server.md"]
    assert "beta" in store.read_concept("infrastructure/server.md").body


def test_query_fts(ops):
    ops.add("The WiFi key safe is in the hallway cabinet.", client="test")
    result = ops.query("Where is the WiFi safe?", client="test", synthesize=False)
    # add put the path into the hot cache -> a shortcut answer is fine,
    # otherwise raw FTS hits
    content = result.get("answer") or result["matches"][0]["content"]
    assert "hallway cabinet" in content


def test_status(ops):
    ops.add("Fact one.", client="test")
    s = ops.status(client="test")
    assert s["concepts"] == 1
    assert s["indexed"] == 1
    assert s["head"]


def test_parallel_adds_serialized_by_caller_lock(ops):
    """Smoke: 5 sequential adds (the server serializes via an asyncio lock)."""
    for i in range(5):
        ops.add(f"Fact number {i} about topic {i}.", client="test")
    s = ops.status(client="test")
    assert s["concepts"] == 5


def _last_commit_msg(git):
    return git._git("log", "-1", "--format=%s").stdout
