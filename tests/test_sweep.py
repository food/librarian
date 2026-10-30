"""Sweep pipeline: map-reduce-verify + wiring via the shared endpoint."""
from librarian.agent.llm import LlmUnavailable
from librarian.bundle.okf import Concept, ConceptMeta
from librarian.ops import MemoryOps
from librarian.sweep import RelatedVerdict, sweep_orphan


class StubLlm:
    """Map/verify answers: related only when the server candidate is asked about."""

    def __init__(self, fail_on: str = ""):
        self.fail_on = fail_on
        self.calls = []

    def chat_json(self, system, user, schema, model=None):
        self.calls.append(user)
        if self.fail_on and self.fail_on in user:
            raise LlmUnavailable("test: down")
        related = "infrastructure/ai-server.md" in user
        return RelatedVerdict(related=related, score=8 if related else 0)


def put(store, path, title, body, type_="fact"):
    store.write_concept(Concept(path=path,
                                meta=ConceptMeta(type=type_, title=title),
                                body=body))


def make_ops(settings, store, git, fts, llm):
    put(store, "knowledge/qwen-tts.md", "Qwen-TTS",
        "The TTS server shares the GPU with Ollama.\n")           # the orphan
    put(store, "infrastructure/ai-server.md", "AI server",
        "The LAN server ai hosts GPU services.\n")                # related
    put(store, "people/anna.md", "Anna", "Knows nobody.\n")       # unrelated
    return MemoryOps(settings, store, git, fts, llm)


def test_sweep_finds_verifies_and_wires(settings, store, git, fts):
    ops = make_ops(settings, store, git, fts, StubLlm())
    r = sweep_orphan(ops, "knowledge/qwen-tts.md")
    assert r["scanned"] == 2                      # orphan itself excluded
    assert r["map_hits"] == ["infrastructure/ai-server.md"]
    assert r["confirmed"] == ["infrastructure/ai-server.md"]

    linked = ops.wire_orphan("knowledge/qwen-tts.md", r["confirmed"], client="test")
    assert linked == ["infrastructure/ai-server.md"]
    assert "/infrastructure/ai-server.md" in store.read_concept("knowledge/qwen-tts.md").body
    assert "/knowledge/qwen-tts.md" in store.read_concept("infrastructure/ai-server.md").body


def test_sweep_survives_llm_failure_per_candidate(settings, store, git, fts):
    # a failure on the unrelated candidate must not cost the hit
    ops = make_ops(settings, store, git, fts, StubLlm(fail_on="people/anna.md"))
    r = sweep_orphan(ops, "knowledge/qwen-tts.md")
    assert r["confirmed"] == ["infrastructure/ai-server.md"]
