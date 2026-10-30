"""Librarian query loop: FTS prefilter -> bounded traversal -> synthesis.

Bounded and strictly validated: at most query_max_iterations rounds, each
round the LLM reads up to 3 additional files (only existing paths are accepted).
Deterministic fallback: raw FTS hits (the caller handles LlmUnavailable).
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from ..bundle.store import BundleStore
from ..config import Settings
from ..search.fts import FtsIndex
from . import prompts
from .llm import Llm, LlmUnavailable


class QueryStep(BaseModel):
    action: Literal["read", "answer"]
    paths: list[str] = Field(default_factory=list)   # on read: max 3 concept paths
    answer: str = ""                                  # on answer
    sources: list[str] = Field(default_factory=list)  # on answer: paths used


class QueryResult(BaseModel):
    answer: str
    sources: list[str]
    trace: list[dict]


def run_query_loop(settings: Settings, store: BundleStore, fts: FtsIndex,
                   llm: Llm, question: str, docs=None) -> QueryResult:
    """Raises LlmUnavailable if the LLM fails to deliver - the caller does the fallback.

    docs: optional read-only DocsStore (reference shelf); doc: paths are
    read as raw text but are never a write target.
    """
    from ..bundle.docs import is_doc

    def read_any(path: str) -> str | None:
        try:
            if is_doc(path):
                return docs.read_text(path).strip() if docs else None
            return store.read_concept(path).body.strip()
        except Exception:
            return None

    def path_exists(path: str) -> bool:
        return docs.exists(path) if is_doc(path) and docs else store.exists(path)

    hits = fts.search(question, settings.cfg.fts_candidates)
    candidates = "\n".join(
        f"- {h.path} [{h.type}] {h.title}: {h.description}" for h in hits) or "(none)"

    read_docs: dict[str, str] = {}
    trace: list[dict] = [{"step": "fts", "hits": [h.path for h in hits]}]

    # Initial corpus: read the top 3 hits directly (saves an LLM round)
    for h in hits[:3]:
        body = read_any(h.path)
        if body is not None:
            read_docs[h.path] = body

    for iteration in range(settings.cfg.query_max_iterations):
        docs_text = "\n\n".join(
            f"### {path}\n{body[:3000]}" for path, body in read_docs.items()
        ) or "(nothing read yet)"
        step = llm.chat_json(
            prompts.QUERY_SYSTEM,
            prompts.QUERY_USER.format(question=question, candidates=candidates,
                                      docs=docs_text),
            QueryStep,
        )
        if step.action == "answer" and step.answer.strip():
            sources = [s.lstrip("/") for s in step.sources if s.lstrip("/") in read_docs]
            trace.append({"step": "answer", "iteration": iteration, "sources": sources})
            return QueryResult(answer=step.answer.strip(),
                               sources=sources or list(read_docs), trace=trace)
        # read round: accept only existing, not yet read paths
        wanted = [p if is_doc(p) else p.lstrip("/") for p in step.paths][:3]
        new_paths = [p for p in wanted if p not in read_docs and path_exists(p)]
        trace.append({"step": "read", "iteration": iteration,
                      "wanted": wanted, "accepted": new_paths})
        if not new_paths:
            break  # LLM is going in circles -> bail out and fall back
        for p in new_paths:
            body = read_any(p)
            if body is not None:
                read_docs[p] = body

    raise LlmUnavailable(
        f"Query loop without answer after budget (read: {list(read_docs)}); "
        f"Trace: {json.dumps(trace, ensure_ascii=False)[:500]}")
