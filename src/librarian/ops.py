"""Orchestration of the memory operations: code -> (LLM) -> code.

Every LLM step has a deterministic fallback; structural work
(slugs, frontmatter, backlinks, indexes, log, lint, commit) is done by code.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .agent import prompts
from .agent.llm import Llm, LlmUnavailable
from .bundle import links as linkmod
from .bundle import okf
from .bundle.gitops import GitOps
from .bundle.indexer import append_log, regenerate_indexes
from .bundle.lint import run_lint
from .bundle.okf import Concept, ConceptMeta
from .bundle.store import SECTIONS, BundleStore
from .config import Settings
from .search.fts import FtsIndex


# Fan-out brakes: both loops make ONE LLM call per hit, sequentially,
# with up to 300 s timeout - while holding the server's global write lock.
# A marker that is too generic (min. 3 characters!) must not block the
# service for hours. The next maintain/dream run does the rest.
MAX_REWRITES = 10        # memory_update: occurrences per correction
MAX_ORPHANS_PER_RUN = 5  # maintain: orphan wirings per run


class UpdateTarget(BaseModel):
    target_path: str
    obsolete_markers: list[str] = Field(default_factory=list)
    new_fact: str


class RelatedPaths(BaseModel):
    related_paths: list[str] = Field(default_factory=list)


class ClassifyResult(BaseModel):
    action: Literal["enrich", "create"]
    target_path: str = ""
    section: str = "knowledge"
    title: str = ""
    slug: str = ""
    type: str = "fact"
    description: str = ""
    related_paths: list[str] = Field(default_factory=list)


class EnrichResult(BaseModel):
    body: str


class EnrichPatch(BaseModel):
    index: int = -1          # paragraph to replace; -1 = new paragraph
    paragraph: str


def _new_paragraph_at(blocks: list[str]) -> int:
    """New paragraph before the related heading (backlinks stay at the bottom), else at the end."""
    from .bundle.links import related_heading
    heading = related_heading()
    for i in range(len(blocks) - 1, -1, -1):
        if blocks[i].strip() == heading:
            return i
    return len(blocks)


class MemoryOps:
    def __init__(self, settings: Settings, store: BundleStore, git: GitOps,
                 fts: FtsIndex, llm: Llm, hot: "HotCache | None" = None,
                 docs: "DocsStore | None" = None):
        from .bundle.docs import DocsStore
        from .hotcache import HotCache
        self.s = settings
        self.store = store
        self.git = git
        self.fts = fts
        self.llm = llm
        self.hot = hot or HotCache(fts, settings.cfg.hot_cache_ttl_days,
                                   settings.cfg.hot_cache_max_entries)
        self.docs = docs or DocsStore(settings.env.docs_path)

    # ---------- internal ----------

    def _lint_errors(self) -> set[tuple[str, str, str]]:
        return {(f.check, f.path, f.message) for f in run_lint(self.store).errors}

    def _finalize_mutation(self, baseline: set, log_msg: str, commit_msg: str) -> str:
        """Regenerate indexes + log, lint, commit.

        Roll back only on lint errors this mutation NEWLY introduced -
        legacy issues must not block writes permanently (status/maintain report those).
        """
        regenerate_indexes(self.store)
        append_log(self.store, log_msg)
        new_errors = self._lint_errors() - baseline
        if new_errors:
            self.git.rollback_uncommitted()
            raise RuntimeError(f"Mutation rolled back, new lint errors: {sorted(new_errors)}")
        head = self.git.commit_all(commit_msg)
        self.fts.reindex(self.store, head)
        return head

    def _unique_path(self, section: str, slug: str) -> str:
        slug = okf.slugify(slug)
        path = f"{section}/{slug}.md"
        n = 2
        while self.store.exists(path):
            path = f"{section}/{slug}-{n}.md"
            n += 1
        return path

    def _fallback_inbox(self, fact: str, client: str, reason: str, baseline: set) -> dict:
        """Deterministic fallback: the fact lands unfiled in inbox/."""
        slug = okf.slugify(" ".join(fact.split()[:6]))
        path = self._unique_path("inbox", f"{date.today().isoformat()}-{slug}")
        concept = Concept(
            path=path,
            meta=ConceptMeta(type="fact", title=fact[:80],
                             description="unfiled (LLM fallback)",
                             timestamp=okf.now_iso(), source=client),
            body=fact + "\n",
        )
        self.store.write_concept(concept)
        head = self._finalize_mutation(
            baseline,
            f"add (inbox fallback): [{concept.meta.title}](/{path})",
            f"memory_add(inbox): {slug} [via {client}] - fallback: {reason}")
        self.hot.bump(path, concept.meta.title)
        self.fts.trace(okf.now_iso(), client, "add", fact[:200], "inbox-fallback", reason)
        return {"result": "created", "path": path, "note": f"inbox fallback ({reason}); "
                "dreaming/maintain will file it later", "commit": head[:8]}

    # ---------- Operations ----------

    def add(self, fact: str, client: str) -> dict:
        """File a fact: FTS candidates -> LLM (enrich|create) -> code writes."""
        fact = fact.strip()
        if not fact:
            raise ValueError("Empty fact")
        baseline = self._lint_errors()
        # Docs are read-only reference docs - excluded as enrich target
        candidates = self.fts.search(fact, self.s.cfg.fts_candidates, memory_only=True)
        cand_text = "\n".join(
            f"- {h.path} [{h.type}] {h.title}: {h.description}" for h in candidates
        ) or "(none)"
        try:
            decision = self.llm.chat_json(
                prompts.CLASSIFY_SYSTEM,
                prompts.CLASSIFY_USER.format(client=client, fact=fact, candidates=cand_text),
                ClassifyResult,
            )
        except LlmUnavailable as e:
            return self._fallback_inbox(fact, client, str(e), baseline)

        if decision.action == "enrich":
            target = decision.target_path.lstrip("/")
            if not self.store.exists(target):
                # LLM hallucinated -> hard validation kicks in, fall back to create
                decision.action = "create"
            else:
                return self._enrich(fact, client, target, baseline)
        return self._create(fact, client, decision, baseline)

    def _enrich(self, fact: str, client: str, target: str, baseline: set) -> dict:
        concept = self.store.read_concept(target)
        # The model sees the whole body but answers with only ONE paragraph.
        # Writing a 25 KB body back in full took longer than the 30 s of the
        # cloud tier and came back as broken JSON locally (10.09.).
        blocks = concept.body.rstrip("\n").split("\n\n") if concept.body.strip() else []
        numbered = "\n\n".join(f"[{i}] {b}" for i, b in enumerate(blocks))
        try:
            patch = self.llm.chat_json(
                prompts.ENRICH_SYSTEM,
                prompts.ENRICH_USER.format(fact=fact, path=target, body=numbered),
                EnrichPatch,
            )
            paragraph = re.sub(r"^\[\d+\]\s*", "", patch.paragraph.strip())
            if len(paragraph) < 10:
                raise LlmUnavailable("Enrich paragraph suspiciously empty")
            # Headings and link lists are never overwritten - if the index
            # points at one, the fact is added as its own paragraph.
            if (0 <= patch.index < len(blocks)
                    and not blocks[patch.index].lstrip().startswith(("#", "- ["))):
                if len(paragraph) < len(blocks[patch.index].strip()) // 2:
                    raise LlmUnavailable("Enrich paragraph suspiciously shrunk")
                blocks[patch.index] = paragraph
            else:
                blocks.insert(_new_paragraph_at(blocks), paragraph)
        except LlmUnavailable as e:
            return self._fallback_inbox(fact, client, str(e), baseline)
        concept.body = "\n\n".join(blocks) + "\n"
        concept.meta.timestamp = okf.now_iso()
        self.store.write_concept(concept)
        head = self._finalize_mutation(
            baseline,
            f"enrich: [{concept.meta.title or target}](/{target})",
            f"memory_add(enrich {target}): {fact[:60]} [via {client}]")
        self.hot.bump(target, concept.meta.description or fact[:100])
        self.fts.trace(okf.now_iso(), client, "add", fact[:200], "enrich", target)
        return {"result": "enriched", "path": target, "commit": head[:8]}

    def _create(self, fact: str, client: str, d: ClassifyResult, baseline: set) -> dict:
        section = d.section if d.section in SECTIONS else "knowledge"
        path = self._unique_path(section, d.slug or d.title or fact[:40])
        title = d.title or fact[:60]
        body = fact + "\n"
        # Link both ways: new concept links related ones, related ones link back
        related = [p.lstrip("/") for p in d.related_paths if self.store.exists(p.lstrip("/"))]
        for rel_path in related[:3]:
            rel_concept = self.store.read_concept(rel_path)
            body = linkmod.append_backlink(body, rel_path, rel_concept.meta.title or rel_path)
            rel_concept.body = linkmod.append_backlink(rel_concept.body, path, title)
            self.store.write_concept(rel_concept)
        concept = Concept(
            path=path,
            meta=ConceptMeta(type=d.type or "fact", title=title,
                             description=d.description, timestamp=okf.now_iso(),
                             source=client),
            body=body,
        )
        self.store.write_concept(concept)
        head = self._finalize_mutation(
            baseline,
            f"create: [{title}](/{path})",
            f"memory_add(create {path}) [via {client}]")
        self.hot.bump(path, d.description or title)
        self.fts.trace(okf.now_iso(), client, "add", fact[:200], "create", path)
        return {"result": "created", "path": path, "linked": related, "commit": head[:8]}

    def query(self, question: str, client: str, synthesize: bool = True) -> dict:
        """Hot-cache shortcut -> librarian loop (synthesis) -> FTS fallback."""
        # The docs mirror can change independently of the container (nightly
        # git pull) - the fingerprint check costs only a few stat() calls
        self.fts.ensure_docs_fresh(self.docs)
        # Shortcut: an unambiguous hot hit is answered WITHOUT the LLM (~ms)
        hot_path = self.hot.shortcut(self.fts.search(question, self.s.cfg.fts_candidates))
        if hot_path:
            try:
                c = self.store.read_concept(hot_path)
                self.hot.bump(hot_path)
                self.fts.trace(okf.now_iso(), client, "query", question[:200],
                               "hot_cache", hot_path)
                return {"question": question, "answer": c.body.strip(),
                        "sources": [hot_path], "via": "hot_cache"}
            except Exception:
                self.hot.remove(hot_path)  # file gone -> remove stale cache entry
        if synthesize:
            from .agent.loop import run_query_loop
            try:
                r = run_query_loop(self.s, self.store, self.fts, self.llm, question,
                                   docs=self.docs)
                self.fts.trace(okf.now_iso(), client, "query", question[:200],
                               "librarian", json.dumps(r.trace, ensure_ascii=False)[:2000])
                for src in r.sources:
                    try:
                        self.hot.bump(src, self.store.read_concept(src).meta.description)
                    except Exception:
                        pass
                return {"question": question, "answer": r.answer, "sources": r.sources}
            except LlmUnavailable as e:
                fallback_reason = str(e)[:200]
        else:
            fallback_reason = "synthesis disabled"
        hits = self.fts.search(question, self.s.cfg.fts_candidates)
        results = []
        for h in hits[:5]:
            try:
                c = self.store.read_concept(h.path)
            except Exception:
                continue
            results.append({"path": h.path, "title": c.meta.title, "type": c.meta.type,
                            "content": c.body.strip()})
        self.fts.trace(okf.now_iso(), client, "query", question[:200], "fts",
                       ",".join(r["path"] for r in results))
        return {"question": question, "matches": results,
                "note": ("" if results else "No matches in memory.")
                        + (f" (FTS-Fallback: {fallback_reason})" if synthesize else "")}

    def update(self, correction: str, client: str) -> dict:
        """Contradiction pipeline: the stale fact must disappear EVERYWHERE.

        LLM determines markers of the stale fact; code greps ALL concepts for them;
        LLM rewrites each occurrence; code verifies via grep that the
        markers are gone - otherwise full rollback.
        """
        correction = correction.strip()
        if not correction:
            raise ValueError("Empty correction")
        baseline = self._lint_errors()
        candidates = self.fts.search(correction, self.s.cfg.fts_candidates, memory_only=True)
        cand_text = "\n".join(
            f"- {h.path} [{h.type}] {h.title}: {h.description}" for h in candidates
        ) or "(none)"
        try:
            target = self.llm.chat_json(
                prompts.UPDATE_SYSTEM,
                prompts.UPDATE_USER.format(client=client, correction=correction,
                                           candidates=cand_text),
                UpdateTarget,
            )
        except LlmUnavailable as e:
            return self._fallback_inbox(f"CORRECTION: {correction}", client, str(e), baseline)

        def _norm(s: str) -> str:
            # Normalize path separators: C:/x and C:\x should match
            return s.lower().replace("\\", "/")

        _WORD = "abcdefghijklmnopqrstuvwxyz0123456789_"

        def _marker_in(marker: str, body: str) -> bool:
            # Word-boundary match instead of a bare substring: a marker whose edge
            # is a word character must not count in the middle of a longer token
            # - otherwise the new, correct term triggers the old marker
            # (e.g. 'NEWS' in 'NEWSLETTER', 'upload_id' in 'upload_ids') and the
            # rewrite is wrongly treated as "stale fact marker survived".
            m, b = _norm(marker), _norm(body)
            if not m:
                return False
            start = 0
            while (i := b.find(m, start)) >= 0:
                j = i + len(m)
                left_ok = not (m[0] in _WORD and i > 0 and b[i - 1] in _WORD)
                right_ok = not (m[-1] in _WORD and j < len(b) and b[j] in _WORD)
                if left_ok and right_ok:
                    return True
                start = i + 1
            return False

        # A marker that occurs in the NEW fact itself is useless as an
        # obsolescence signal: corrections like "no longer runs on 0.9x"
        # must name the old value. If it stayed in the marker set, no
        # correct rewrite could ever pass the gate - the correction would end
        # up in the inbox endlessly (observed on projects/jarvis.md, 12.-16.08.).
        markers = [m for m in (m.strip() for m in target.obsolete_markers)
                   if len(m) >= 3 and not _marker_in(m, target.new_fact)]

        def _hit(body: str) -> bool:
            return any(_marker_in(m, body) for m in markers)

        def _curated():
            """Curated knowledge without the inbox.

            inbox/ holds raw reports - for a failed correction, the correction
            text itself, i.e. the stale fact verbatim. If those were grepped
            and verified too, every new attempt would fail on its own
            fallback entry and create the next one (observed:
            4 identical inbox corpses 08-12 to 08-15). The inbox is re-filed
            by dreaming/maintain, where the correction takes effect.
            """
            return (c for c in self.store.iter_concepts()
                    if not c.path.startswith("inbox/"))

        # Grep occurrences deterministically across the WHOLE bundle
        affected: list[str] = []
        for concept in _curated():
            if _hit(concept.body):
                affected.append(concept.path)
        if not affected:
            # nothing to replace -> file as a new fact
            return self.add(target.new_fact or correction, client)
        if len(affected) > MAX_REWRITES:
            # abort before the first rewrite - otherwise we burn N x 300 s
            return self._fallback_inbox(
                f"CORRECTION: {correction}", client,
                f"markers too unspecific {markers}: {len(affected)} occurrences "
                f"(max. {MAX_REWRITES})", baseline)

        rewritten: list[str] = []
        try:
            for path in affected:
                concept = self.store.read_concept(path)
                # Only paragraphs with a marker go to the model, the rest stays
                # byte-identical. A 25 KB body as ONE JSON string came back
                # broken from gemma4 (projects/moviemaker.md, 10.09.).
                blocks = concept.body.split("\n\n")
                for i, block in enumerate(blocks):
                    if not _hit(block):
                        continue
                    result = self.llm.chat_json(
                        prompts.REWRITE_SYSTEM,
                        prompts.REWRITE_USER.format(new_fact=target.new_fact,
                                                    markers=", ".join(markers),
                                                    path=path, body=block),
                        EnrichResult,
                    )
                    new_block = result.body.strip()
                    if _hit(new_block):
                        raise LlmUnavailable(f"Stale fact marker survived rewrite in {path}")
                    # As with enrich: a paragraph that shrinks to under half
                    # lost more than the stale fact (observed 10.09.:
                    # timer value and history were dropped while rewriting).
                    if len(new_block) < max(10, len(block.strip()) // 2):
                        raise LlmUnavailable(f"Rewrite paragraph suspiciously shrunk in {path}")
                    blocks[i] = new_block
                concept.body = "\n\n".join(blocks).rstrip() + "\n"
                concept.meta.timestamp = okf.now_iso()
                self.store.write_concept(concept)
                rewritten.append(path)
            # Verification across the whole bundle (lint gate of the contradiction rule)
            for concept in _curated():
                if _hit(concept.body):
                    raise LlmUnavailable(f"Marker still present in {concept.path}")
        except LlmUnavailable as e:
            self.git.rollback_uncommitted()
            return self._fallback_inbox(f"CORRECTION (update failed: {e}): {correction}",
                                        client, str(e), baseline)
        head = self._finalize_mutation(
            baseline,
            f"update: {target.new_fact[:80]}",
            f"memory_update({','.join(rewritten)}): {correction[:60]} [via {client}]")
        for p in rewritten:
            self.hot.bump(p, target.new_fact[:100])
        self.fts.trace(okf.now_iso(), client, "update", correction[:200], "rewrite",
                       json.dumps({"markers": markers, "rewritten": rewritten},
                                  ensure_ascii=False))
        return {"result": "updated", "rewritten": rewritten,
                "markers_removed": markers, "commit": head[:8]}

    def merge(self, source: str, target: str, client: str) -> dict:
        """Duplicate merge (destructive - only after human approval/review).

        LLM merges the bodies (fallback: append); code redirects ALL
        references to the source, takes over aliases, deletes the source and
        verifies via lint that nothing breaks - otherwise rollback.
        """
        source = source.lstrip("/")
        target = target.lstrip("/")
        if source == target or not (self.store.exists(source) and self.store.exists(target)):
            raise ValueError(f"Merge impossible: {source} -> {target}")
        baseline = self._lint_errors()
        src = self.store.read_concept(source)
        dst = self.store.read_concept(target)
        try:
            merged = self.llm.chat_json(
                prompts.MERGE_SYSTEM,
                prompts.MERGE_USER.format(target=target, target_body=dst.body,
                                          source=source, source_body=src.body),
                EnrichResult)
            new_body = merged.body.strip()
            if len(new_body) < max(len(dst.body), len(src.body)) // 2:
                raise LlmUnavailable("Merge body suspiciously shrunk")
        except LlmUnavailable:
            # deterministic fallback: nothing is lost, just ugly
            new_body = dst.body.strip() + "\n\n## Merged from " \
                f"{src.meta.title or source}\n\n" + src.body.strip()
        dst.body = new_body + "\n"
        # Take over aliases so the target stays findable under the old name
        for alias in [src.meta.title, *src.meta.aliases]:
            if alias and alias != dst.meta.title and alias not in dst.meta.aliases:
                dst.meta.aliases.append(alias)
        dst.meta.timestamp = okf.now_iso()
        try:
            # redirect all references bundle-wide to the target (code, no LLM)
            for concept in self.store.iter_concepts():
                if concept.path in (source, target):
                    continue
                replaced = concept.body.replace(f"(/{source})", f"(/{target})")
                if replaced != concept.body:
                    concept.body = replaced
                    self.store.write_concept(concept)
            # Remove self-references in the merge result ((/target) in its own body is ok)
            dst.body = dst.body.replace(f"(/{source})", f"(/{target})")
            self.store.write_concept(dst)
            self.store.delete(source)
            head = self._finalize_mutation(
                baseline,
                f"merge: [{src.meta.title or source}] -> [{dst.meta.title or target}](/{target})",
                f"memory_merge({source} -> {target}) [via {client}]")
        except Exception:
            self.git.rollback_uncommitted()
            raise
        self.hot.remove(source)
        self.hot.bump(target, dst.meta.description)
        self.fts.trace(okf.now_iso(), client, "merge", f"{source} -> {target}",
                       "merge", head[:8])
        return {"result": "merged", "target": target, "removed": source,
                "commit": head[:8]}

    def wire_orphan(self, orphan_path: str, related_paths: list[str],
                    client: str) -> list[str]:
        """Set backlinks in both directions, indexes, own commit.

        Shared endpoint for maintain() (FTS prefilter) and sweep
        (full-corpus pipeline). Returns the paths actually linked.
        """
        orphan = self.store.read_concept(orphan_path)
        linked: list[str] = []
        for rel_path in [p.lstrip("/") for p in related_paths][:2]:
            if not self.store.exists(rel_path) or rel_path == orphan_path:
                continue
            other = self.store.read_concept(rel_path)
            other.body = linkmod.append_backlink(
                other.body, orphan_path, orphan.meta.title or orphan_path)
            orphan.body = linkmod.append_backlink(
                orphan.body, rel_path, other.meta.title or rel_path)
            self.store.write_concept(other)
            linked.append(rel_path)
        if linked:
            self.store.write_concept(orphan)
            regenerate_indexes(self.store)
            head = self.git.commit_all(
                f"maintain: orphan {orphan_path} wired to {','.join(linked)} "
                f"[via {client}]")
            self.fts.reindex(self.store, head)
        return linked

    def maintain(self, client: str, wire_orphans: bool = True) -> dict:
        """Lint -> deterministic fixes -> orphan wiring (LLM, validated individually).

        Duplicate candidates are NOT merged automatically but put in the
        review_queue (destructive -> human decides, M4/M5 UI).
        """
        from .bundle.lint import fix_deterministic
        report = run_lint(self.store)
        actions: list[str] = []

        fixes = fix_deterministic(self.store, report)
        if fixes:
            head = self.git.commit_all(f"maintain: deterministic fixes [via {client}]")
            self.fts.reindex(self.store, head)
            actions += fixes

        # Duplicates -> review queue (once per pair)
        for f in run_lint(self.store).by_check("duplicate"):
            detail = json.dumps({"path": f.path, **f.context}, ensure_ascii=False)
            self.fts.queue_review("duplicate", detail)
            actions.append(f"review-queue: duplicate {f.path} <-> {f.context.get('other')}")

        # Wire orphans via LLM - each repair gets its own commit
        if wire_orphans:
            orphans = [f for f in run_lint(self.store).by_check("orphan")
                       if not f.path.startswith("inbox/")]  # inbox is unwired by design
            if len(orphans) > MAX_ORPHANS_PER_RUN:
                actions.append(f"orphans deferred: {len(orphans) - MAX_ORPHANS_PER_RUN} of "
                               f"{len(orphans)} (budget {MAX_ORPHANS_PER_RUN}/run)")
            for f in orphans[:MAX_ORPHANS_PER_RUN]:
                try:
                    orphan = self.store.read_concept(f.path)
                except Exception:
                    continue
                # docs are not a link target - otherwise the LLM picks them and
                # validation silently discards them (orphan would stay orphaned forever)
                hits = self.fts.search(f"{orphan.meta.title} {orphan.meta.description} "
                                       f"{orphan.body[:200]}", 6, memory_only=True)
                cand = [h for h in hits if h.path != f.path]
                if not cand:
                    continue
                cand_text = "\n".join(f"- {h.path} [{h.type}] {h.title}: {h.description}"
                                      for h in cand)
                try:
                    rel = self.llm.chat_json(
                        prompts.ORPHAN_SYSTEM,
                        prompts.ORPHAN_USER.format(path=f.path, title=orphan.meta.title,
                                                   body=orphan.body[:1500],
                                                   candidates=cand_text),
                        RelatedPaths,
                    )
                except LlmUnavailable:
                    continue  # orphan stays; next maintain/dream attempt
                linked = self.wire_orphan(f.path, rel.related_paths, client)
                if linked:
                    actions.append(f"orphan {f.path} -> {linked}")

        final = run_lint(self.store)
        self.fts.trace(okf.now_iso(), client, "maintain", "", "maintain",
                       json.dumps(actions, ensure_ascii=False)[:2000])
        return {"actions": actions, "lint": final.summary(),
                "open_findings": [f"{x.severity} {x.check} {x.path}: {x.message}"
                                  for x in final.findings][:30],
                "review_open": self.fts.open_reviews()}

    def status(self, client: str) -> dict:
        report = run_lint(self.store)
        paths = self.store.list_concept_paths()
        by_section: dict[str, int] = {}
        for p in paths:
            by_section[p.split("/")[0]] = by_section.get(p.split("/")[0], 0) + 1
        return {
            "concepts": len(paths),
            "by_section": by_section,
            "lint": report.summary(),
            "lint_findings": [f"{f.severity} {f.check} {f.path}: {f.message}"
                              for f in report.findings][:20],
            "head": self.git.head()[:8],
            "indexed": self.fts.count(),
            "docs_indexed": self.fts.docs_count(),
        }


def build_ops(settings: Settings) -> MemoryOps:
    from .bundle.docs import DocsStore
    store = BundleStore(settings.env.bundle_path)
    store.init_layout()
    git = GitOps(settings.env.bundle_path)
    git.ensure_repo()
    fts = FtsIndex(Path(settings.env.data_path) / "librarian.db")
    fts.ensure_fresh(store, git.head())
    docs = DocsStore(settings.env.docs_path)
    fts.ensure_docs_fresh(docs)
    llm = Llm(settings)
    return MemoryOps(settings, store, git, fts, llm, docs=docs)
