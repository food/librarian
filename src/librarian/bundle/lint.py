"""Deterministic linter (no LLM) + deterministic fixes.

Checks:
- frontmatter:        file parseable, 'type' present (OKF conformance)
- broken-link:        link target does not exist
- orphan:             concept without incoming links from other concepts
                      (indexes do NOT count as link sources)
- index-missing:      concept missing from the section index (only possible after manual edits)
- index-ghost:        index entry points to a non-existent concept
- backlink-asym:      A links to B, but B does not link back to A (auto-fix)
- duplicate:          title/alias collision (case-insensitive) -> review candidate
- naming:             path does not follow the slug convention
- stub:               body practically empty
- superseded:         concept marked superseded is still referenced
- log-format:         log.md without '# Log' header or broken date groups

fix_deterministic() repairs: index-missing/index-ghost (regeneration),
backlink-asym (add the backlink), resolvable broken-links (unique basename
elsewhere). Everything else belongs in maintain/dreaming (LLM/review).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import links as linkmod
from .indexer import regenerate_indexes
from .store import BundleStore

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9\-./]*\.md$")
DATE_HDR_RE = re.compile(r"^## (\d{4}-\d{2}-\d{2})$")


@dataclass
class Finding:
    check: str      # e.g. "broken-link"
    severity: str   # "error" | "warning"
    path: str
    message: str
    # machine-readable context for fixes (e.g. link target, partner path)
    context: dict = field(default_factory=dict)


@dataclass
class LintReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def by_check(self, check: str) -> list[Finding]:
        return [f for f in self.findings if f.check == check]

    def summary(self) -> str:
        if not self.findings:
            return "Lint: 0 findings"
        return f"Lint: {len(self.findings)} findings ({len(self.errors)} errors)"


def _resolve(target: str, path_set: set[str], all_paths: list[str]) -> str | None:
    """Resolve a link target to an existing concept (also [[wiki]] without a path)."""
    if target in path_set:
        return target
    if "/" not in target:
        return linkmod.resolve_wiki_target(target, all_paths)
    return None


def run_lint(store: BundleStore) -> LintReport:
    report = LintReport()
    all_paths = store.list_concept_paths()
    path_set = set(all_paths)
    inbound: dict[str, int] = {p: 0 for p in all_paths}
    outgoing: dict[str, set[str]] = {}
    concepts = []

    for path in all_paths:
        try:
            concepts.append(store.read_concept(path))
        except Exception as e:  # noqa: BLE001 - broken file = finding, not a crash
            report.findings.append(Finding("frontmatter", "error", path, str(e)))

    # Links, naming, stubs
    for c in concepts:
        if not SLUG_RE.match(c.path):
            report.findings.append(Finding(
                "naming", "warning", c.path,
                "Path does not follow the slug convention (ascii, lowercase, '-')"))
        if len(c.body.strip()) < 15:
            report.findings.append(Finding("stub", "warning", c.path, "Body practically empty"))
        resolved_targets: set[str] = set()
        for target in linkmod.extract_links(c.body, c.path):
            resolved = _resolve(target, path_set, all_paths)
            if resolved:
                inbound[resolved] += 1
                resolved_targets.add(resolved)
            else:
                report.findings.append(Finding(
                    "broken-link", "error", c.path,
                    f"Target not found: {target}", {"target": target}))
        outgoing[c.path] = resolved_targets

    # Orphans, backlink symmetry, superseded
    for c in concepts:
        if inbound.get(c.path, 0) == 0:
            report.findings.append(Finding(
                "orphan", "warning", c.path, "no other concept links here"))
        for target in outgoing.get(c.path, set()):
            if c.path not in outgoing.get(target, set()):
                report.findings.append(Finding(
                    "backlink-asym", "warning", target,
                    f"does not link back to {c.path}", {"partner": c.path}))
        extra = c.meta.model_extra or {}
        if extra.get("superseded") and inbound.get(c.path, 0) > 0:
            report.findings.append(Finding(
                "superseded", "error", c.path,
                "superseded but still referenced - stale fact must be removed everywhere"))

    # Duplicate candidates (title/alias, case-insensitive)
    seen_names: dict[str, str] = {}
    for c in concepts:
        names = [c.meta.title] + c.meta.aliases
        for name in filter(None, names):
            key = name.strip().lower()
            if key in seen_names and seen_names[key] != c.path:
                report.findings.append(Finding(
                    "duplicate", "warning", c.path,
                    f"Title/alias '{name}' collides with {seen_names[key]}",
                    {"other": seen_names[key], "name": name}))
            else:
                seen_names[key] = c.path

    # Index completeness (only relevant after manual edits - indexes are generated)
    # also check sections WITHOUT concepts - their index can contain ghosts
    sections = {p.split("/")[0] for p in all_paths}
    sections |= {store.rel(p.parent) for p in store.root.glob("*/index.md")}
    for section in sections:
        idx_path = f"{section}/index.md"
        if not store.exists(idx_path):
            report.findings.append(Finding("index-missing", "warning", idx_path,
                                           "Section index missing"))
            continue
        idx_text = store.read_text(idx_path)
        idx_targets = set(linkmod.extract_links(idx_text, idx_path))
        for p in (p for p in all_paths if p.startswith(section + "/")):
            if p not in idx_targets:
                report.findings.append(Finding("index-missing", "warning", idx_path,
                                               f"{p} missing from index", {"target": p}))
        for t in idx_targets:
            if t.endswith("index.md"):
                continue
            if t not in path_set:
                report.findings.append(Finding("index-ghost", "warning", idx_path,
                                               f"Ghost entry: {t}", {"target": t}))

    # log.md format
    if store.exists("log.md"):
        lines = store.read_text("log.md").splitlines()
        if not lines or lines[0].strip() != "# Log":
            report.findings.append(Finding("log-format", "warning", "log.md",
                                           "first line must be '# Log'"))
        dates = [m.group(1) for line in lines if (m := DATE_HDR_RE.match(line.strip()))]
        if dates != sorted(dates, reverse=True):
            report.findings.append(Finding("log-format", "warning", "log.md",
                                           "Date groups not sorted in descending order"))

    return report


def fix_deterministic(store: BundleStore, report: LintReport,
                      max_rounds: int = 3) -> list[str]:
    """Deterministic repairs until convergence.

    A fix can produce follow-up findings (repaired broken link -> new
    asymmetric backlink), so we re-lint and keep repairing until there is
    nothing left to do (at most max_rounds rounds).
    The caller commits (own commit per maintain run) and reindexes.
    """
    all_fixes: list[str] = []
    for _ in range(max_rounds):
        fixes = _fix_round(store, report)
        all_fixes += fixes
        if not [f for f in fixes if not f.startswith("Indexes")]:
            break
        report = run_lint(store)
    return all_fixes


def _fix_round(store: BundleStore, report: LintReport) -> list[str]:
    fixes: list[str] = []
    all_paths = store.list_concept_paths()

    # Backlink symmetry: add the missing backlink
    for f in report.by_check("backlink-asym"):
        partner = f.context.get("partner", "")
        if not (store.exists(f.path) and store.exists(partner)):
            continue
        concept = store.read_concept(f.path)
        partner_title = store.read_concept(partner).meta.title or partner
        concept.body = linkmod.append_backlink(concept.body, partner, partner_title)
        store.write_concept(concept)
        fixes.append(f"backlink {f.path} -> {partner} added")

    # Resolvable broken links: unique basename exists elsewhere
    for f in report.by_check("broken-link"):
        target = f.context.get("target", "")
        base = target.removesuffix(".md").split("/")[-1] + ".md"
        resolved = linkmod.resolve_wiki_target(base, all_paths)
        if resolved and store.exists(f.path):
            text = store.read_text(f.path)
            fixed = text.replace(f"(/{target})", f"(/{resolved})").replace(f"({target})", f"(/{resolved})")
            if fixed != text:
                store.write_text(f.path, fixed)
                fixes.append(f"broken-link in {f.path}: {target} -> {resolved}")

    # Regenerating indexes fixes index-missing/index-ghost wholesale
    if report.by_check("index-missing") or report.by_check("index-ghost") or fixes:
        regenerate_indexes(store)
        fixes.append("Indexes regenerated")

    return fixes
