"""Deterministic generation of index.md (per level) and log.md.

Indexes are generated entirely from frontmatter (title/description) and
never edited by the LLM - so they cannot "lie".
"""
from __future__ import annotations

from datetime import date

from .okf import Concept
from .store import SECTIONS, BundleStore

SECTION_TITLES = {
    "people": "People",
    "projects": "Projects",
    "infrastructure": "Infrastructure",
    "preferences": "Preferences & Conventions",
    "knowledge": "Knowledge & Facts",
    "events": "Events",
    "inbox": "Inbox (unsorted)",
}


def _entry_line(c: Concept) -> str:
    title = c.meta.title or c.concept_id.split("/")[-1]
    desc = f" - {c.meta.description}" if c.meta.description else ""
    return f"- [{title}](/{c.path}){desc}"


def regenerate_indexes(store: BundleStore) -> None:
    """Rewrite all section indexes and the root index."""
    by_section: dict[str, list[Concept]] = {s: [] for s in SECTIONS}
    for concept in store.iter_concepts():
        section = concept.path.split("/")[0]
        if section in by_section:
            by_section[section].append(concept)

    for section, concepts in by_section.items():
        title = SECTION_TITLES.get(section, section)
        lines = [f"# {title}", ""]
        for c in sorted(concepts, key=lambda c: c.path):
            lines.append(_entry_line(c))
        if not concepts:
            lines.append("_empty_")
        store.write_text(f"{section}/index.md", "\n".join(lines) + "\n")

    # Root index: sections + counts + one-liners
    lines = ["# Index", "", "Memory library (OKF bundle). Sections:", ""]
    for section in SECTIONS:
        n = len(by_section.get(section, []))
        lines.append(f"- [{SECTION_TITLES.get(section, section)}](/{section}/index.md) - {n} concepts")
    store.write_text("index.md", "\n".join(lines) + "\n")


def append_log(store: BundleStore, message: str) -> None:
    """log.md: date groups, newest first (written by the code, not the LLM)."""
    today = date.today().isoformat()
    text = store.read_text("log.md") if store.exists("log.md") else "# Log\n"
    lines = text.splitlines()
    header = f"## {today}"
    entry = f"- {message}"
    if header in lines:
        idx = lines.index(header)
        lines.insert(idx + 1, entry)
    else:
        # insert the new date group directly under the H1 (newest first)
        insert_at = 1
        while insert_at < len(lines) and not lines[insert_at].startswith("## "):
            if lines[insert_at].strip():
                insert_at += 1
            else:
                insert_at += 1
        lines = lines[:1] + ["", header, entry] + lines[1:]
    store.write_text("log.md", "\n".join(lines).strip() + "\n")
