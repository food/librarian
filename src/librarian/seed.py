"""Seed text: compact live overview of what is in memory.

Goes into the MCP instructions and (short form) into tool descriptions so
client models KNOW a query is worthwhile - instead of guessing from memory.
"""
from __future__ import annotations

from .bundle.indexer import SECTION_TITLES
from .bundle.store import BundleStore
from .hotcache import HotCache


def render_seed(store: BundleStore, hot: HotCache | None = None) -> str:
    sections: dict[str, list[str]] = {}
    for concept in store.iter_concepts():
        section = concept.path.split("/")[0]
        label = concept.meta.title or concept.concept_id.split("/")[-1]
        if concept.meta.description:
            label += f" ({concept.meta.description})"
        sections.setdefault(section, []).append(label)

    lines = [
        "This is the user's shared long-term memory (all AI tools share it).",
        "ALWAYS query it first via memory_query before assuming or asking for",
        "knowledge about the user, their projects or their infrastructure.",
        "Store new permanent facts via memory_add (the librarian files them).",
        "",
        "Current corpus:",
    ]
    for section, items in sorted(sections.items()):
        title = SECTION_TITLES.get(section, section)
        preview = "; ".join(items[:8])
        more = f" ... +{len(items) - 8} more" if len(items) > 8 else ""
        lines.append(f"- {title} ({len(items)}): {preview}{more}")
    if not sections:
        lines.append("- (still empty - store noteworthy things via memory_add)")
    if hot:
        recent = hot.recent(6)
        if recent:
            lines.append("")
            lines.append("Recently stored/used:")
            for r in recent:
                summary = f" - {r['summary']}" if r["summary"] else ""
                lines.append(f"- {r['path']}{summary}")
    return "\n".join(lines)
