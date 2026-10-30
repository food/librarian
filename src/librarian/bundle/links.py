"""Link extraction and graph helpers.

When writing, the librarian uses bundle-absolute markdown links
[Title](/people/anna.md). When reading/linting, relative links and
[[wiki-links]] are recognized as well (robustness against manual edits/imports).
"""
from __future__ import annotations

import posixpath
import re

MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s#]+)(?:#[^)]*)?\)")
WIKI_LINK = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")


def extract_links(body: str, from_path: str) -> list[str]:
    """All bundle-internal link targets as normalized bundle-relative paths (with .md)."""
    targets: list[str] = []
    for m in MD_LINK.finditer(body):
        href = m.group(1)
        if "://" in href or href.startswith("mailto:"):
            continue  # external links are not graph edges
        if href.startswith("/"):
            norm = href.lstrip("/")
        else:
            base = posixpath.dirname(from_path)
            norm = posixpath.normpath(posixpath.join(base, href))
        if norm.startswith(".."):
            continue
        if not norm.endswith(".md"):
            norm += ".md"
        targets.append(norm)
    for m in WIKI_LINK.finditer(body):
        name = m.group(1).strip()
        norm = name.lstrip("/")
        if not norm.endswith(".md"):
            norm += ".md"
        targets.append(norm)
    # drop duplicates, keep order stable
    seen: set[str] = set()
    out: list[str] = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def resolve_wiki_target(name: str, all_paths: list[str]) -> str | None:
    """[[name]] without a directory: find the unique concept with a matching file name."""
    base = name.removesuffix(".md")
    matches = [p for p in all_paths if p.removesuffix(".md").split("/")[-1] == base]
    return matches[0] if len(matches) == 1 else None


def related_heading() -> str:
    """Heading of the backlink section. Configurable (RELATED_HEADING) so an
    existing bundle written with a different heading keeps parsing."""
    from ..config import get_settings
    return get_settings().cfg.related_heading


def append_backlink(body: str, target_path: str, target_title: str) -> str:
    """Add a backlink under the related heading (idempotent)."""
    heading = related_heading()
    link = f"[{target_title}](/{target_path})"
    if f"(/{target_path})" in body:
        return body
    if heading in body:
        return body.replace(heading, f"{heading}\n\n- {link}", 1)
    return body.rstrip() + f"\n\n{heading}\n\n- {link}\n"
