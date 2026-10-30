"""OKF v0.1: markdown + YAML frontmatter. Only required field: type.

Parser/serializer + pydantic schema. Unknown frontmatter keys are
tolerated and preserved on serialization (OKF compatibility).
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Any

import frontmatter
from pydantic import BaseModel, ConfigDict, Field, field_validator

# index.md / log.md are reserved per OKF and have NO frontmatter
RESERVED_FILES = {"index.md", "log.md"}


class ConceptMeta(BaseModel):
    """Frontmatter of a concept. OKF fields + two custom fields."""

    model_config = ConfigDict(extra="allow")

    type: str
    title: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    timestamp: str = ""  # ISO, set by the code on every mutation
    source: str = ""     # custom: which client contributed this
    aliases: list[str] = Field(default_factory=list)  # custom: duplicate matching

    @field_validator("type")
    @classmethod
    def type_not_empty(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError("frontmatter 'type' must not be empty")
        return str(v).strip()

    @field_validator("tags", "aliases", mode="before")
    @classmethod
    def coerce_list(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return [str(x) for x in v]


class Concept(BaseModel):
    """A parsed concept: path (bundle-relative, with .md), meta, body."""

    path: str
    meta: ConceptMeta
    body: str

    @property
    def concept_id(self) -> str:
        """OKF: concept ID = path without .md"""
        return self.path.removesuffix(".md")


def parse(path: str, text: str) -> Concept:
    post = frontmatter.loads(text)
    if not post.metadata:
        raise ValueError(f"{path}: no frontmatter found")
    meta = ConceptMeta(**post.metadata)
    return Concept(path=path, meta=meta, body=post.content.strip() + "\n")


def serialize(concept: Concept) -> str:
    meta = concept.meta.model_dump(exclude_defaults=False)
    # don't write out empty optional fields - files stay lean and human-readable
    for key in ("title", "description", "source", "timestamp"):
        if not meta.get(key):
            meta.pop(key, None)
    for key in ("tags", "aliases"):
        if not meta.get(key):
            meta.pop(key, None)
    post = frontmatter.Post(concept.body.strip() + "\n", **meta)
    return frontmatter.dumps(post, sort_keys=False) + "\n"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def slugify(text: str) -> str:
    """ASCII slug, lowercase, '-'-separated. The code enforces this, the LLM only suggests."""
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("ß", "ss")
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text or "unnamed"
