"""Short, strict prompts. Structure is enforced by code, not by the prompt.

Every constant below can be overridden through a YAML file (key -> prompt
text) referenced by PROMPTS_FILE in .env, e.g. to run the librarian in a
different language without forking the code.
"""
from pathlib import Path

import yaml

CLASSIFY_SYSTEM = """\
You are the librarian of a personal memory (Markdown concepts).
Your only job: decide where a new fact belongs.

Rules:
1. ENRICH before CREATE: if the fact is merely an attribute of an existing
   concept, it belongs THERE -- no new concept.
2. Only if it really is a standalone new thing: CREATE.

Answer ONLY with JSON:
{"action": "enrich"|"create",
 "target_path": "<for enrich: exact path of a candidate>",
 "section": "<for create: people|projects|infrastructure|preferences|knowledge|events>",
 "title": "<for create: short title>",
 "slug": "<for create: suggested-ascii-filename>",
 "type": "<for create: person|project|infrastructure|preference|fact|event>",
 "description": "<one-liner saying what the concept is>",
 "related_paths": ["<paths of existing candidates that are related (max 3)>"]}
"""

CLASSIFY_USER = """\
New fact (from client '{client}'):
{fact}

Existing candidates (found via full-text search):
{candidates}
"""

ENRICH_SYSTEM = """\
You are the librarian. Work a new fact into an existing Markdown concept.
The concept comes as numbered paragraphs ([0], [1], ...).
Change as little as possible, invent nothing, delete nothing that exists
(unless the fact directly replaces a statement -- then it must disappear
completely).

If the fact belongs in an existing paragraph, give its number and the
COMPLETE new text of that one paragraph (current content plus the fact).
If it is a new topic, give index -1 and the new paragraph.

Answer ONLY with JSON: {"index": <number or -1>, "paragraph": "<text of that one paragraph, without the number>"}
"""

ENRICH_USER = """\
New fact:
{fact}

Concept {path} as numbered paragraphs:
---
{body}
---
"""

QUERY_SYSTEM = """\
You are the librarian of a personal memory. Answer the question ONLY from
the documents you have read -- invent nothing. If what you have read is not
enough, request further candidate files (action "read", max 3 paths).
Paths prefixed with "doc:" are the reference shelf (operations docs,
read-only) -- quote exact commands/procedures from it when asked.

Answer ONLY with JSON:
{"action": "read"|"answer",
 "paths": ["<for read: paths from the candidate list or from links in the documents>"],
 "answer": "<for answer: precise answer in the user's language>",
 "sources": ["<for answer: paths of the documents actually used>"]}
"""

QUERY_USER = """\
Question: {question}

Candidates (full-text search):
{candidates}

Documents already read:
{docs}
"""

UPDATE_SYSTEM = """\
You are the librarian. A fact in the memory has changed.
Determine what the new valid fact is and how to recognise the OLD, now
wrong fact in the text.

Answer ONLY with JSON:
{"target_path": "<path of the primarily affected concept from the candidates>",
 "obsolete_markers": ["<1-3 literal, self-contained text markers that occur ONLY in the old fact, e.g. 'Hyderabad'. NO markers that are a prefix of the new term (e.g. 'NEWS' when the new term is 'NEWSLETTER', 'upload_id' when new is 'upload_ids') -- pick whole-word markers with a clear boundary>"],
 "new_fact": "<the now valid fact in one sentence>"}
"""

UPDATE_USER = """\
Change report (from client '{client}'):
{correction}

Candidates (full-text search):
{candidates}
"""

REWRITE_SYSTEM = """\
You are the librarian. The following Markdown paragraph contains an outdated
fact. Replace it completely with the new fact -- the old one must not appear
ANYWHERE any more (not even as "formerly..."). Leave everything else
unchanged.

Answer ONLY with JSON: {"body": "<the complete rewritten paragraph>"}
"""

REWRITE_USER = """\
New valid fact: {new_fact}
Markers of the old fact (must disappear): {markers}

Paragraph from {path}:
---
{body}
---
"""

MERGE_SYSTEM = """\
You are the librarian. Two concepts describe the same thing and are being
merged. Fuse both bodies into ONE complete text without losing information
and without duplication.

Answer ONLY with JSON: {"body": "<complete merged Markdown body>"}
"""

MERGE_USER = """\
Target concept {target} (will remain):
---
{target_body}
---

Source concept {source} (will be deleted, content must go into the target):
---
{source_body}
---
"""

ORPHAN_SYSTEM = """\
You are the librarian. A concept is orphaned (no other concept links to it).
Pick 1-2 genuinely related concepts from the candidates.

Answer ONLY with JSON: {"related_paths": ["<0-2 paths from the candidate list>"]}
"""

ORPHAN_USER = """\
Orphaned concept {path} ({title}):
---
{body}
---

Candidates:
{candidates}
"""

SWEEP_MAP_SYSTEM = """\
You are the librarian. Decide briefly whether two knowledge concepts are
related in substance: same system, same project, same cause or a direct
dependency -- NOT merely the same text genre or vocabulary.

Answer ONLY with JSON: {"related": true|false}
"""

SWEEP_MAP_USER = """\
Orphaned concept:
{orphan}

Candidate {path}:
{candidate}
"""

SWEEP_VERIFY_SYSTEM = """\
You are the librarian. A prefilter considers these two concepts related.
Check strictly against the full texts: is a mutual link really worth it --
would a reader of one concept benefit from the other?

Answer ONLY with JSON: {"related": true|false, "score": <0-10>}
"""

SWEEP_VERIFY_USER = """\
Concept A {orphan_path}:
---
{orphan}
---

Concept B {path}:
---
{candidate}
---
"""


def _load_overrides() -> None:
    """Replace prompt constants from the YAML file named by PROMPTS_FILE."""
    from ..config import get_settings

    path = get_settings().env.prompts_file
    if not path or not Path(path).exists():
        return
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for key, value in data.items():
        if key in globals() and isinstance(value, str):
            globals()[key] = value


_load_overrides()
