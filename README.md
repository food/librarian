# Librarian -- Local-first AI Memory

Shared long-term memory for all AI tools (Claude Code, Open WebUI, custom
agents), following the librarian pattern: a **library** (Markdown bundle per
[OKF v0.1](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md),
git-versioned) and a **librarian** (internal LLM agent) who is the only one
to touch the library. Clients speak MCP tools only -- the memory workload
lives in the tool, not in the client model. Design inspiration:
[Understory](https://github.com/thecodacus/understory).

## Architecture (short version)

- **MCP Streamable HTTP** on `:8765/mcp` (FastMCP), bearer key **per client**
  (`CLIENT_KEYS` in `.env`), REST under `/api/*` for deterministic scripts.
- **Determinism sandwich:** code -> (one narrowly scoped LLM call,
  JSON-validated) -> code. Structure (frontmatter, slugs, backlinks, indexes,
  log, lint, git commit) is enforced by code; the LLM only decides content.
  Every LLM step has a deterministic fallback (`inbox/`).
- **Loaded-model preference:** `librarian_models` is a prioritised candidate
  list. If one of them is already loaded according to Ollama `/api/ps`, it is
  used (no model swap). Calls go through LiteLLM (`litellm_model_map`), with
  `litellm_fallback_model` as the fallback lane.
- **SQLite FTS5** (`data/librarian.db`) as deterministic prefilter -- fully
  reconstructible from the bundle, needs no backup.
- **Bundle** (`memory-bundle/`, its own git repo): auto-commit per mutation.

## Setup

```bash
cp .env.example .env   # fill in keys (openssl rand -hex 24)
docker compose up -d --build
curl http://localhost:8765/health
```

Configuration lives in `.env` (secrets and deployment-specific values; any
`config.yaml` key can be overridden there, upper-case, lists as JSON) and
`config.yaml` (tuning defaults). The librarian's prompts can be swapped out
via `PROMPTS_FILE` -- e.g. to run the librarian in another language.

## Clients

```bash
# Claude Code (per machine):
claude mcp add --transport http memory http://<host>:8765/mcp \
  --header "Authorization: Bearer <KEY>"
```
- **Open WebUI:** Admin -> External Tools -> "MCP (Streamable HTTP)" -> URL + bearer key.
- Any other MCP client: Streamable HTTP endpoint + `Authorization: Bearer` header.

## MCP tools

| Tool | Purpose |
|---|---|
| `memory_query` | librarian loop: FTS prefilter -> traversal -> synthesis with sources (fallback: raw FTS hits) |
| `memory_add` | file a fact (enrich-before-create, link-both-ways) |
| `memory_update` | report a changed fact -- the stale fact disappears everywhere (grep-verified, else rollback) |
| `memory_maintain` | lint + deterministic fixes + orphan wiring; duplicates -> review queue |
| `memory_status` | corpus stats, lint report, fresh seed |

REST equivalents: `GET /api/query` - `POST /api/add` - `POST /api/update` - `POST /api/maintain`.

## Development

```bash
pip install -e .[dev]
pytest
```

Model A/B: `docker exec librarian python /app/scripts/model_ab.py [--quick]`

Full-corpus orphan sweep (map-reduce-verify instead of FTS top-6):
`docker exec librarian python -m librarian.sweep [path ...] [--apply]` --
dry-run by default; cheap map model via `sweep_model:` in config.yaml.

Features: core bundle ops - librarian loop / update / maintain - hot cache -
dreaming (nightly consolidation) + review merge - web UI (graph / traces /
review / docs) - reference shelf (read-only docs mirror with freshness check
per query).
