"""FastMCP server: streamable HTTP on /mcp + REST /api + /health.

Auth: static bearer key per client (CLIENT_KEYS in .env). The client name
ends up in traces and git commit messages. /health is reachable without auth.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
from pathlib import Path

import anyio
import uvicorn
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import get_settings
from .ops import MemoryOps, build_ops
from .seed import render_seed

current_client: contextvars.ContextVar[str] = contextvars.ContextVar("client", default="unknown")

# Single writer: all mutations go through this lock serially
_write_lock = asyncio.Lock()

settings = get_settings()
ops: MemoryOps = build_ops(settings)

mcp = FastMCP(
    name="librarian-memory",
    instructions=render_seed(ops.store, ops.hot),
)


def refresh_seed() -> None:
    """Refresh instructions - new MCP sessions get the fresh seed."""
    seed = render_seed(ops.store, ops.hot)
    mcp.instructions = seed
    # FastMCP mirrors instructions into the low-level server; set both defensively
    try:
        mcp._mcp_server.instructions = seed  # noqa: SLF001
    except AttributeError:
        pass


# ---------------- MCP tools ----------------

@mcp.tool(
    description=(
        "Query the user's long-term memory (projects, people, infra, "
        "preferences, facts). ALWAYS look here first before assuming knowledge "
        "about the user or asking them again."
    )
)
async def memory_query(question: str) -> str:
    client = current_client.get()
    result = await anyio.to_thread.run_sync(ops.query, question, client)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(
    description=(
        "Store a fact worth remembering permanently. The librarian files it "
        "itself (extends existing concepts or creates new ones) - "
        "just pass the fact in 1-3 sentences."
    )
)
async def memory_add(fact: str) -> str:
    client = current_client.get()
    try:
        async with _write_lock:
            result = await anyio.to_thread.run_sync(ops.add, fact, client)
    except (ValueError, RuntimeError) as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    refresh_seed()
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(
    description=(
        "Report a CHANGED fact (e.g. move, new IP, changed configuration). "
        "The librarian replaces the old fact everywhere in memory - the old "
        "state disappears completely. Use memory_add for new facts."
    )
)
async def memory_update(correction: str) -> str:
    client = current_client.get()
    try:
        async with _write_lock:
            result = await anyio.to_thread.run_sync(ops.update, correction, client)
    except (ValueError, RuntimeError) as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    refresh_seed()
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(
    description=(
        "Memory maintenance: lint check, deterministic repairs (indexes, "
        "backlinks), wire up orphaned concepts, put duplicate candidates in the "
        "review queue. Returns a report."
    )
)
async def memory_maintain() -> str:
    client = current_client.get()
    async with _write_lock:
        result = await anyio.to_thread.run_sync(ops.maintain, client)
    refresh_seed()
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(
    description="State of the memory: corpus per section, lint report, fresh seed overview."
)
async def memory_status() -> str:
    client = current_client.get()
    result = await anyio.to_thread.run_sync(ops.status, client)
    result["seed"] = await anyio.to_thread.run_sync(render_seed, ops.store, ops.hot)
    return json.dumps(result, ensure_ascii=False)


# ---------------- REST (for deterministic scripts, e.g. cron jobs) ----------------

@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


@mcp.custom_route("/api/query", methods=["GET"])
async def api_query(request: Request) -> JSONResponse:
    q = request.query_params.get("q", "")
    result = await anyio.to_thread.run_sync(ops.query, q, current_client.get())
    return JSONResponse(result)


@mcp.custom_route("/api/add", methods=["POST"])
async def api_add(request: Request) -> JSONResponse:
    payload = await request.json()
    fact = payload.get("fact", "")
    try:
        async with _write_lock:
            result = await anyio.to_thread.run_sync(ops.add, fact, current_client.get())
    except (ValueError, RuntimeError) as e:
        # RuntimeError = lint gate rolled back - clean error instead of 500
        return JSONResponse({"error": str(e)}, status_code=409)
    refresh_seed()
    return JSONResponse(result)


@mcp.custom_route("/api/update", methods=["POST"])
async def api_update(request: Request) -> JSONResponse:
    payload = await request.json()
    try:
        async with _write_lock:
            result = await anyio.to_thread.run_sync(
                ops.update, payload.get("correction", ""), current_client.get())
    except (ValueError, RuntimeError) as e:
        return JSONResponse({"error": str(e)}, status_code=409)
    refresh_seed()
    return JSONResponse(result)


@mcp.custom_route("/api/maintain", methods=["POST"])
async def api_maintain(request: Request) -> JSONResponse:
    async with _write_lock:
        result = await anyio.to_thread.run_sync(ops.maintain, current_client.get())
    return JSONResponse(result)


# ---------------- Web UI (static; data comes via the authenticated /api routes) ----------------

@mcp.custom_route("/ui", methods=["GET"])
async def ui(_: Request) -> HTMLResponse:
    html = (Path(__file__).parent / "webui" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@mcp.custom_route("/api/graph", methods=["GET"])
async def api_graph(_: Request) -> JSONResponse:
    return JSONResponse(await anyio.to_thread.run_sync(ops.fts.graph))


@mcp.custom_route("/api/concept", methods=["GET"])
async def api_concept(request: Request) -> JSONResponse:
    path = request.query_params.get("path", "")
    try:
        concept = await anyio.to_thread.run_sync(ops.store.read_concept, path)
    except Exception as e:  # noqa: BLE001 - broken/missing path -> 404 instead of 500
        return JSONResponse({"error": str(e)}, status_code=404)
    return JSONResponse({"path": concept.path, "meta": concept.meta.model_dump(),
                         "body": concept.body})


@mcp.custom_route("/api/doc", methods=["GET"])
async def api_doc(request: Request) -> JSONResponse:
    """Reference shelf: read one doc file - raw + rendered HTML."""
    path = request.query_params.get("path", "")
    try:
        text = await anyio.to_thread.run_sync(ops.docs.read_text, path)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=404)
    import markdown
    _, content = ops.docs.read_split(path)  # don't render the frontmatter
    html = await anyio.to_thread.run_sync(
        lambda: markdown.markdown(
            content, extensions=["tables", "fenced_code", "sane_lists", "toc"],
            extension_configs={"toc": {"permalink": False}}))
    return JSONResponse({"path": path, "title": ops.docs.title_of(path),
                         "body": text, "html": html})


@mcp.custom_route("/api/docs", methods=["GET"])
async def api_docs(request: Request) -> JSONResponse:
    """Reference shelf: list of all doc files (+ optional FTS search via ?q=)."""
    q = request.query_params.get("q", "")
    if q:
        hits = await anyio.to_thread.run_sync(ops.fts.search, q, 20)
        return JSONResponse([{"path": h.path, "title": h.title}
                             for h in hits if h.path.startswith("doc:")])
    paths = await anyio.to_thread.run_sync(ops.docs.list_paths)
    return JSONResponse([{"path": p, "title": ops.docs.title_of(p)} for p in paths])


@mcp.custom_route("/api/deps", methods=["GET"])
async def api_deps(_: Request) -> JSONResponse:
    """Service dependency graph from the doc frontmatter."""
    # Refresh the docs index when the fingerprint changed - otherwise the web UI
    # shows stale graphs after a git pull until a query happens to reindex.
    await anyio.to_thread.run_sync(ops.fts.ensure_docs_fresh, ops.docs)
    return JSONResponse(await anyio.to_thread.run_sync(ops.fts.deps_graph))


@mcp.custom_route("/api/traces", methods=["GET"])
async def api_traces(_: Request) -> JSONResponse:
    return JSONResponse(await anyio.to_thread.run_sync(ops.fts.traces_recent))


@mcp.custom_route("/api/reviews", methods=["GET"])
async def api_reviews(_: Request) -> JSONResponse:
    return JSONResponse(await anyio.to_thread.run_sync(ops.fts.open_reviews))


@mcp.custom_route("/api/review", methods=["POST"])
async def api_review(request: Request) -> JSONResponse:
    """Review decision: status done/dismissed - or action=merge (executes it)."""
    payload = await request.json()
    review_id = int(payload.get("id", 0))
    if payload.get("action") == "merge":
        review = next((r for r in ops.fts.open_reviews() if r["id"] == review_id), None)
        if not review or review["kind"] != "duplicate":
            return JSONResponse({"error": "no open duplicate review"}, status_code=400)
        detail = json.loads(review["detail"])
        try:
            async with _write_lock:
                result = await anyio.to_thread.run_sync(
                    ops.merge, detail["path"], detail["other"], current_client.get())
        except (ValueError, RuntimeError) as e:
            return JSONResponse({"error": str(e)}, status_code=409)
        ops.fts.review_set_status(review_id, "merged")
        refresh_seed()
        return JSONResponse(result)
    ok = await anyio.to_thread.run_sync(
        ops.fts.review_set_status, review_id,
        str(payload.get("status", "dismissed")))
    return JSONResponse({"ok": ok})


# ---------------- Auth middleware ----------------

class BearerAuth:
    """Checks Authorization: Bearer <key> against CLIENT_KEYS; /health stays open."""

    def __init__(self, app: ASGIApp, keys: dict[str, str]):
        self.app = app
        self.keys = keys  # key -> client-name

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # /ui is static HTML without data - it fetches all data via /api with a key
        if scope["type"] != "http" or scope.get("path") in ("/health", "/ui"):
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        auth = headers.get(b"authorization", b"").decode()
        token = auth.removeprefix("Bearer ").strip()
        name = self.keys.get(token)
        if not name:
            response = JSONResponse({"error": "unauthorized"}, status_code=401)
            await response(scope, receive, send)
            return
        current_client.set(name)
        await self.app(scope, receive, send)


def main() -> None:
    keys = settings.env.parsed_client_keys()
    if not keys:
        raise SystemExit("CLIENT_KEYS is empty - the server does not start without auth.")
    app = BearerAuth(mcp.http_app(path="/mcp"), keys)
    uvicorn.run(app, host=settings.cfg.server_host, port=settings.cfg.server_port)


if __name__ == "__main__":
    main()
