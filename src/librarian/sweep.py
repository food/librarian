"""Full-corpus sweep: orphan wiring as a fixed map-reduce-verify pipeline.

Background (RLM pilot 2026-08-31, projects/rlm-experiments.md in the bundle):
The FTS prefilter in maintain() only sees candidates with word overlap -
semantically related concepts without shared keywords stay invisible forever.
A free LLM orchestrator (REPL root per the RLM paper) was unstable in the pilot,
so the pipeline is hard-coded here and the LLM makes ONLY
two narrowly scoped yes/no decisions:

  1. Map:    one cheap call per concept - "related to the orphan?"
             (title/description/start of body only, small context)
  2. Reduce: Python collects the yes candidates - no LLM
  3. Verify: one confirmation call per candidate with the full texts;
             only doubly confirmed ones are wired (max. 2, like maintain)

Cost: one map call per concept in the bundle (~2-5 s depending on GPU load) -
a deliberate night/batch job, not a server feature. Hence a
separate CLI instead of building it into maintain(); the dream timer stays untouched.

Usage in the container:  python -m librarian.sweep [orphan-path ...] [--apply]
Without paths: all orphans per lint (max. --max, default 3).
Default is DRY-RUN (report only), as with dreaming.
"""
from __future__ import annotations

import argparse
import os
import time

from pydantic import BaseModel

from .agent import prompts
from .agent.llm import LlmUnavailable
from .bundle.lint import run_lint
from .config import get_settings
from .ops import MemoryOps, build_ops

MAX_LINKS = 2            # like maintain: at most 2 links per orphan
MAX_RUNTIME_S = 30 * 60  # total budget, like dreaming


class RelatedVerdict(BaseModel):
    related: bool = False
    score: int = 0  # 0-10; only for ordering among the verify winners


def _head(concept) -> str:
    return (f"{concept.meta.title}: {concept.meta.description}\n"
            f"{concept.body[:300]}")


def sweep_orphan(ops: MemoryOps, orphan_path: str,
                 deadline: float | None = None) -> dict:
    """Map-reduce-verify for ONE orphan. Mutates nothing - candidate search only."""
    orphan = ops.store.read_concept(orphan_path)
    kw = {"model": ops.s.cfg.sweep_model} if ops.s.cfg.sweep_model else {}
    result = {"orphan": orphan_path, "scanned": 0, "map_hits": [],
              "confirmed": [], "aborted": ""}

    # Map: one small call per concept, whole corpus instead of FTS top-N
    for path in ops.store.list_concept_paths():
        if path == orphan_path or path.startswith("inbox/"):
            continue
        if deadline and time.monotonic() > deadline:
            result["aborted"] = "time budget in map"
            return result
        result["scanned"] += 1
        try:
            v = ops.llm.chat_json(
                prompts.SWEEP_MAP_SYSTEM,
                prompts.SWEEP_MAP_USER.format(
                    orphan=_head(orphan), path=path,
                    candidate=_head(ops.store.read_concept(path))),
                RelatedVerdict, **kw)
        except LlmUnavailable:
            continue  # one missed candidate doesn't sink the sweep
        if v.related:
            result["map_hits"].append(path)

    # Verify: full texts, stricter standard - the reduce step in between is pure Python
    scored: list[tuple[int, str]] = []
    for path in result["map_hits"]:
        if deadline and time.monotonic() > deadline:
            result["aborted"] = "time budget in verify"
            break
        try:
            v = ops.llm.chat_json(
                prompts.SWEEP_VERIFY_SYSTEM,
                prompts.SWEEP_VERIFY_USER.format(
                    orphan_path=orphan_path, orphan=orphan.body[:1500],
                    path=path, candidate=ops.store.read_concept(path).body[:1500]),
                RelatedVerdict, **kw)
        except LlmUnavailable:
            continue
        if v.related:
            scored.append((v.score, path))
    scored.sort(reverse=True)
    result["confirmed"] = [p for _, p in scored[:MAX_LINKS]]
    return result


def run(paths: list[str], apply: bool, max_orphans: int) -> dict:
    settings = get_settings(os.environ.get("LIBRARIAN_CONFIG", "config.yaml"))
    ops = build_ops(settings)
    if not paths:
        paths = [f.path for f in run_lint(ops.store).by_check("orphan")
                 if not f.path.startswith("inbox/")][:max_orphans]
    deadline = time.monotonic() + MAX_RUNTIME_S
    reports = []
    for p in paths:
        r = sweep_orphan(ops, p, deadline)
        if apply and r["confirmed"]:
            r["linked"] = ops.wire_orphan(p, r["confirmed"], client="sweep")
        reports.append(r)
    return {"mode": "apply" if apply else "dry-run", "reports": reports}


def main() -> None:
    ap = argparse.ArgumentParser(description="Full-corpus orphan sweep")
    ap.add_argument("paths", nargs="*", help="orphan paths; empty = all per lint")
    ap.add_argument("--apply", action="store_true", help="wire instead of only reporting")
    ap.add_argument("--max", type=int, default=3, help="max. orphans without explicit paths")
    args = ap.parse_args()

    result = run(args.paths, args.apply, args.max)
    print(f"Sweep ({result['mode']}):")
    for r in result["reports"]:
        status = f"scanned {r['scanned']}, map hits {len(r['map_hits'])}, " \
                 f"confirmed {r['confirmed']}"
        if r.get("linked"):
            status += f" -> linked {r['linked']}"
        if r["aborted"]:
            status += f" [aborted: {r['aborted']}]"
        print(f" - {r['orphan']}: {status}")


if __name__ == "__main__":
    main()
