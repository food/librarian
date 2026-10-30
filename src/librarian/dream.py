"""Dreaming: nightly self-consolidation of the memory.

Flow (multiple safety nets):
1. Git tag `pre-dream-<date>` as restore point
2. Analysis (code, no LLM): lint, inbox content, orphans, duplicates
3. Budget: max. DREAM_MAX_OPS LLM operations, max. 30 min runtime
4. One commit per operation (`via dream`); lint gate like every mutation
5. Destructive actions (duplicate merge) are NOT executed but put in the
   review queue - a human decides (web UI)
6. Telegram summary if there is something to report

Modes: default is DRY-RUN (report only). Enable: env DREAM_APPLY=true
(.env + `docker compose up -d`) - deliberately only after the observation phase.

Usage: python -m librarian.dream [--apply | --dry-run]
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import date

from . import notify
from .agent import prompts
from .agent.llm import LlmUnavailable
from .bundle.lint import fix_deterministic, run_lint
from .config import get_settings
from .ops import ClassifyResult, MemoryOps, build_ops

MAX_RUNTIME_S = 30 * 60


def file_inbox_item(ops: MemoryOps, path: str) -> str | None:
    """File one inbox concept normally. None = skipped (LLM down)."""
    concept = ops.store.read_concept(path)
    fact = concept.body.strip()
    baseline = ops._lint_errors()  # noqa: SLF001 - dream is part of the package
    candidates = ops.fts.search(fact, ops.s.cfg.fts_candidates)
    cand_text = "\n".join(
        f"- {h.path} [{h.type}] {h.title}: {h.description}"
        for h in candidates if not h.path.startswith("inbox/")) or "(none)"
    try:
        decision = ops.llm.chat_json(
            prompts.CLASSIFY_SYSTEM,
            prompts.CLASSIFY_USER.format(client="dream", fact=fact, candidates=cand_text),
            ClassifyResult)
    except LlmUnavailable:
        return None
    # Deletion + refiling end up in the SAME commit (finalize commits everything)
    ops.store.delete(path)
    if decision.action == "enrich" and ops.store.exists(decision.target_path.lstrip("/")):
        r = ops._enrich(fact, "dream", decision.target_path.lstrip("/"), baseline)  # noqa: SLF001
    else:
        r = ops._create(fact, "dream", decision, baseline)  # noqa: SLF001
    return f"inbox {path} -> {r.get('path', '?')} ({r['result']})"


def run(apply: bool) -> dict:
    settings = get_settings(os.environ.get("LIBRARIAN_CONFIG", "config.yaml"))
    ops = build_ops(settings)
    max_ops = int(os.environ.get("DREAM_MAX_OPS", "10"))
    started = time.monotonic()
    used_ops = 0
    actions: list[str] = []
    errors: list[str] = []

    def budget_left() -> bool:
        return used_ops < max_ops and time.monotonic() - started < MAX_RUNTIME_S

    if apply:
        ops.git.tag(f"pre-dream-{date.today().isoformat()}")

    # --- Analysis (no LLM) ---
    report = run_lint(ops.store)
    inbox = [p for p in ops.store.list_concept_paths() if p.startswith("inbox/")]
    orphans = [f.path for f in report.by_check("orphan")
               if not f.path.startswith("inbox/")]
    duplicates = report.by_check("duplicate")
    plan = (f"Analysis: {report.summary()} | inbox: {len(inbox)} | "
            f"orphans: {len(orphans)} | duplicates: {len(duplicates)}")

    if not apply:
        planned = ([f"fix: {f.check} {f.path}" for f in report.findings
                    if f.check in ("backlink-asym", "index-missing", "index-ghost")]
                   + [f"file inbox: {p}" for p in inbox]
                   + [f"wire orphan: {p}" for p in orphans]
                   + [f"review-queue: duplicate {f.path}" for f in duplicates])
        return {"mode": "dry-run", "plan": plan, "would_do": planned,
                "actions": [], "errors": []}

    # --- 1. deterministic fixes (free, don't count toward the budget) ---
    try:
        fixes = fix_deterministic(ops.store, report)
        if fixes:
            head = ops.git.commit_all("dream: deterministic fixes")
            ops.fts.reindex(ops.store, head)
            actions += [f"dream-fix: {f}" for f in fixes]
    except Exception as e:  # noqa: BLE001
        ops.git.rollback_uncommitted()
        errors.append(f"fix_deterministic: {e}")

    # --- 2. file the inbox (LLM, budgeted) ---
    for path in inbox:
        if not budget_left():
            actions.append("Budget exhausted - rest tomorrow")
            break
        used_ops += 1
        try:
            result = file_inbox_item(ops, path)
            actions.append(result or f"inbox {path}: skipped (LLM)")
        except Exception as e:  # noqa: BLE001
            ops.git.rollback_uncommitted()
            errors.append(f"inbox {path}: {e}")

    # --- 3. Wire orphans + put duplicates in the review queue ---
    #     (maintain does exactly that, with its own commit per repair)
    if budget_left():
        try:
            m = ops.maintain(client="dream")
            used_ops += len([a for a in m["actions"] if a.startswith("orphan")])
            actions += [f"dream-{a}" for a in m["actions"]]
        except Exception as e:  # noqa: BLE001
            errors.append(f"maintain: {e}")

    final = run_lint(ops.store)
    return {"mode": "apply", "plan": plan, "actions": actions, "errors": errors,
            "lint_after": final.summary(),
            "review_open": len(ops.fts.open_reviews()), "ops_used": used_ops}


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    # Default comes from DREAM_APPLY (false = observation phase / dry-run)
    apply = args.apply or (not args.dry_run
                           and os.environ.get("DREAM_APPLY", "").lower() == "true")

    result = run(apply)
    print(result["plan"])
    for line in result.get("would_do", []) + result["actions"]:
        print(" -", line)
    for line in result["errors"]:
        print(" ! ERROR:", line)

    label = "Dreaming" if result["mode"] == "apply" else "Dreaming (DRY-RUN)"
    todo = result.get("would_do", [])
    if result["actions"] or result["errors"] or todo:
        lines = [f"{label} - {result['plan']}"]
        lines += [f"- {a}" for a in (result["actions"] or todo)[:12]]
        lines += [f"! {e}" for e in result["errors"][:5]]
        if result["mode"] == "apply":
            lines.append(f"After: {result['lint_after']}, "
                         f"{result['review_open']} open reviews")
        notify.send("\n".join(lines))

    raise SystemExit(1 if result["errors"] else 0)


if __name__ == "__main__":
    main()
