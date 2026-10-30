"""Model A/B for the librarian: compare two models on librarian decisions.

Runs inside the librarian container:
  docker exec librarian python /app/scripts/model_ab.py [--models model-a,model-b] [--quick]

Environment:
  LIBRARIAN_CONFIG  path to the config file (default /app/config.yaml)
  AB_MODELS         default value for --models (comma-separated model names)

Mechanical scoring, no judge LLM:
- classify: action as expected; for enrich the target_path must be a candidate
- rewrite (contradiction): stale marker gone, body not collapsed
"""
from __future__ import annotations

import argparse
import os
import time

from librarian.agent import prompts
from librarian.agent.llm import Llm, LlmUnavailable
from librarian.config import get_settings
from librarian.ops import ClassifyResult, EnrichResult

CONFIG_PATH = os.environ.get("LIBRARIAN_CONFIG", "/app/config.yaml")
DEFAULT_MODELS = os.environ.get("AB_MODELS", "librarian-qwen36,librarian-gemma4")

CANDIDATES = """\
- people/anna-schmidt.md [person] Anna Schmidt: neighbour and cat sitter
- infrastructure/router-fritzbox.md [infrastructure] FritzBox router: home network router with IP 192.0.2.1
- projects/garden-remodel.md [project] Garden remodel: redesign of the garden in 2026
- knowledge/insurance.md [fact] Insurance: overview of current policies"""

CLASSIFY_CASES = [
    # (fact, expected_action, expected_target_if_enrich)
    ("Anna Schmidt has a new phone number: 0171-2223344.", "enrich", "people/anna-schmidt.md"),
    ("The FritzBox now runs firmware version 8.2.", "enrich", "infrastructure/router-fritzbox.md"),
    ("The garden remodel now includes a pond after all.", "enrich", "projects/garden-remodel.md"),
    ("The household contents insurance was cancelled as of January.", "enrich", "knowledge/insurance.md"),
    ("The user bought a new Cube e-bike.", "create", None),
    ("There is a new LAN server called 'backup' with IP 192.0.2.99.", "create", None),
    ("The neighbour Klaus Meier regularly borrows tools.", "create", None),
    ("New hobby: the user has started archery.", "create", None),
]

REWRITE_CASES = [
    # (new_fact, markers, body)
    ("Anna now lives at 12 Station Road.", ["Garden Way"],
     "Anna Schmidt is a neighbour.\nLives at 5 Garden Way.\nFeeds the cats during holidays.\n"),
    ("The FritzBox now has the IP 192.0.2.254.", ["192.0.2.1"],
     "Home network router.\nIP: 192.0.2.1.\nWiFi on channel 6.\n"),
    ("The garden remodel is finished.", ["planned for summer"],
     "Redesign of the garden, planned for summer 2026.\nBudget 5000 euros.\n"),
    ("Contents insurance is now with Allianz.", ["HUK"],
     "Contents: HUK, policy 12345.\nLiability: ARAG.\n"),
]


def run(models: list[str], quick: bool) -> None:
    settings = get_settings(CONFIG_PATH)
    llm = Llm(settings)
    classify = CLASSIFY_CASES[:4] if quick else CLASSIFY_CASES
    rewrite = REWRITE_CASES[:2] if quick else REWRITE_CASES

    for model in models:
        ok = 0
        total = 0
        latencies: list[float] = []
        fails: list[str] = []

        for fact, expected, target in classify:
            total += 1
            t0 = time.monotonic()
            try:
                r = llm.chat_json(
                    prompts.CLASSIFY_SYSTEM,
                    prompts.CLASSIFY_USER.format(client="ab-test", fact=fact,
                                                 candidates=CANDIDATES),
                    ClassifyResult, model=model)
                latencies.append(time.monotonic() - t0)
                if r.action != expected:
                    fails.append(f"classify '{fact[:40]}': {r.action} != {expected}")
                elif expected == "enrich" and r.target_path.lstrip("/") != target:
                    fails.append(f"classify '{fact[:40]}': target {r.target_path} != {target}")
                else:
                    ok += 1
            except LlmUnavailable as e:
                latencies.append(time.monotonic() - t0)
                fails.append(f"classify '{fact[:40]}': {str(e)[:80]}")

        for new_fact, markers, body in rewrite:
            total += 1
            t0 = time.monotonic()
            try:
                r = llm.chat_json(
                    prompts.REWRITE_SYSTEM,
                    prompts.REWRITE_USER.format(new_fact=new_fact,
                                                markers=", ".join(markers),
                                                path="test.md", body=body),
                    EnrichResult, model=model)
                latencies.append(time.monotonic() - t0)
                if any(m.lower() in r.body.lower() for m in markers):
                    fails.append(f"rewrite '{new_fact[:40]}': marker survived")
                elif len(r.body.strip()) < len(body) // 3:
                    fails.append(f"rewrite '{new_fact[:40]}': body collapsed")
                else:
                    ok += 1
            except LlmUnavailable as e:
                latencies.append(time.monotonic() - t0)
                fails.append(f"rewrite '{new_fact[:40]}': {str(e)[:80]}")

        avg = sum(latencies) / len(latencies) if latencies else 0
        print(f"\n=== {model}: {ok}/{total} ok, avg {avg:.1f}s/call")
        for f in fails:
            print(f"   FAIL {f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run(args.models.split(","), args.quick)
