"""Telegram notification (deterministic, bot API directly - no LLM).

Usage: python -m librarian.notify "message"
Env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (in .env; if missing, silently
skipped - dreaming/backup work without Telegram too).
"""
from __future__ import annotations

import os
import sys

import httpx


def send(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False
    r = httpx.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text[:4000]},
        timeout=20,
    )
    r.raise_for_status()
    return True


if __name__ == "__main__":
    ok = send(" ".join(sys.argv[1:]) or "librarian: (empty message)")
    print("sent" if ok else "skipped (no token/chat configured)")
