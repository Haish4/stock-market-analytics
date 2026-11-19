"""Optional Telegram notifications (monitoring bonus).

Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID as environment variables. Without
them every call is a silent no-op, so the pipeline never depends on Telegram.
"""
from __future__ import annotations

import os

import requests


def send(text: str) -> bool:
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": text}, timeout=30)
    return r.ok
