"""
Send the nudge to Telegram. Token / chat id come from .env
(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID).

    python -m notify.telegram --find-chat-id    # after pressing Start on the bot; saves it to .env
    python -m notify.telegram "test message"
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

log = logging.getLogger("notify")
API = "https://api.telegram.org/bot{token}/{method}"
ENV = Path(__file__).resolve().parent.parent / ".env"


class NotifyError(RuntimeError):
    pass


def _creds(chat_id: str | None = None) -> tuple[str, str]:
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        raise NotifyError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in .env "
                          "(run `python -m notify.telegram --find-chat-id`)")
    return token, chat


def send(text: str, retries: int = 3, timeout: int = 15, backoff: float = 2.0, chat_id: str | None = None) -> None:
    """Send a plain-text message. Retries network errors, 429 (honouring retry_after)
    and 5xx with exponential backoff; other 4xx fail immediately. Raises NotifyError."""
    token, chat = _creds(chat_id)
    url = API.format(token=token, method="sendMessage")
    last = "unknown error"
    for attempt in range(retries):
        wait = backoff * 2 ** attempt
        try:
            r = requests.post(url, json={"chat_id": chat, "text": text}, timeout=timeout)
            if r.status_code == 429:
                wait = float(r.json().get("parameters", {}).get("retry_after", wait))
                last = "HTTP 429 rate limited"
            elif r.status_code >= 500:
                last = f"HTTP {r.status_code}"
            elif r.ok:
                log.info("telegram message sent (%d chars)", len(text))
                return
            else:
                raise NotifyError(f"Telegram rejected the message: {r.status_code} {r.json().get('description', '')}")
        except requests.RequestException as e:
            # never log the URL: it contains the bot token
            last = type(e).__name__
        log.warning("telegram send failed (%s), retry %d/%d in %.0fs", last, attempt + 1, retries, wait)
        if attempt < retries - 1:
            time.sleep(wait)
    raise NotifyError(f"Telegram unreachable after {retries} attempts ({last})")


def find_chat_id(timeout: int = 15) -> str:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise NotifyError("TELEGRAM_BOT_TOKEN not set")
    r = requests.get(API.format(token=token, method="getUpdates"), timeout=timeout)
    r.raise_for_status()
    chats = [u["message"]["chat"] for u in r.json().get("result", []) if "message" in u]
    if not chats:
        raise NotifyError("no messages yet: open the bot in Telegram, press Start, then retry")
    return str(chats[-1]["id"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("text", nargs="?", default="Window test message")
    p.add_argument("--find-chat-id", action="store_true")
    a = p.parse_args()
    if a.find_chat_id:
        chat = find_chat_id()
        if "TELEGRAM_CHAT_ID" not in (ENV.read_text() if ENV.exists() else ""):
            with ENV.open("a") as f:
                f.write(f"TELEGRAM_CHAT_ID={chat}\n")
        print(f"chat id {chat} saved to .env")
        return
    send(a.text)


if __name__ == "__main__":
    main()
