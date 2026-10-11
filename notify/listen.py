"""
Answer messages from the owner on Telegram, so the bot is not a mute notifier.

    python -m notify.listen          # long-polls until Ctrl-C

Commands: /today (the message), /why (reason and details), /help. Anything else gets the
help text. Only the chat in TELEGRAM_CHAT_ID is answered; every other chat is ignored,
so strangers who find the bot cannot trigger planning. Run one listener at a time
(Telegram allows one getUpdates consumer per bot).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable

import requests
from dotenv import load_dotenv

from notify import telegram
from window import today

log = logging.getLogger("listen")

HELP = ("I'm Window. Every morning at 05:30 I send the best time to go outside in Hyderabad.\n\n"
        "/today  the current recommendation\n/why  why that window, with the numbers\n/help  this message")


def reply_for(text: str, get_plan: Callable[[], dict]) -> str:
    cmd = (text or "").strip().split()[0].split("@")[0].lower() if (text or "").strip() else ""
    if cmd in ("/start", "/help"):
        return HELP
    if cmd not in ("/today", "/now", "/why"):
        return "Send /today for the recommendation, or /help."
    try:
        r = get_plan()
    except Exception as e:  # never go silent on the owner
        log.exception("could not get a plan")
        return f"I could not build a plan right now ({type(e).__name__}). Try again in a minute."
    p, w = r["plan"], r["window"]
    if cmd == "/why":
        return f"{p['activity'].capitalize()} at {p['place']}, {p['start']}-{p['end']} ({r['day']}).\n{p['reason']}"
    return p["message"]


def handle(update: dict, owner: str, get_plan: Callable[[], dict]) -> tuple[str, str] | None:
    """(chat_id, reply) for an update from the owner, or None to stay silent."""
    msg = update.get("message")
    if not msg or "text" not in msg:
        return None
    chat = str(msg["chat"]["id"])
    if chat != str(owner):
        log.warning("ignoring message from another chat")
        return None
    return chat, reply_for(msg["text"], get_plan)


def _updates(token: str, offset: int | None, timeout: int) -> list[dict]:
    params = {"timeout": timeout, "allowed_updates": '["message"]'}
    if offset is not None:
        params["offset"] = offset
    r = requests.get(telegram.API.format(token=token, method="getUpdates"), params=params, timeout=timeout + 15)
    r.raise_for_status()
    return r.json()["result"]


def run(poll_seconds: int = 25) -> None:
    token, owner = os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]
    backlog = _updates(token, None, 0)  # skip anything sent while the listener was off
    offset = backlog[-1]["update_id"] + 1 if backlog else None
    log.info("listening (skipped %d old messages); Ctrl-C to stop", len(backlog))
    failures = 0
    while True:
        try:
            updates = _updates(token, offset, poll_seconds)
            failures = 0
        except (requests.RequestException, KeyError, ValueError) as e:
            failures += 1
            wait = min(60, 2 ** failures)
            log.warning("getUpdates failed (%s), retry in %ds", type(e).__name__, wait)  # no URL: it holds the token
            time.sleep(wait)
            continue
        for u in updates:
            offset = u["update_id"] + 1
            t0 = time.time()
            out = handle(u, owner, today.load_plan)
            if out:
                try:
                    telegram.send(out[1], chat_id=out[0])
                except telegram.NotifyError as e:
                    log.error("reply failed: %s", e)
                    continue
                sent = u["message"].get("date", t0)
                log.info("answered %r: %.1fs after it was sent, %.1fs to build and send the reply",
                         u["message"]["text"][:20], t0 - sent, time.time() - t0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()
    try:
        run()
    except KeyboardInterrupt:
        log.info("stopped")


if __name__ == "__main__":
    main()
