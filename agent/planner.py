"""
Plan the outing. Python picks the time window (best mean score), Gemma (Ollama,
local) picks the place and writes the nudge as strict JSON, validated against
that window and the place list. One retry, then agent/fallback.py.

    python -m agent.planner

Env: WINDOW_MODEL (default gemma3:4b), OLLAMA_HOST (default http://localhost:11434).
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Callable

import requests
from dotenv import load_dotenv

from agent import fallback, tools

log = logging.getLogger("planner")

KEYS = ("start", "end", "place", "activity", "message", "reason")
SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "string"} for k in KEYS},
    "required": list(KEYS),
}
SYSTEM = (
    "You are Window, a terse assistant that tells one person the best time to go outside in "
    "Hyderabad today. Use only the data given. Reply with JSON only."
)


def ollama_chat(messages: list[dict], timeout: int = 120) -> str:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    r = requests.post(f"{host}/api/chat", timeout=timeout, json={
        "model": os.getenv("WINDOW_MODEL", "gemma3:4b"), "messages": messages, "stream": False,
        "format": SCHEMA, "options": {"temperature": 0.2},
    })
    r.raise_for_status()
    return r.json()["message"]["content"]


@contextmanager
def stage(name: str, timings: dict):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = round((time.perf_counter() - t0) * 1000)
        log.info("stage %-9s %6d ms", name, timings[name])


def build_messages(forecast: dict, window: dict, places: list[dict]) -> list[dict]:
    data = {
        "day": forecast["day"], "chosen_window": window,
        "hours": [{k: h[k] for k in ("time", "score", "feels_c", "aqi", "air", "rain_pct", "uv")} for h in forecast["hours"]],
        "places": [{k: p[k] for k in ("name", "type", "shade", "km")} for p in places],
    }
    user = (
        f"DATA:\n{json.dumps(data)}\n\n"
        f"The window {window['start']}-{window['end']} is already chosen; copy start and end exactly. "
        "Pick ONE place, copying its name exactly from places (prefer shade when hot or high UV). "
        "activity: 2-4 words. "
        f"message: ONE short sentence of about 100 characters (hard limit 140), must say the day ('{forecast['day']}'), the time range and the place; "
        "use plain words (say 'feels like 27C', never field names like feels_c or aqi, and do not mention the score). "
        "Describe air quality only with the 'air' label from the data (e.g. 'air is moderate'); never call it good, clean or low-pollution unless the label is 'good'. "
        "reason: one sentence on why this window beats the rest of the day, comparing chosen_window with the other hours "
        "(heat, air quality, UV, rain) using numbers from the data, in plain words with no field names, and consistent with the 'air' label."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def validate(obj: object, window: dict, place_names: set[str]) -> list[str]:
    if not isinstance(obj, dict):
        return ["response is not a JSON object"]
    errs = [f"{k} must be a non-empty string" for k in KEYS
            if not isinstance(obj.get(k), str) or not obj[k].strip()]
    if errs:
        return errs
    if (obj["start"], obj["end"]) != (window["start"], window["end"]):
        errs.append(f"start/end must be exactly {window['start']} and {window['end']}")
    if obj["place"] not in place_names:
        errs.append(f"place must be one of {sorted(place_names)}")
    if len(obj["message"]) > fallback.MAX_MESSAGE:
        errs.append(f"message is {len(obj['message'])} chars, max {fallback.MAX_MESSAGE}")
    return errs


def plan(df=None, now=None, chat: Callable[[list[dict]], str] | None = None, backoff: float = 2.0) -> dict:
    """Returns {"plan", "source" ("ollama"|"fallback"), "day", "window", "latency_ms", "model"}."""
    chat = chat or ollama_chat
    timings: dict[str, int] = {}
    with stage("forecast", timings):
        forecast = tools.get_forecast(df, now)
        window = tools.best_window(forecast["hours"])
    with stage("places", timings):
        places = tools.get_places()
    names = {p["name"] for p in places}

    messages = build_messages(forecast, window, places)
    chosen = None
    for attempt in (1, 2):
        try:
            with stage(f"llm{attempt}", timings):
                raw = chat(messages)
        except (requests.RequestException, KeyError, ValueError) as e:
            log.warning("ollama call failed (attempt %d): %s", attempt, e)
            if attempt == 1:
                time.sleep(backoff)
            continue
        try:
            obj = json.loads(raw)
            errs = validate(obj, window, names)
        except json.JSONDecodeError:
            obj, errs = None, ["response was not valid JSON"]
        if not errs:
            chosen = {k: obj[k].strip() for k in KEYS}
            break
        log.warning("invalid plan (attempt %d): %s", attempt, "; ".join(errs))
        messages = messages + [{"role": "assistant", "content": raw},
                               {"role": "user", "content": "Invalid: " + "; ".join(errs) + ". Return corrected JSON only."}]

    source = "ollama"
    if chosen is None:
        source = "fallback"
        log.warning("using templated fallback")
        with stage("fallback", timings):
            chosen = fallback.build(window, places, forecast["day"])
    return {"plan": chosen, "source": source, "day": forecast["day"], "window": window,
            "latency_ms": timings, "model": os.getenv("WINDOW_MODEL", "gemma3:4b")}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()
    print(json.dumps(plan(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
