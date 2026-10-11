"""Templated nudge used when Ollama is unreachable or returns invalid JSON."""
from __future__ import annotations

from agent.tools import CAUTION_WORDS, explain_window, pick_place

MAX_MESSAGE = 140


def build(window: dict, places: list[dict], day: str, hours: list[dict] | None = None) -> dict:
    place = pick_place(places, window)
    hot = window["avg_feels_c"] >= 30 or window["max_uv"] >= 6
    activity = "short shaded walk" if hot else "walk"
    when = day.capitalize()
    caution = window.get("caution", [])
    if caution:  # lead with the warning instead of an upbeat invitation
        adj = " and ".join(CAUTION_WORDS[c] for c in caution[:2])
        lead = f"Best of a {adj} day: {day} {window['start']}-{window['end']} at {place['name']}."
        tail = f" Air is {window['air']}; limit time outside." if "air" in caution else " Take it easy."
        message = lead + tail
    else:
        tail = f" at {place['name']}. Feels {window['avg_feels_c']:.0f}C, air {window['air']} (AQI {window['max_aqi']}), rain {window['max_rain_pct']:.0f}%."
        message = f"{when} {window['start']}-{window['end']}{tail}"
    if len(message) > MAX_MESSAGE:
        message = message[:MAX_MESSAGE - 1].rstrip() + "…"
    return {
        "start": window["start"], "end": window["end"], "place": place["name"], "activity": activity,
        "message": message,
        "reason": explain_window(window, hours) if hours else
                  f"Highest outdoor score of the day ({window['score']}/100).",
    }
