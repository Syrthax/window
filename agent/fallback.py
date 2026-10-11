"""Templated nudge used when Ollama is unreachable or returns invalid JSON."""
from __future__ import annotations

from agent.tools import pick_place

MAX_MESSAGE = 140


def build(window: dict, places: list[dict], day: str) -> dict:
    place = pick_place(places, window)
    hot = window["avg_feels_c"] >= 30 or window["max_uv"] >= 6
    activity = "short shaded walk" if hot else "walk"
    when = day.capitalize()
    tail = f" at {place['name']}. Feels {window['avg_feels_c']:.0f}C, air {window['air']} (AQI {window['max_aqi']}), rain {window['max_rain_pct']:.0f}%."
    message = f"{when} {window['start']}-{window['end']}{tail}"
    if len(message) > MAX_MESSAGE:
        message = message[:MAX_MESSAGE - 1].rstrip() + "…"
    return {
        "start": window["start"], "end": window["end"], "place": place["name"], "activity": activity,
        "message": message,
        "reason": f"Highest outdoor score of the day ({window['score']}/100): feels {window['avg_feels_c']:.0f}C, "
                  f"AQI up to {window['max_aqi']}, UV up to {window['max_uv']}, rain chance up to {window['max_rain_pct']:.0f}%.",
    }
