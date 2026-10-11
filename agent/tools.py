"""
Tools the planner uses: `get_forecast()` and `get_places()`, plus the
deterministic window picker. Gemma never chooses *when*; it chooses where and
how to say it, so a small model cannot invent a bad time.

Gemma 3 has no native tool calling in Ollama, so the planner calls these
functions itself and puts the results in the prompt.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pandas as pd

from model.features import DATA
from model.score import pm25_to_aqi, score_frame

TZ = "Asia/Kolkata"
PLACES = DATA / "places.hyd.json"


def home() -> tuple[float, float]:
    """Where 'nearby' is measured from (central Hyderabad unless overridden in .env)."""
    return float(os.getenv("WINDOW_LAT", 17.3850)), float(os.getenv("WINDOW_LON", 78.4867))


def now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TZ).tz_localize(None)


def _km(a: tuple[float, float], b: tuple[float, float]) -> float:
    (la1, lo1), (la2, lo2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    d = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(d))


def get_places(n: int = 5, origin: tuple[float, float] | None = None, path: Path = PLACES) -> list[dict]:
    """The n outdoor spots nearest to `origin`, with distance in km."""
    origin = origin or home()
    places = json.loads(path.read_text())
    for p in places:
        p["km"] = round(_km(origin, (p["lat"], p["lon"])), 1)
    return sorted(places, key=lambda p: p["km"])[:n]


def get_forecast(df: pd.DataFrame | None = None, now: pd.Timestamp | None = None,
                 location: str = "hyderabad") -> dict:
    """Scored daylight hours for the plan day: the rest of today, or tomorrow if
    fewer than 2 daylight hours remain today."""
    if df is None:
        df = pd.read_csv(DATA / "hourly.csv", parse_dates=["time"])
        df = df[df.location == location]
    now = (now if now is not None else now_ist()).floor("h")
    df = df.sort_values("time").copy()
    df["score"] = score_frame(df)
    future = df[(df.time >= now) & (df.is_day == 1) & df.score.notna()]

    rows, label = future[future.time.dt.date == now.date()], "today"
    if len(rows) < 2:
        rows, label = future[future.time.dt.date == (now + pd.Timedelta(days=1)).date()], "tomorrow"
    if rows.empty:
        raise ValueError("no scored daylight hours available; refresh data with ingest.openmeteo")

    hours = [{
        "hour": int(r.time.hour), "time": r.time.strftime("%H:%M"), "score": float(r.score),
        "feels_c": round(float(r.apparent_temperature), 1), "aqi": pm25_to_aqi(float(r.pm2_5)),
        "rain_pct": float(r.precipitation_probability), "uv": round(float(r.uv_index), 1),
    } for r in rows.itertuples()]
    return {"day": label, "date": str(rows.time.iloc[0].date()), "hours": hours}


def best_window(hours: list[dict], length: int = 2) -> dict:
    """Consecutive daylight hours with the highest mean score (earliest wins ties).
    Falls back to a single hour if no `length`-hour run exists."""
    for n in (length, 1):
        best = None
        for i in range(len(hours) - n + 1):
            run = hours[i:i + n]
            if any(run[k]["hour"] != run[0]["hour"] + k for k in range(n)):
                continue
            mean = sum(h["score"] for h in run) / n
            if best is None or mean > best[0]:
                best = (mean, run)
        if best:
            mean, run = best
            return {
                "start": f"{run[0]['hour']:02d}:00", "end": f"{(run[-1]['hour'] + 1) % 24:02d}:00",
                "score": round(mean, 1), "avg_feels_c": round(sum(h["feels_c"] for h in run) / n, 1),
                "max_aqi": max(h["aqi"] for h in run), "max_rain_pct": max(h["rain_pct"] for h in run),
                "max_uv": max(h["uv"] for h in run),
            }
    raise ValueError("no hours to build a window from")


def pick_place(places: list[dict], window: dict) -> dict:
    """Nearest place; prefer shaded ones when it is hot or the UV is high."""
    if window["avg_feels_c"] >= 30 or window["max_uv"] >= 6:
        shaded = [p for p in places if p["shade"]]
        if shaded:
            return shaded[0]
    return places[0]
