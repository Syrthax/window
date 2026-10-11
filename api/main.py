"""
FastAPI backend for the demo page.

    uvicorn api.main:app --reload        # http://127.0.0.1:8000

GET /today   chosen window, message, reason and the 24h score curve for the plan day
GET /eval    forecast evaluation table (data/eval.json)
GET /health

/today serves data/last_plan.json (written by `python -m window.run`) while it is
current; if it is missing or stale it re-plans from the cached CSV (no ingest, no
TabPFN, about 5s with Gemma). If web/dist exists it is served at / so one process
runs the whole demo.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from agent.tools import aqi_label
from model.features import DATA
from model.score import pm25_to_aqi, score_frame
from window import today as today_mod

load_dotenv()
log = logging.getLogger("api")
DIST = Path(__file__).resolve().parent.parent / "web" / "dist"

app = FastAPI(title="Window")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["GET"])


def _hourly() -> pd.DataFrame:
    return today_mod.hourly()


def _num(x, digits: int = 1):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), digits)


def curve(df: pd.DataFrame, date: str) -> list[dict]:
    """All 24 hours of `date`, scored. Uses the CSV's CAMS pm2_5 (the TabPFN blend only
    touches a few near-term hours and is not persisted)."""
    day = df[df.time.dt.strftime("%Y-%m-%d") == date].copy()
    day["score"] = score_frame(day)
    out = []
    for r in day.itertuples():
        aqi = None if pd.isna(r.pm2_5) else pm25_to_aqi(float(r.pm2_5))
        out.append({"hour": int(r.time.hour), "time": r.time.strftime("%H:%M"), "score": _num(r.score),
                    "feels_c": _num(r.apparent_temperature), "aqi": aqi,
                    "air": None if aqi is None else aqi_label(aqi), "rain_pct": _num(r.precipitation_probability, 0),
                    "uv": _num(r.uv_index), "is_day": bool(r.is_day) if pd.notna(r.is_day) else None})
    return out


def _load_plan(df: pd.DataFrame) -> dict:
    try:
        return today_mod.load_plan(df)
    except ValueError as e:  # no scored daylight hours
        raise HTTPException(503, f"no plan available: {e}")


@app.get("/today")
def today() -> dict:
    df = _hourly()
    p = _load_plan(df)
    return {k: p.get(k) for k in ("date", "day", "source", "model", "nowcast", "run_at", "plan", "window")} | {
        "curve": curve(df, p["date"])}


@app.get("/eval")
def evaluation() -> dict:
    path = DATA / "eval.json"
    if not path.exists():
        raise HTTPException(404, "data/eval.json missing; run `python -m model.forecast`")
    return json.loads(path.read_text())


@app.get("/health")
def health() -> dict:
    return {"ok": True}


if DIST.exists():  # must come after the routes
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")
