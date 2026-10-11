"""Today's plan for the API and the Telegram listener: serve data/last_plan.json
while it is current, otherwise re-plan from the cached CSV (no ingest, no TabPFN)."""
from __future__ import annotations

import json
import logging

import pandas as pd

from agent import planner
from agent.tools import now_ist
from model.features import DATA

log = logging.getLogger("today")
LAST = DATA / "last_plan.json"


def hourly() -> pd.DataFrame:
    df = pd.read_csv(DATA / "hourly.csv", parse_dates=["time"])
    return df[df.location == "hyderabad"].reset_index(drop=True)


def load_plan(df: pd.DataFrame | None = None) -> dict:
    """Current plan. Raises ValueError if no scored daylight hours are available."""
    today = str(now_ist().date())
    if LAST.exists():
        try:
            cached = json.loads(LAST.read_text())
            if cached.get("date", "") >= today:
                return cached
            log.info("last_plan.json is stale (%s < %s), re-planning", cached.get("date"), today)
        except json.JSONDecodeError:
            log.warning("last_plan.json unreadable, re-planning")
    result = planner.plan(hourly() if df is None else df)
    result["run_at"] = now_ist().isoformat(timespec="seconds")
    LAST.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result
