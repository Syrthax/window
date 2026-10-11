# Claude Code prompt: build "Window" (paste this whole file into Claude Code, or save it as CLAUDE.md in an empty folder and say "Execute CLAUDE.md")

You are working in an empty folder that will become the git repo `window`. Build the project below end to end, task by task. Commit after each task with a clear message. Append decisions, numbers (MAE, latency) and fixes to `docs/build-log.md` as you go. Stop and ask me only at the checkpoints marked **ASK**.

## What this is
Personal tool + DEV Hacktoberfest Week 1 ("Touch Grass") submission.
Predicts the best outdoor hour today in Hyderabad from heat, air quality and rain,
then sends ONE short notification with a time window and a nearby spot.
The screen should be the shortest part of the experience.

## Hard constraints
- Deadline: Oct 11, 2026 11:59 PM PDT (= Oct 12 12:29 PM IST). Internal hard stop: Oct 11 10 PM IST.
- New project only: all code written in this repo during Oct 5–11. Do not copy code from other repos.
- Open-source AI must be the core: TabPFN (local) + Gemma via Ollama (local). No closed LLM APIs in the runtime path.
- Runs at $0 on a laptop; must degrade gracefully offline (cached data, templated nudge if Ollama is down).
- Python 3.11+ backend. React (Vite) only for the demo page on Day 4.
- Timestamps are naive IST everywhere (Open-Meteo `timezone=Asia/Kolkata`). Never mix UTC.

## Architecture
```
ingest/openmeteo.py   Open-Meteo weather + air quality, past_days=92, forecast_days=2
                      retry/backoff, raw JSON cache in data/raw, idempotent upsert -> data/hourly.csv
ingest/validate.py    gaps / nulls / range checks, non-zero exit on failure
model/features.py     lag + calendar features, train/test split by time (no shuffling)
model/forecast.py     TabPFNRegressor: pm2_5 at t+1..t+6h; eval vs baselines; writes data/eval.json
model/score.py        transparent rule: outdoor score 0-100 per hour from apparent temp, pm2_5->AQI, precip prob, UV, is_day
agent/planner.py      Gemma (Ollama HTTP API) with tools get_forecast(), get_places(); returns strict JSON
agent/fallback.py     templated nudge when Ollama is unreachable or returns invalid JSON
data/places.hyd.json  10-15 outdoor spots: name, lat, lon, type (park/lake/trail), shade (bool)
notify/telegram.py    send the nudge (token/chat id from .env)
api/main.py           FastAPI: GET /today (scores + chosen window + reason), GET /eval
web/                  Vite + React: 24h score curve, chosen window highlighted, "why" text
```

## Status
- Not started. Task 0 creates the scaffold below verbatim.
- User is handling: partner credits, Hugging Face license for TabPFN weights if gated, Telegram bot token

## Task queue (do in order, commit after each with a clear message)
1. Run `python -m ingest.openmeteo` and `python -m ingest.validate`. Fix any issues. Report row counts, null %, missing hours.
   - Expect `precipitation_probability` null in history: exclude from training features, use at inference only.
2. TabPFN smoke test on CPU; record fit/predict time. If >60s, train on last 30 days only.
3. model/features.py + model/forecast.py
   - Target: pm2_5 at horizons 1,3,6h. Features: pm2_5 lags (1,2,3,6,12,24h), rolling means, forecast weather at target hour, hour-of-day, day-of-week.
   - Time-based split: last 14 days = test.
   - Baselines: persistence (y_t+h = y_t) and Open-Meteo's own CAMS forecast value at t+h.
   - Output MAE/RMSE table per horizon to data/eval.json and stdout (markdown table — goes in the blog post).
   - If TabPFN does not beat CAMS, keep the result honestly and flag it; plan B is a personal-comfort classifier (see below).
4. model/score.py — documented, deterministic scoring; unit tests in tests/test_score.py for edge cases (night, rain, AQI > 150, heat index > 40).
5. agent/planner.py — Ollama at http://localhost:11434, model from env `WINDOW_MODEL` (default `gemma3:4b`).
   - Output schema: {"start":"HH:MM","end":"HH:MM","place":str,"activity":str,"message":str (<=140 chars),"reason":str}
   - Validate JSON; 1 retry; then agent/fallback.py. Log latency per stage.
6. notify/telegram.py + a `python -m window.run` entrypoint (ingest -> predict -> score -> plan -> notify). Cron/Task Scheduler instructions in README.
7. api/main.py + web/ demo page. Deploy-ready for Render (render.yaml) — optional.
8. Optional: Sentry agent tracing around planner + model stages.

## Plan B (if forecasting is weak)
`feedback/log.py`: after each outing, record conditions + user rating 1-5. TabPFNClassifier learns personal comfort from ~15 rows.

## Engineering rules
- No silent failures: every external call (Open-Meteo, Ollama, Telegram) has timeout, retry/backoff, and a logged fallback.
- Keep raw caches so the full pipeline runs with `--offline` for demos.
- Small, readable modules; type hints; no notebooks in the main path.
- Do not commit .env, data/raw, or model weights.
- Keep README "Post-deadline commits" section accurate.

## Write-up support (judging weights writing highest)
Keep notes in `docs/build-log.md` as you go: decisions, numbers (MAE, latency), what broke and how it was fixed.
These become the "How I Built It" and "Why Open Innovation Matters" sections.

## Task 0: scaffold (create these files exactly, then `git init` and make the first commit)

`requirements.txt`
```
requests>=2.31
pandas>=2.1
tabpfn
scikit-learn
fastapi
uvicorn
python-dotenv
pytest
```

`.gitignore`
```
.venv/
__pycache__/
data/raw/
*.pyc
.env
node_modules/
```

`README.md`: project summary, quickstart, data note ("Air quality is CAMS model output, not ground-station readings"), "Project started Oct 6, 2026", and a "Post-deadline commits" section containing "_None._".

`ingest/__init__.py`: empty.

`ingest/openmeteo.py`
```python
"""
Pull hourly weather + air quality for one location from Open-Meteo.

- past_days=92 gives ~3 months of history from the same models used for the
  forecast, so history and forecast columns are consistent.
- Raw JSON is cached in data/raw/ so the demo can run offline (--offline).
- Output is upserted into data/hourly.csv keyed on `time` (re-runs are safe;
  newer fetches overwrite older forecast rows with analysis values).

Usage:
    python -m ingest.openmeteo
    python -m ingest.openmeteo --lat 17.42 --lon 78.43 --name kbr_park
    python -m ingest.openmeteo --offline      # rebuild CSV from latest cache
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

log = logging.getLogger("ingest")

TZ = "Asia/Kolkata"
DATA = Path(__file__).resolve().parent.parent / "data"
RAW = DATA / "raw"

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_VARS = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "precipitation", "precipitation_probability", "cloud_cover",
    "wind_speed_10m", "uv_index", "is_day",
]
AQ_VARS = ["pm2_5", "pm10", "ozone", "us_aqi"]


def fetch(url: str, params: dict, retries: int = 4, timeout: int = 20) -> dict:
    """GET with exponential backoff on network errors / 429 / 5xx.
    4xx (bad params) fails immediately with Open-Meteo's `reason`."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 400:
                raise ValueError(f"Bad request: {r.json().get('reason', r.text)}")
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except (requests.RequestException,) as e:
            wait = 2 ** attempt
            log.warning("fetch %s failed (%s), retry %d/%d in %ds",
                        url, e, attempt + 1, retries, wait)
            time.sleep(wait)
    raise RuntimeError(f"Giving up on {url} after {retries} attempts")


def cache_path(source: str, name: str) -> Path:
    stamp = datetime.now(ZoneInfo(TZ)).strftime("%Y%m%dT%H%M")
    return RAW / f"{name}_{source}_{stamp}.json"


def latest_cache(source: str, name: str) -> dict:
    files = sorted(RAW.glob(f"{name}_{source}_*.json"))
    if not files:
        raise FileNotFoundError(f"No cached {source} payload for '{name}' in {RAW}")
    log.info("using cache %s", files[-1].name)
    return json.loads(files[-1].read_text())


def to_frame(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"])  # local (IST) naive timestamps
    return df


def pull(lat: float, lon: float, name: str, offline: bool) -> pd.DataFrame:
    common = {"latitude": lat, "longitude": lon, "timezone": TZ, "past_days": 92}
    if offline:
        w, aq = latest_cache("weather", name), latest_cache("aq", name)
    else:
        RAW.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        w = fetch(WEATHER_URL, {**common, "hourly": ",".join(WEATHER_VARS), "forecast_days": 2})
        aq = fetch(AQ_URL, {**common, "hourly": ",".join(AQ_VARS), "forecast_days": 2})
        log.info("fetched both sources in %.2fs", time.perf_counter() - t0)
        cache_path("weather", name).write_text(json.dumps(w))
        cache_path("aq", name).write_text(json.dumps(aq))

    df = to_frame(w).merge(to_frame(aq), on="time", how="outer").sort_values("time")
    now = pd.Timestamp.now(tz=TZ).tz_localize(None).floor("h")
    df["is_forecast"] = df["time"] > now
    df["location"] = name
    return df


def upsert(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path, parse_dates=["time"])
        df = pd.concat([old, df], ignore_index=True)
    df = (df.drop_duplicates(subset=["location", "time"], keep="last")
            .sort_values(["location", "time"]))
    df.to_csv(path, index=False)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lat", type=float, default=17.3850)
    p.add_argument("--lon", type=float, default=78.4867)
    p.add_argument("--name", default="hyderabad")
    p.add_argument("--offline", action="store_true")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = pull(a.lat, a.lon, a.name, a.offline)
    out = upsert(df, DATA / "hourly.csv")
    log.info("upserted %d rows for '%s' (%d total, %d forecast)",
             len(df), a.name, len(out), int(df["is_forecast"].sum()))


if __name__ == "__main__":
    main()
```

`ingest/validate.py`
```python
"""
Sanity-check data/hourly.csv before modelling. Exits non-zero on hard failures.

    python -m ingest.validate
"""
import sys
from pathlib import Path

import pandas as pd

PATH = Path(__file__).resolve().parent.parent / "data" / "hourly.csv"

# (min, max) plausible ranges for Hyderabad; outside = data bug, not weather
RANGES = {
    "temperature_2m": (5, 50),
    "relative_humidity_2m": (0, 100),
    "precipitation": (0, 200),
    "pm2_5": (0, 1000),
    "pm10": (0, 2000),
    "us_aqi": (0, 500),
    "uv_index": (0, 16),
}


def main() -> int:
    df = pd.read_csv(PATH, parse_dates=["time"])
    fail = False

    for loc, g in df.groupby("location"):
        print(f"\n== {loc}: {len(g)} rows, {g.time.min()} -> {g.time.max()}")

        full = pd.date_range(g.time.min(), g.time.max(), freq="h")
        missing = full.difference(g.time)
        print(f"missing hours: {len(missing)}" + (f" (first: {missing[:3].tolist()})" if len(missing) else ""))
        if len(missing) > 0.02 * len(full):
            fail = True

        hist = g[~g.is_forecast]
        nulls = (hist.isna().mean() * 100).round(1)
        print("null % (history):")
        print(nulls[nulls > 0].to_string() or "  none")

        for col, (lo, hi) in RANGES.items():
            if col in g:
                bad = g[(g[col] < lo) | (g[col] > hi)]
                if len(bad):
                    print(f"OUT OF RANGE {col}: {len(bad)} rows")
                    fail = True

    print("\nhistory summary:")
    cols = [c for c in ["apparent_temperature", "pm2_5", "us_aqi", "precipitation"] if c in df]
    print(df[~df.is_forecast][cols].describe().round(1).to_string())

    print("\nFAIL" if fail else "\nOK")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
```

## Checkpoints (ASK)
- After task 1: show me the `validate` output.
- After task 3: show me the MAE table per horizon vs both baselines, and recommend Plan A (forecasting) or Plan B (personal comfort) before continuing.
- Before task 7: confirm whether I want the Render deploy.

## Secrets I will provide in `.env`
`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, optional `SENTRY_DSN`, optional `WINDOW_MODEL` (default `gemma3:4b`).
If TabPFN weights are gated on Hugging Face and the download fails, stop and tell me to accept the license and run `huggingface-cli login`.

## Start now with Task 0.