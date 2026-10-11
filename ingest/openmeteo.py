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
