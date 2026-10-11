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
