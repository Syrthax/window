"""
Feature building for pm2_5 forecasting at horizon h hours.

Row = one issue time t. Everything in X is known at t, except weather at the
target hour t+h, which comes from the weather forecast (in history we only have
Open-Meteo's analysis values, so offline scores are slightly optimistic).
y = pm2_5 at t+h.

`precipitation_probability` is 0 in all history rows (placeholder), so it is
never used as a feature.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

LAGS = (1, 2, 3, 6, 12, 24)
ROLLS = (3, 6, 24)
WEATHER = ["apparent_temperature", "relative_humidity_2m", "precipitation",
           "cloud_cover", "wind_speed_10m", "uv_index", "is_day"]
HORIZONS = (1, 3, 6)


def load_history(location: str = "hyderabad", path: Path = DATA / "hourly.csv") -> pd.DataFrame:
    """Hourly history (no forecast rows) on a gap-free hourly index."""
    df = pd.read_csv(path, parse_dates=["time"])
    df = df[(df.location == location) & ~df.is_forecast].set_index("time").sort_index()
    return df.asfreq("h")


def build(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Feature matrix indexed by issue time t, with `y` and `target_time` columns.
    Rows with any missing feature or target are dropped."""
    pm = df["pm2_5"]
    X = pd.DataFrame(index=df.index)
    X["pm2_5_now"] = pm
    for lag in LAGS:
        X[f"pm2_5_lag{lag}"] = pm.shift(lag)
    for w in ROLLS:
        X[f"pm2_5_mean{w}"] = pm.rolling(w).mean()
    for c in WEATHER:  # weather at the target hour
        X[f"{c}_t+h"] = df[c].shift(-h)
    tt = X.index + pd.Timedelta(hours=h)
    X["hour"] = tt.hour
    X["dow"] = tt.dayofweek
    X["y"] = pm.shift(-h)
    X["target_time"] = tt
    return X.dropna()


def split(X: pd.DataFrame, test_days: int = 14, train_days: int | None = 30):
    """Time split on *target* time: test = last `test_days`; train = rows whose
    target is before the cutoff (optionally only the last `train_days` before it).
    No shuffling, no overlap."""
    cutoff = X["target_time"].max() - pd.Timedelta(days=test_days)
    test = X[X["target_time"] > cutoff]
    train = X[X["target_time"] <= cutoff]
    if train_days:
        train = train[train["target_time"] > cutoff - pd.Timedelta(days=train_days)]
    return train, test


def feature_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in ("y", "target_time")]
