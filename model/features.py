"""
Feature building for pm2_5 forecasting at horizon h hours.

Row = one issue time t. Everything in X is known at t, except weather at the
target hour t+h, which comes from the weather forecast (in history we only have
Open-Meteo's analysis values, so offline scores are slightly optimistic).
y = pm2_5 at t+h.

`precipitation_probability` is 0 in all history rows (placeholder), so it is
never used as a feature.

A `Spec` switches on the standard improvements evaluated in model/forecast.py:
- extended: daily-cycle features (value at the same hour 1 and 2 days before the
  target), short-term trend, 24h volatility, pm10/ozone now, current weather and
  the weather change to t+h, cyclical hour encoding
- delta: learn the change y(t+h) - y(t) instead of the level (the model then only
  has to explain departures from persistence)
- keep_missing_weather: TabPFN handles NaN natively, so rows from the first ~19
  days (no weather in Open-Meteo's history) are kept instead of dropped
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

LAGS = (1, 2, 3, 6, 12, 24)
ROLLS = (3, 6, 24)
WEATHER = ["apparent_temperature", "relative_humidity_2m", "precipitation",
           "cloud_cover", "wind_speed_10m", "uv_index", "is_day"]
HORIZONS = (1, 3, 6)


@dataclass(frozen=True)
class Spec:
    name: str = "v0_level"
    extended: bool = False
    delta: bool = False
    keep_missing_weather: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


V0 = Spec()
VARIANTS = (
    V0,
    Spec("v1_delta", delta=True),
    Spec("v2_delta_ext", extended=True, delta=True),
    Spec("v3_delta_ext_allhist", extended=True, delta=True, keep_missing_weather=True),
)


def load_history(location: str = "hyderabad", path: Path = DATA / "hourly.csv") -> pd.DataFrame:
    """Hourly history (no forecast rows) on a gap-free hourly index."""
    df = pd.read_csv(path, parse_dates=["time"])
    df = df[(df.location == location) & ~df.is_forecast].set_index("time").sort_index()
    return df.asfreq("h")


def build(df: pd.DataFrame, h: int, with_target: bool = True, spec: Spec = V0) -> pd.DataFrame:
    """Feature matrix indexed by issue time t, with `y` and `target_time` columns.
    Rows with a missing pm2_5 feature or target are dropped; rows with missing
    weather are dropped too unless `spec.keep_missing_weather`. With
    `with_target=False` (inference) `y` is omitted."""
    pm = df["pm2_5"]
    X = pd.DataFrame(index=df.index)
    X["pm2_5_now"] = pm
    for lag in LAGS:
        X[f"pm2_5_lag{lag}"] = pm.shift(lag)
    for w in ROLLS:
        X[f"pm2_5_mean{w}"] = pm.rolling(w).mean()
    weather_cols = []
    for c in WEATHER:  # weather at the target hour
        X[f"{c}_t+h"] = df[c].shift(-h)
        weather_cols.append(f"{c}_t+h")
    tt = X.index + pd.Timedelta(hours=h)
    X["hour"] = tt.hour
    X["dow"] = tt.dayofweek

    if spec.extended:
        X["pm2_5_same_hour_1d"] = pm.shift(24 - h)   # value at t+h-24 (known at t)
        X["pm2_5_same_hour_2d"] = pm.shift(48 - h)
        X["pm2_5_trend1"] = pm - pm.shift(1)
        X["pm2_5_trend3"] = pm - pm.shift(3)
        X["pm2_5_std24"] = pm.rolling(24).std()
        X["pm10_now"] = df["pm10"]
        X["ozone_now"] = df["ozone"]
        X["hour_sin"] = np.sin(2 * np.pi * tt.hour / 24)
        X["hour_cos"] = np.cos(2 * np.pi * tt.hour / 24)
        for c in ("wind_speed_10m", "relative_humidity_2m", "apparent_temperature"):
            X[f"{c}_now"] = df[c]
            X[f"{c}_change"] = df[c].shift(-h) - df[c]
            weather_cols += [f"{c}_now", f"{c}_change"]

    X["target_time"] = tt
    if with_target:
        X["y"] = pm.shift(-h)
    required = [c for c in X.columns if not (spec.keep_missing_weather and c in weather_cols)]
    return X.dropna(subset=required)


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


def latest_issue_row(df_all: pd.DataFrame, h: int, spec: Spec = V0) -> pd.DataFrame:
    """Feature row for the most recent *observed* hour, for predicting t+h.
    `df_all` must include forecast rows so weather at t+h is present."""
    df = df_all.set_index("time").sort_index().asfreq("h")
    t = df.index[~df["is_forecast"].fillna(True).astype(bool)].max()
    return build(df, h, with_target=False, spec=spec).loc[[t]]
