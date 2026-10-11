"""
Inference: TabPFN predicts pm2_5 at t+1, t+3, t+6h from the latest observed hour,
and those hours are blended 50/50 with Open-Meteo's CAMS forecast.

Why a blend: offline, TabPFN beats persistence at +1h and +6h but not at +3h, and a
true CAMS-forecast baseline could not be reconstructed (see docs/build-log.md), so
it replaces nothing outright. Hours other than t+1/3/6 keep the CAMS forecast.
"""
from __future__ import annotations

import logging
import time

import pandas as pd

from model.features import HORIZONS, build, feature_cols, latest_issue_row, load_history
from model.forecast import _prepare_env

log = logging.getLogger("predict")


def nowcast(df_all: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> dict[pd.Timestamp, float]:
    """{target_time: predicted pm2_5}, training one TabPFN per horizon on all history."""
    from tabpfn import TabPFNRegressor

    _prepare_env()
    hist = load_history()
    out: dict[pd.Timestamp, float] = {}
    for h in horizons:
        t0 = time.perf_counter()
        X = build(hist, h)
        row = latest_issue_row(df_all, h)
        cols = feature_cols(X)
        model = TabPFNRegressor(device="cpu").fit(X[cols], X["y"])
        pred = float(model.predict(row[cols])[0])
        out[pd.Timestamp(row["target_time"].iloc[0])] = max(pred, 0.0)
        log.info("tabpfn +%dh -> %s = %.1f ug/m3 (%.1fs)", h, row["target_time"].iloc[0], pred,
                 time.perf_counter() - t0)
    return out


def blend(df_all: pd.DataFrame, preds: dict[pd.Timestamp, float], weight: float = 0.5) -> pd.DataFrame:
    """Copy of df_all where pm2_5 at the predicted hours = weight*TabPFN + (1-weight)*CAMS."""
    df = df_all.copy()
    for t, p in preds.items():
        m = df["time"] == t
        if m.any() and pd.notna(df.loc[m, "pm2_5"]).all():
            df.loc[m, "pm2_5"] = weight * p + (1 - weight) * df.loc[m, "pm2_5"]
    return df
