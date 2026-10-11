"""
Inference: pm2_5 at t+1, t+3, t+6h from the latest observed hour, then those hours
are blended 50/50 with Open-Meteo's CAMS forecast.

The model is whatever `python -m model.forecast` selected on validation
(data/eval.json -> `selected` feature/target variant, and per-horizon stacking
weights over TabPFN, persistence and same-hour-yesterday). Without eval.json it
falls back to the original variant with TabPFN alone.

Why still 50/50 with CAMS: a true CAMS-forecast baseline cannot be reconstructed
from Open-Meteo (see docs/build-log.md), so neither source is shown to be better.
Hours other than t+1/3/6 keep the CAMS forecast.
"""
from __future__ import annotations

import json
import logging
import time

import pandas as pd

from model.features import DATA, HORIZONS, V0, Spec, build, latest_issue_row, load_history
from model.forecast import _prepare_env, fit_predict

log = logging.getLogger("predict")
DEFAULT_WEIGHTS = {"model": 1.0, "persistence": 0.0, "yesterday": 0.0}


def load_config(path=DATA / "eval.json") -> tuple[Spec, dict[int, dict[str, float]]]:
    try:
        cfg = json.loads(path.read_text())
        spec = Spec(**cfg["selected"])
        weights = {r["horizon_h"]: r["weights"] for r in cfg["results"]}
        return spec, weights
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as e:
        log.warning("no usable eval.json (%s); using %s with TabPFN only", type(e).__name__, V0.name)
        return V0, {}


def stacked(weights: dict[str, float], members: dict[str, float]) -> float:
    """Weighted mean over the members that are available (renormalised)."""
    avail = {k: w for k, w in weights.items() if w > 0 and pd.notna(members.get(k))}
    if not avail:
        return members["model"]
    total = sum(avail.values())
    return sum(w * members[k] for k, w in avail.items()) / total


def nowcast(df_all: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> dict[pd.Timestamp, float]:
    """{target_time: predicted pm2_5}, one TabPFN per horizon trained on all history."""
    _prepare_env()
    spec, weights = load_config()
    hist = load_history()
    out: dict[pd.Timestamp, float] = {}
    for h in horizons:
        t0 = time.perf_counter()
        row = latest_issue_row(df_all, h, spec)
        target = pd.Timestamp(row["target_time"].iloc[0])
        members = {
            "model": float(fit_predict(spec, build(hist, h, spec=spec), row)[0]),
            "persistence": float(row["pm2_5_now"].iloc[0]),
            "yesterday": float(hist["pm2_5"].get(target - pd.Timedelta(hours=24), float("nan"))),
        }
        pred = max(stacked(weights.get(h, DEFAULT_WEIGHTS), members), 0.0)
        out[target] = pred
        log.info("+%dh -> %s = %.1f ug/m3 (tabpfn %.1f, %s, %.1fs)", h, target, pred, members["model"],
                 spec.name, time.perf_counter() - t0)
    return out


def blend(df_all: pd.DataFrame, preds: dict[pd.Timestamp, float], weight: float = 0.5) -> pd.DataFrame:
    """Copy of df_all where pm2_5 at the predicted hours = weight*model + (1-weight)*CAMS."""
    df = df_all.copy()
    for t, p in preds.items():
        m = df["time"] == t
        if m.any() and pd.notna(df.loc[m, "pm2_5"]).all():
            df.loc[m, "pm2_5"] = weight * p + (1 - weight) * df.loc[m, "pm2_5"]
    return df
