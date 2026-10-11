"""
TabPFNRegressor: pm2_5 at t+1, t+3, t+6 hours. Evaluates against baselines on
the last 14 days and writes data/eval.json.

    python -m model.forecast

Baselines (all computable at issue time t, no leakage):
- persistence:   y(t+h) = y(t)
- yesterday:     y(t+h) = y(t+h-24)
- climatology:   mean pm2_5 by hour-of-day over the training window
- CAMS forecast: NOT available. Open-Meteo returns the same model's analysis for
  past hours, and `pm2_5_previous_day1` is empty, so a historical "what CAMS
  predicted at t for t+h" cannot be reconstructed. Reported as null.
"""
from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from model.features import DATA, HORIZONS, build, feature_cols, load_history, split

log = logging.getLogger("forecast")


def _prepare_env() -> None:
    load_dotenv()  # TABPFN_TOKEN
    try:  # python.org builds ship without CA certs; TabPFN's license check uses urllib
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass


def metrics(y: pd.Series, p: pd.Series) -> dict[str, float]:
    e = np.asarray(p, dtype=float) - np.asarray(y, dtype=float)
    return {"mae": round(float(np.abs(e).mean()), 3), "rmse": round(float(np.sqrt((e ** 2).mean())), 3)}


def evaluate(h: int, df: pd.DataFrame, train_days: int | None = None) -> dict:
    from tabpfn import TabPFNRegressor

    X = build(df, h)
    train, test = split(X, train_days=train_days)
    cols = feature_cols(X)
    log.info("h=%d train=%d rows test=%d rows", h, len(train), len(test))

    t0 = time.perf_counter()
    model = TabPFNRegressor(device="cpu")
    model.fit(train[cols], train["y"])
    pred = model.predict(test[cols])
    secs = time.perf_counter() - t0

    pm = df["pm2_5"]
    yday = pm.shift(24 - h).reindex(test.index)
    clim = train.groupby("hour")["y"].mean()
    ok = yday.notna()

    return {
        "horizon_h": h,
        "n_train": len(train),
        "n_test": len(test),
        "fit_predict_s": round(secs, 1),
        "tabpfn": metrics(test["y"], pred),
        "persistence": metrics(test["y"], test["pm2_5_now"]),
        "yesterday": metrics(test["y"][ok], yday[ok]),
        "climatology": metrics(test["y"], test["hour"].map(clim).fillna(train["y"].mean())),
        "cams": None,
    }


def markdown(rows: list[dict]) -> str:
    out = ["| Horizon | Metric | TabPFN | Persistence | Yesterday | Climatology | CAMS |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        for m in ("mae", "rmse"):
            cams = "n/a" if r["cams"] is None else r["cams"][m]
            out.append(f"| +{r['horizon_h']}h | {m.upper()} | {r['tabpfn'][m]} | "
                       f"{r['persistence'][m]} | {r['yesterday'][m]} | {r['climatology'][m]} | {cams} |")
    return "\n".join(out)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train-days", type=int, default=0,
                   help="0 = all history (default); N = last N days")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _prepare_env()
    df = load_history()
    rows = [evaluate(h, df, a.train_days or None) for h in HORIZONS]
    (DATA / "eval.json").write_text(json.dumps(
        {"train_days": a.train_days or None, "test_days": 14, "results": rows}, indent=2))
    print(markdown(rows))


if __name__ == "__main__":
    main()
