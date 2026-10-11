"""
TabPFNRegressor: pm2_5 at t+1, t+3, t+6 hours, evaluated against baselines.
Writes data/eval.json (also read by model/predict.py and the demo page).

    python -m model.forecast

Protocol (no peeking at the test set):
1. Time-ordered windows on target time: [ train | validation 14d | test 14d ].
2. Each feature/target variant in model.features.VARIANTS is fit on train and
   scored on validation. The variant with the lowest mean validation MAE across
   horizons is selected.
3. Stacking: per horizon, convex weights over (TabPFN, persistence, yesterday)
   are fit on the validation predictions (grid, step 0.05).
4. Only then: refit on train+validation, predict the test window once, report.

Baselines (all computable at issue time t, no leakage):
- persistence:   y(t+h) = y(t)
- yesterday:     y(t+h) = y(t+h-24)
- climatology:   mean pm2_5 by hour-of-day over the training rows
- CAMS forecast: NOT available. Open-Meteo returns the same model's analysis for
  past hours, and `pm2_5_previous_day1` is empty, so a historical "what CAMS
  predicted at t for t+h" cannot be reconstructed. Reported as null.
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from model.features import DATA, HORIZONS, VARIANTS, Spec, build, feature_cols, load_history

log = logging.getLogger("forecast")

TEST_DAYS = VAL_DAYS = 14
MEMBERS = ("model", "persistence", "yesterday")


def _prepare_env() -> None:
    load_dotenv()  # TABPFN_TOKEN
    try:  # python.org builds ship without CA certs; TabPFN's license check uses urllib
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass


def metrics(y, p) -> dict[str, float]:
    e = np.asarray(p, dtype=float) - np.asarray(y, dtype=float)
    return {"mae": round(float(np.abs(e).mean()), 3), "rmse": round(float(np.sqrt((e ** 2).mean())), 3)}


def fit_predict(spec: Spec, train: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """Fit TabPFN on `train`, predict pm2_5 levels for `rows`."""
    from tabpfn import TabPFNRegressor

    cols = feature_cols(train)
    y = train["y"] - train["pm2_5_now"] if spec.delta else train["y"]
    pred = TabPFNRegressor(device="cpu").fit(train[cols], y).predict(rows[cols])
    return np.maximum(pred + rows["pm2_5_now"].to_numpy() if spec.delta else pred, 0.0)


def stack_weights(y, members: dict[str, np.ndarray], step: float = 0.05) -> dict[str, float]:
    """Convex weights (sum to 1, each >= 0) minimising MAE, by grid search."""
    names, y = list(members), np.asarray(y, dtype=float)
    grid = np.round(np.arange(0, 1 + 1e-9, step), 2)
    best = (np.inf, None)
    for w in itertools.product(grid, repeat=len(names) - 1):
        last = round(1 - sum(w), 2)
        if last < -1e-9:
            continue
        ws = (*w, max(last, 0.0))
        mae = np.abs(sum(wi * members[n] for wi, n in zip(ws, names)) - y).mean()
        if mae < best[0] - 1e-12:
            best = (mae, ws)
    return dict(zip(names, (float(x) for x in best[1])))


def combine(weights: dict[str, float], members: dict[str, np.ndarray]) -> np.ndarray:
    return sum(weights[n] * members[n] for n in weights)


def evaluate_horizon(h: int, hist: pd.DataFrame) -> dict:
    frames = {s.name: build(hist, h, spec=s) for s in VARIANTS}
    end = min(X["target_time"].max() for X in frames.values())
    test_cut = end - pd.Timedelta(days=TEST_DAYS)
    val_cut = test_cut - pd.Timedelta(days=VAL_DAYS)

    def window(X, lo, hi):
        return X.index[(X["target_time"] > lo) & (X["target_time"] <= hi)]

    # identical evaluation rows for every variant and baseline
    val_idx = sorted(set.intersection(*(set(window(X, val_cut, test_cut)) for X in frames.values())))
    test_idx = sorted(set.intersection(*(set(window(X, test_cut, end)) for X in frames.values())))
    ref = frames[VARIANTS[0].name]
    y_val, y_test = ref.loc[val_idx, "y"].to_numpy(), ref.loc[test_idx, "y"].to_numpy()

    yday = hist["pm2_5"].shift(24 - h)
    base = {}
    for name, idx in (("val", val_idx), ("test", test_idx)):
        base[name] = {"persistence": ref.loc[idx, "pm2_5_now"].to_numpy(),
                      "yesterday": yday.reindex(idx).to_numpy()}
    train_for_test = ref[ref["target_time"] <= test_cut]
    clim = train_for_test.groupby("hour")["y"].mean()
    clim_test = ref.loc[test_idx, "hour"].map(clim).fillna(train_for_test["y"].mean()).to_numpy()

    variants, preds = {}, {}
    for s in VARIANTS:
        X = frames[s.name]
        t0 = time.perf_counter()
        p_val = fit_predict(s, X[X["target_time"] <= val_cut], X.loc[val_idx])
        p_test = fit_predict(s, X[X["target_time"] <= test_cut], X.loc[test_idx])
        preds[s.name] = {"val": p_val, "test": p_test}
        variants[s.name] = {"val": metrics(y_val, p_val), "test": metrics(y_test, p_test),
                            "n_train": int((X["target_time"] <= test_cut).sum()),
                            "fit_predict_s": round(time.perf_counter() - t0, 1)}
        log.info("h=%d %-22s val MAE %.3f  (n_train %d, %.0fs)", h, s.name, variants[s.name]["val"]["mae"],
                 variants[s.name]["n_train"], variants[s.name]["fit_predict_s"])

    return {"horizon_h": h, "n_val": len(val_idx), "n_test": len(test_idx), "variants": variants,
            "_preds": preds, "_base": base, "_y": {"val": y_val, "test": y_test},
            "persistence": metrics(y_test, base["test"]["persistence"]),
            "yesterday": metrics(y_test, base["test"]["yesterday"]),
            "climatology": metrics(y_test, clim_test), "cams": None}


def finalize(rows: list[dict]) -> tuple[Spec, list[dict]]:
    """Pick the variant on validation, fit stacking weights on validation, score test."""
    mean_val = {s.name: np.mean([r["variants"][s.name]["val"]["mae"] for r in rows]) for s in VARIANTS}
    chosen = min(VARIANTS, key=lambda s: mean_val[s.name])
    log.info("selected %s (mean val MAE %.3f)", chosen.name, mean_val[chosen.name])
    for r in rows:
        p, b, y = r.pop("_preds")[chosen.name], r.pop("_base"), r.pop("_y")
        val_members = {"model": p["val"], **b["val"]}
        test_members = {"model": p["test"], **b["test"]}
        w = stack_weights(y["val"], val_members)
        r["weights"] = w
        r["tabpfn"] = r["variants"][chosen.name]["test"]
        r["final"] = metrics(y["test"], combine(w, test_members))
        r["val"] = {"final": metrics(y["val"], combine(w, val_members)),
                    "persistence": metrics(y["val"], b["val"]["persistence"]),
                    "yesterday": metrics(y["val"], b["val"]["yesterday"])}
        r["n_train"] = r["variants"][chosen.name]["n_train"]
    return chosen, rows


def markdown(rows: list[dict], v0: str = VARIANTS[0].name) -> str:
    out = ["| Horizon | Metric | **Final (stacked)** | TabPFN (selected) | TabPFN v0 (original) | Persistence | Yesterday | Climatology | CAMS |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for m in ("mae", "rmse"):
            out.append(f"| +{r['horizon_h']}h | {m.upper()} | **{r['final'][m]}** | {r['tabpfn'][m]} | "
                       f"{r['variants'][v0]['test'][m]} | {r['persistence'][m]} | {r['yesterday'][m]} | "
                       f"{r['climatology'][m]} | n/a |")
    return "\n".join(out)


def validation_table(rows: list[dict]) -> str:
    out = ["| Variant | " + " | ".join(f"+{r['horizon_h']}h val MAE" for r in rows) + " | mean |",
           "|---|" + "---|" * (len(rows) + 1)]
    for s in VARIANTS:
        v = [r["variants"][s.name]["val"]["mae"] for r in rows]
        out.append(f"| {s.name} | " + " | ".join(f"{x:.3f}" for x in v) + f" | {np.mean(v):.3f} |")
    return "\n".join(out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _prepare_env()
    hist = load_history()
    chosen, rows = finalize([evaluate_horizon(h, hist) for h in HORIZONS])
    (DATA / "eval.json").write_text(json.dumps({
        "test_days": TEST_DAYS, "val_days": VAL_DAYS, "selected": chosen.to_dict(),
        "selection": "lowest mean validation MAE across horizons; stacking weights fit on validation",
        "results": rows}, indent=2))
    print("\nValidation (used for selection):\n" + validation_table(rows))
    print(f"\nSelected: {chosen.name}\n\nTest (last {TEST_DAYS} days, scored once):\n" + markdown(rows))
    for r in rows:
        print(f"+{r['horizon_h']}h stacking weights: {r['weights']}")


if __name__ == "__main__":
    main()
