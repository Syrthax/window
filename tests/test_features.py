import numpy as np
import pandas as pd
import pytest

from model.features import VARIANTS, build
from model.forecast import combine, stack_weights


def synth(n=24 * 6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-09-01", periods=n, freq="h")
    df = pd.DataFrame(index=idx)
    df["pm2_5"] = 20 + 10 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 2, n)
    for c in ("pm10", "ozone", "apparent_temperature", "relative_humidity_2m", "precipitation",
              "cloud_cover", "wind_speed_10m", "uv_index"):
        df[c] = rng.uniform(0, 50, n)
    df["is_day"] = ((idx.hour >= 6) & (idx.hour < 18)).astype(int)
    return df


@pytest.mark.parametrize("spec", VARIANTS, ids=lambda s: s.name)
@pytest.mark.parametrize("h", [1, 3, 6])
def test_no_future_pm25_leaks_into_features(spec, h):
    df = synth()
    t = df.index[72]
    X1 = build(df, h, spec=spec)
    df2 = df.copy()
    df2.loc[df2.index > t, "pm2_5"] += 1000  # change every future pm2_5 value
    X2 = build(df2, h, spec=spec)
    cols = [c for c in X1.columns if c not in ("y", "target_time")]
    pd.testing.assert_series_equal(X1.loc[t, cols], X2.loc[t, cols])
    assert X2.loc[t, "y"] != X1.loc[t, "y"]  # the target does see the future


def test_keep_missing_weather_keeps_more_rows():
    df = synth()
    df.iloc[:48, df.columns.get_loc("wind_speed_10m")] = np.nan
    strict, loose = VARIANTS[2], VARIANTS[3]
    assert len(build(df, 3, spec=loose)) > len(build(df, 3, spec=strict))


def test_stack_never_worse_than_best_member():
    y = np.arange(50, dtype=float)
    members = {"model": y + 0.1, "persistence": y + 5, "yesterday": y - 7}
    w = stack_weights(y, members)
    assert abs(sum(w.values()) - 1) < 1e-9 and all(v >= 0 for v in w.values())
    best_single = min(np.abs(m - y).mean() for m in members.values())
    assert np.abs(combine(w, members) - y).mean() <= best_single


def test_stack_weights_can_mix():
    y = np.zeros(10)
    w = stack_weights(y, {"model": np.full(10, 2.0), "persistence": np.full(10, -2.0), "yesterday": np.full(10, 9.0)})
    assert w["model"] == pytest.approx(0.5) and w["persistence"] == pytest.approx(0.5)
