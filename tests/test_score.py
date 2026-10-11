import math

import pytest

from model.score import components, pm25_to_aqi, score_frame, score_hour

# a perfect hour: mild, clean, dry, low UV, daylight
GOOD = dict(apparent_temp=24.0, pm2_5=5.0, precip_prob=0.0, uv=3.0, is_day=1)


def test_perfect_hour_scores_100():
    assert score_hour(**GOOD) == 100.0


def test_night_is_capped_even_if_otherwise_perfect():
    assert score_hour(**{**GOOD, "is_day": 0}) == 30.0


def test_night_ignores_uv_penalty():
    assert components(**{**GOOD, "is_day": 0, "uv": 11})["penalties"]["uv"] == 0.0


def test_rain_probability_lowers_score():
    assert score_hour(**{**GOOD, "precip_prob": 80}) == 60.0


def test_actual_rain_is_capped():
    s = score_hour(**GOOD, precipitation=0.5)
    assert s == 20.0 and "rain" in components(**GOOD, precipitation=0.5)["caps"]


def test_light_drizzle_below_cap_only_penalised():
    c = components(**GOOD, precipitation=0.2)
    assert "rain" not in c["caps"] and c["score"] == 96.0


def test_aqi_over_150_is_capped():
    c = components(**{**GOOD, "pm2_5": 60.0})  # AQI ~153
    assert c["aqi"] > 150 and c["score"] <= 15.0 and "aqi" in c["caps"]


def test_aqi_exactly_150_not_capped():
    c = components(**{**GOOD, "pm2_5": 55.4})
    assert c["aqi"] == 150 and "aqi" not in c["caps"]


def test_heat_index_over_40_is_capped():
    c = components(**{**GOOD, "apparent_temp": 41.0})
    assert c["score"] <= 10.0 and "heat" in c["caps"]


def test_heat_index_exactly_40_not_capped():
    c = components(**{**GOOD, "apparent_temp": 40.0})
    assert "heat" not in c["caps"] and c["score"] == 44.0  # 100 - 4 * (40 - 26)


def test_cold_is_penalised():
    assert score_hour(**{**GOOD, "apparent_temp": 10.0}) == 76.0


def test_high_uv_penalty_maxes_out_at_30():
    assert components(**{**GOOD, "uv": 14})["penalties"]["uv"] == 30.0


def test_score_never_negative_or_above_100():
    worst = score_hour(50.0, 400.0, 100.0, 14.0, 0, 30.0)
    assert 0.0 <= worst <= 10.0
    assert score_hour(**GOOD) <= 100.0


def test_hotter_is_never_better():
    scores = [score_hour(**{**GOOD, "apparent_temp": t}) for t in range(26, 45)]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("pm,aqi", [(0.0, 0), (9.0, 50), (9.1, 51), (35.4, 100),
                                    (35.5, 101), (55.4, 150), (55.5, 151), (500.0, 500)])
def test_pm25_to_aqi_breakpoints(pm, aqi):
    assert pm25_to_aqi(pm) == aqi


def test_pm25_to_aqi_monotonic():
    vals = [pm25_to_aqi(x / 2) for x in range(0, 700)]
    assert vals == sorted(vals)


@pytest.mark.parametrize("bad", [None, math.nan, -1.0])
def test_bad_pm25_raises(bad):
    with pytest.raises(ValueError):
        pm25_to_aqi(bad)


def test_missing_input_raises():
    with pytest.raises(ValueError):
        score_hour(**{**GOOD, "uv": math.nan})


def test_score_frame_nan_for_missing_rows():
    import pandas as pd
    df = pd.DataFrame([
        dict(apparent_temperature=24, pm2_5=5, precipitation_probability=0, uv_index=3, is_day=1, precipitation=0),
        dict(apparent_temperature=None, pm2_5=5, precipitation_probability=0, uv_index=3, is_day=1, precipitation=0),
    ])
    s = score_frame(df)
    assert s.iloc[0] == 100.0 and math.isnan(s.iloc[1])
