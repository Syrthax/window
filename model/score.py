"""
Transparent outdoor-comfort score, 0-100 per hour. Deterministic, no ML.

    score = 100 - temperature - air_quality - rain - uv penalties
    then hard caps for conditions that make being outside a bad idea.

Inputs
- apparent_temp  deg C (Open-Meteo apparent_temperature, used as the heat index)
- pm2_5          ug/m3, converted to US AQI with the EPA 2024 breakpoints
- precip_prob    % (0-100), forecast only; history holds a 0 placeholder
- uv             UV index (penalised only in daylight)
- is_day         1 / 0
- precipitation  mm in the hour (optional, default 0)

Penalties (points)
- temp:  0 inside 18-26 C; 4/C above 26; 3/C below 18
- AQI:   0.3 per AQI point above 50 (so AQI 150 costs 30)
- rain:  0.5 per % precip probability, plus 20 per mm of actual rain (capped at 40)
- UV:    5 per index point above 5, max 30, daylight only

Hard caps (final score can never exceed)
- night (is_day == 0):          30
- AQI > 150 (unhealthy):        15
- heat index > 40 C:            10
- actual rain >= 0.5 mm/h:      20
"""
from __future__ import annotations

import math

import pandas as pd

IDEAL_LOW, IDEAL_HIGH = 18.0, 26.0
NIGHT_CAP, AQI_CAP, HEAT_CAP, RAIN_CAP = 30.0, 15.0, 10.0, 20.0
AQI_CAP_ABOVE, HEAT_CAP_ABOVE, RAIN_CAP_MM = 150, 40.0, 0.5

# EPA 2024 PM2.5 breakpoints: (pm_low, pm_high, aqi_low, aqi_high)
_PM25_BREAKPOINTS = (
    (0.0, 9.0, 0, 50),
    (9.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 125.4, 151, 200),
    (125.5, 225.4, 201, 300),
    (225.5, 500.0, 301, 500),
)


def pm25_to_aqi(pm2_5: float) -> int:
    """US AQI from a PM2.5 concentration (ug/m3), truncated to 0.1 per EPA."""
    if pm2_5 is None or math.isnan(pm2_5) or pm2_5 < 0:
        raise ValueError(f"invalid pm2_5: {pm2_5!r}")
    c = math.floor(pm2_5 * 10) / 10
    for lo, hi, a_lo, a_hi in _PM25_BREAKPOINTS:
        if c <= hi:
            return round((a_hi - a_lo) / (hi - lo) * (c - lo) + a_lo)
    return 500


def components(apparent_temp: float, pm2_5: float, precip_prob: float, uv: float,
               is_day: int, precipitation: float = 0.0) -> dict:
    """All penalties, caps and the final score, for explaining a number."""
    for name, v in (("apparent_temp", apparent_temp), ("pm2_5", pm2_5),
                    ("precip_prob", precip_prob), ("uv", uv), ("precipitation", precipitation)):
        if v is None or math.isnan(v):
            raise ValueError(f"{name} is missing")

    aqi = pm25_to_aqi(pm2_5)
    if apparent_temp > IDEAL_HIGH:
        temp_pen = 4.0 * (apparent_temp - IDEAL_HIGH)
    elif apparent_temp < IDEAL_LOW:
        temp_pen = 3.0 * (IDEAL_LOW - apparent_temp)
    else:
        temp_pen = 0.0
    aqi_pen = 0.3 * max(0, aqi - 50)
    rain_pen = 0.5 * min(max(precip_prob, 0.0), 100.0) + min(20.0 * max(precipitation, 0.0), 40.0)
    uv_pen = min(30.0, 5.0 * max(0.0, uv - 5.0)) if is_day else 0.0

    score = max(0.0, 100.0 - temp_pen - aqi_pen - rain_pen - uv_pen)
    caps = {}
    if not is_day:
        caps["night"] = NIGHT_CAP
    if aqi > AQI_CAP_ABOVE:
        caps["aqi"] = AQI_CAP
    if apparent_temp > HEAT_CAP_ABOVE:
        caps["heat"] = HEAT_CAP
    if precipitation >= RAIN_CAP_MM:
        caps["rain"] = RAIN_CAP
    if caps:
        score = min(score, *caps.values())

    return {"score": round(score, 1), "aqi": aqi,
            "penalties": {"temp": round(temp_pen, 1), "aqi": round(aqi_pen, 1),
                          "rain": round(rain_pen, 1), "uv": round(uv_pen, 1)},
            "caps": caps}


def score_hour(apparent_temp: float, pm2_5: float, precip_prob: float, uv: float,
               is_day: int, precipitation: float = 0.0) -> float:
    """Outdoor score 0-100 (higher = better). Raises ValueError on missing inputs."""
    return components(apparent_temp, pm2_5, precip_prob, uv, is_day, precipitation)["score"]


def score_frame(df: pd.DataFrame) -> pd.Series:
    """Score every row of a frame with Open-Meteo column names; NaN where inputs are missing."""
    def one(r: pd.Series) -> float:
        try:
            return score_hour(r["apparent_temperature"], r["pm2_5"], r["precipitation_probability"],
                              r["uv_index"], int(r["is_day"]), r.get("precipitation", 0.0))
        except (ValueError, TypeError):
            return float("nan")
    return df.apply(one, axis=1)
