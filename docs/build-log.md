# Build log

Decisions, numbers (MAE, latency) and fixes, newest last.

## Task 0: scaffold
- Scaffolded repo per spec (requirements, .gitignore, README, ingest/openmeteo.py, ingest/validate.py).

## Task 1: ingest + validate (2026-10-11)
- Env: system Python is 3.14.3 (only interpreter installed); venv built on it. TabPFN compatibility with 3.14 is untested, check in Task 2.
- Fetch of both Open-Meteo sources took 4.34s. 2256 rows (2026-07-11 00:00 -> 2026-10-12 23:00 IST), 30 flagged forecast, 0 missing hours.
- `validate` result: OK (no range violations, missing hours 0).
- Weather columns (temp, apparent temp, humidity, precip, cloud, wind, UV) are null for 19.6% of history: 436 rows, 2026-07-11 -> 2026-07-29 03:00. The weather endpoint with past_days=92 only has data from ~Jul 29; air quality covers all 92 days. Usable rows for weather-feature models: ~1790 (~75 days).
- `precipitation_probability` is 0 (not null) in all 2226 history rows, so it does not show up in the null report. It is a placeholder, not data. Excluded from training; inference-only.
- pm2_5 history: mean 19.0, median 15.0, max 72.6 ug/m3. us_aqi max 149 (never above 150 in this window).

## Task 2: TabPFN smoke test (CPU)
- TabPFN 9.1.0 installs and runs on Python 3.14.3 (macOS CPU), model `tabpfn_3_5`.
- Setup hurdles, in order:
  1. python.org Python has no CA certs for `urllib`; TabPFN swallows the SSL error and reports it as "gated on Hugging Face". Workaround: `export SSL_CERT_FILE=$(python -c "import certifi;print(certifi.where())")` (or run `Install Certificates.command` once). HF access itself was fine.
  2. Weights need a separate Prior Labs license acceptance (https://ux.priorlabs.ai, Licenses tab) plus `TABPFN_TOKEN` in `.env`. Without it: "no interactive terminal" error.
- Test: pm2_5 at t+1h, features = pm2_5 lags (1,2,3,6,12,24) + hour; test = last 14 days (336 rows).
  - Train on all history (1865 rows): fit 88.6s (includes first-run weight download), predict 11.1s, MAE 5.49 vs persistence 4.65.
  - Train on last 30 days (720 rows): fit 0.1s, predict 4.1s, MAE 5.92 vs persistence 4.65.
- Decision: the 60s rule is triggered by the first run, so train on the last 30 days. Caveat: the first run's fit time was inflated by the download, so full-history training may be under 60s on a warm cache. Re-time it in Task 3.
- Early signal: with only lag features, TabPFN does NOT beat persistence at 1h (5.5-5.9 vs 4.65 MAE). Task 3 adds rolling means and weather features and compares against CAMS, so this is not a verdict yet.

## Task 3: features + TabPFN forecast vs baselines
- Files: `model/features.py` (lags 1,2,3,6,12,24 + current value, rolling means 3/6/24h, weather at target hour, hour/dow of target; time split on *target* time, last 14 days = test), `model/forecast.py` (TabPFNRegressor per horizon 1/3/6h, writes `data/eval.json`, prints markdown).
- CORRECTION to Task 2: the smoke-test "persistence" MAE of 4.65 was wrong. The script used `lag1` (value at t-1) as persistence for a t+1 target, a 2-hour gap. Real 1h persistence MAE is 2.49. The Task 2 claim that TabPFN "does not beat persistence" was therefore unfounded.
- Training window: with weather-valid history (1454 train rows) fit+predict is ~9-10s per horizon. The 88.6s in Task 2 was the one-time weight download, so the "train on last 30 days" fallback is unnecessary. Default is now all history (`--train-days 0`); `--train-days 30` is still available.
- CAMS baseline is NOT obtainable. Open-Meteo's past values are the model's own analysis, i.e. the target itself (using them would be leakage, MAE 0), and `pm2_5_previous_day1` is returned empty (0 of 744 hours). Reported as null/"n/a". Replaced with two honest baselines: "yesterday" (value at t+h-24) and hour-of-day climatology. Idea for later: snapshot forecast rows daily so a true CAMS-forecast baseline exists in future runs.
- Test set: last 14 days of targets (336 rows per horizon). Weather features are analysis values rather than past forecasts, so scores are slightly optimistic.
- Results (MAE / RMSE in ug/m3, training on all history):

| Horizon | Metric | TabPFN | Persistence | Yesterday | Climatology |
|---|---|---|---|---|---|
| +1h | MAE | 2.369 | 2.485 | 7.684 | 21.941 |
| +1h | RMSE | 3.469 | 3.669 | 10.018 | 24.702 |
| +3h | MAE | 7.608 | 6.479 | 7.684 | 21.941 |
| +3h | RMSE | 9.81 | 8.425 | 10.018 | 24.702 |
| +6h | MAE | 9.455 | 10.785 | 7.684 | 21.941 |
| +6h | RMSE | 12.463 | 13.114 | 10.018 | 24.702 |

- Same "yesterday"/climatology numbers across horizons are expected: the 336 test targets are identical for every h and those baselines do not depend on issue time.
- Same run with a 30-day train window: MAE 2.694 / 7.563 / 10.478 at +1/+3/+6h, so more history helps at 1h and 6h.
- Read: TabPFN beats persistence at +1h (-5% MAE) and +6h (-12%), loses at +3h (+17%), and loses to "yesterday" at +6h (9.46 vs 7.68). Not a clear win.

## Task 4: outdoor score
- `model/score.py`: deterministic 0-100 score = 100 minus penalties (apparent temp, AQI, rain, UV) with hard caps (night 30, AQI>150 -> 15, heat index>40 -> 10, rain >=0.5 mm/h -> 20). All constants and the reasoning are in the module docstring. `components()` returns each penalty and cap so the planner can explain a number; `score_hour()` and `score_frame()` wrap it.
- PM2.5 -> AQI uses the EPA 2024 breakpoints (self-computed, since forecast pm2_5 is what we predict; Open-Meteo's `us_aqi` is not used for scoring).
- `tests/test_score.py`: 28 tests pass (night, rain prob and actual rain, AQI 150/151 boundary, heat index 40/41 boundary, cold, UV cap, monotonicity, missing inputs).
- Sanity check on 2026-10-12 forecast: best hour 06:00 (76), midday dip to 36 at 12:00 (apparent temp 34.7, UV 8.1), 16:00 recovers to 65.5; night hours sit at the 30 cap.
- Design choice: dusk hours (is_day=0 from ~18:00) are capped at 30, so "best window" will always be daytime. Revisit if evening walks should count.
- `precipitation_probability` in history is the 0 placeholder, so `score_frame` on history overstates dryness; only forecast rows carry real probabilities.
