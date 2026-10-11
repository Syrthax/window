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
