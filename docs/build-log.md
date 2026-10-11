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
