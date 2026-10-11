# Window

Predicts the best outdoor hour today in Hyderabad from heat, air quality and rain,
then sends ONE short notification with a time window and a nearby spot.
The screen is the shortest part of the experience.

Open-source AI is the core: TabPFN (local) nowcasts PM2.5, Gemma 3 via Ollama (local)
picks the spot and writes the nudge. Runs at $0 on a laptop and degrades gracefully
offline (cached data, templated nudge if Ollama is down).

Built for DEV Hacktoberfest Week 1 ("Touch Grass").

## Pipeline

```
ingest (Open-Meteo) -> predict (TabPFN, +1/+3/+6h PM2.5) -> score (rule, 0-100/hour)
                    -> plan (Python picks the window, Gemma picks the place + wording)
                    -> notify (Telegram)
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env            # then fill it in (see below)
ollama pull gemma3:4b           # needs a recent Ollama (0.40+)

python -m window.run --dry-run  # everything except the Telegram send
python -m window.run            # full run, sends the notification
python -m window.run --offline  # rebuild data from the cached raw JSON
python -m window.run --no-tabpfn
```

Individual stages: `python -m ingest.openmeteo`, `python -m ingest.validate`,
`python -m model.forecast` (evaluation, writes `data/eval.json`), `python -m agent.planner`.
Tests: `python -m pytest`.

### Configuration (`.env`, never committed)

| Variable | Purpose |
|---|---|
| `TELEGRAM_BOT_TOKEN` | from @BotFather |
| `TELEGRAM_CHAT_ID` | run `python -m notify.telegram --find-chat-id` after pressing Start on your bot |
| `TABPFN_TOKEN` | Prior Labs API key; accept the license at https://ux.priorlabs.ai first |
| `WINDOW_MODEL` | optional, default `gemma3:4b` |
| `WINDOW_LAT`, `WINDOW_LON` | optional, "nearby" is measured from here (default central Hyderabad) |
| `SENTRY_DSN` | optional |

### macOS gotcha: SSL certificates

The python.org Python ships without CA certificates for `urllib`. TabPFN's license check
uses `urllib` and reports the failure as "gated model". The code points `SSL_CERT_FILE` at
certifi automatically; to fix it system-wide run `/Applications/Python 3.x/Install Certificates.command`.

## Demo page

```bash
cd web && npm install && npm run build && cd ..
uvicorn api.main:app                  # http://127.0.0.1:8000  (serves the built page + API)
```

For live editing run `uvicorn api.main:app --reload` and `cd web && npm run dev` (page on :5173, proxies the API).
API: `GET /today` (window, message, reason, 24h score curve), `GET /eval` (forecast table), `GET /health`.
`/today` serves `data/last_plan.json` from the last `window.run`, or re-plans from the cached CSV if it is stale.

## Schedule it

The plan is for the day ahead (or tomorrow, if you run it after sunset), so run it early morning.
Ollama must be running (`brew services start ollama`).

**macOS / Linux (cron)**: `crontab -e`, then add (05:30 IST daily; adjust the path):

```
30 5 * * * cd /path/to/window && .venv/bin/python -m window.run >> data/run.log 2>&1
```

If the machine is asleep at 05:30, cron will not run it; use `launchd` or wake the Mac with
`pmset repeat wakeorpoweron MTWRFSU 05:25:00`.

**Windows (Task Scheduler)**:

```
schtasks /Create /SC DAILY /ST 05:30 /TN "Window" /TR "cmd /c cd /d C:\path\to\window && .venv\Scripts\python -m window.run >> data\run.log 2>&1"
```

Exit codes: `0` sent (or dry run), `2` plan made but Telegram failed (the message is still printed),
`1` no plan possible (no data at all).

## Data note

Air quality is CAMS model output, not ground-station readings.
Place coordinates are from OpenStreetMap (© OpenStreetMap contributors, ODbL), checked 2026-10-11.

## Status

Project started Oct 6, 2026. See `docs/build-log.md` for decisions, numbers and what broke.

## Post-deadline commits

_None._
