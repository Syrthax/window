# Window

Predicts the best outdoor hour today in Hyderabad from heat, air quality and rain,
then sends ONE short notification with a time window and a nearby spot.
The screen is the shortest part of the experience.

Open-source AI is the core: TabPFN (local) forecasts PM2.5, Gemma via Ollama (local)
writes the nudge. Runs at $0 on a laptop and degrades gracefully offline.

Built for DEV Hacktoberfest Week 1 ("Touch Grass").

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m ingest.openmeteo      # fetch weather + air quality (cached in data/raw)
python -m ingest.validate       # sanity-check data/hourly.csv
python -m ingest.openmeteo --offline   # rebuild the CSV from the latest cache
```

## Data note

Air quality is CAMS model output, not ground-station readings.

## Status

Project started Oct 6, 2026.

## Post-deadline commits

_None._
