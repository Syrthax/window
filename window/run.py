"""
One run of the whole pipeline: ingest -> predict -> score -> plan -> notify.

    python -m window.run                # live
    python -m window.run --dry-run      # everything except the Telegram send
    python -m window.run --offline      # rebuild data from the latest raw cache (no network for ingest)
    python -m window.run --no-tabpfn    # skip the PM2.5 nowcast

Exit codes: 0 sent (or dry run), 2 plan made but notification failed, 1 no plan possible.
Each external step degrades instead of crashing: ingest falls back to the cached CSV,
TabPFN is skipped on error, Ollama falls back to a template (see agent/fallback.py),
and a failed Telegram send still prints the message.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from agent import planner
from ingest import openmeteo
from model import predict
from notify import telegram

log = logging.getLogger("run")


def ingest(offline: bool) -> pd.DataFrame:
    try:
        df = openmeteo.pull(17.3850, 78.4867, "hyderabad", offline)
        out = openmeteo.upsert(df, openmeteo.DATA / "hourly.csv")
        log.info("ingest ok: %d rows", len(out))
    except Exception as e:  # network down, no cache, bad payload: use what we already have
        path = openmeteo.DATA / "hourly.csv"
        if not path.exists():
            raise
        log.warning("ingest failed (%s: %s); using cached %s", type(e).__name__, e, path.name)
    df = pd.read_csv(openmeteo.DATA / "hourly.csv", parse_dates=["time"])
    return df[df.location == "hyderabad"].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-tabpfn", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()
    timings: dict[str, float] = {}

    def timed(name, fn, *args):
        t0 = time.perf_counter()
        try:
            return fn(*args)
        finally:
            timings[name] = round(time.perf_counter() - t0, 1)

    try:
        df = timed("ingest", ingest, a.offline)
    except Exception:
        log.exception("no data available, cannot plan")
        return 1

    nowcast_used = False
    if not a.no_tabpfn:
        try:
            preds = timed("predict", predict.nowcast, df)
            df = predict.blend(df, preds)
            nowcast_used = True
        except Exception as e:
            log.warning("TabPFN nowcast skipped (%s: %s); using CAMS forecast only", type(e).__name__, e)

    try:
        result = timed("plan", planner.plan, df)  # scoring happens inside get_forecast
    except Exception:
        log.exception("could not build a plan")
        return 1

    result.update(nowcast=nowcast_used, run_at=datetime.now().isoformat(timespec="seconds"),
                  stage_seconds=timings)
    (openmeteo.DATA / "last_plan.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    text = result["plan"]["message"]
    print(f"\n[{result['source']}] {text}\n  why: {result['plan']['reason']}\n")

    if a.dry_run:
        log.info("dry run: not sending. stages(s)=%s", timings)
        return 0
    try:
        telegram.send(text)
    except telegram.NotifyError as e:
        log.error("notification failed: %s", e)
        return 2
    log.info("done. stages(s)=%s", timings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
