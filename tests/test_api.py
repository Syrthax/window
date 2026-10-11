import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import main
from window import today as today_mod
from tests.test_planner import make_df

DATE = "2026-10-12"
PLAN = {"date": DATE, "day": "today", "source": "ollama", "model": "gemma3:4b", "nowcast": False,
        "run_at": "2026-10-12T05:30:00",
        "plan": {"start": "07:00", "end": "09:00", "place": "KBR National Park", "activity": "walk",
                 "message": "Best of a smoggy day: ...", "reason": "because"},
        "window": {"start": "07:00", "end": "09:00", "score": 90.0, "caution": []}}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "_hourly", lambda: make_df())
    monkeypatch.setattr(today_mod, "LAST", tmp_path / "last_plan.json")
    monkeypatch.setattr(main, "DATA", tmp_path)
    return TestClient(main.app)


def test_today_serves_current_cached_plan(client, monkeypatch):
    (today_mod.LAST).write_text(json.dumps(PLAN))
    monkeypatch.setattr(today_mod, "now_ist", lambda: pd.Timestamp("2026-10-12 06:00"))
    r = client.get("/today").json()
    assert r["plan"]["place"] == "KBR National Park" and r["window"]["start"] == "07:00"
    assert len(r["curve"]) == 24 and r["curve"][7]["score"] is not None
    assert r["curve"][2]["score"] == 30.0  # night cap


def test_stale_cache_triggers_replan(client, monkeypatch):
    (today_mod.LAST).write_text(json.dumps({**PLAN, "date": "2026-10-01"}))
    monkeypatch.setattr(today_mod, "now_ist", lambda: pd.Timestamp("2026-10-12 06:00"))
    monkeypatch.setattr(today_mod.planner, "plan", lambda df: {**PLAN, "source": "fallback"})
    assert client.get("/today").json()["source"] == "fallback"
    assert json.loads(today_mod.LAST.read_text())["source"] == "fallback"  # re-cached


def test_no_plan_possible_is_503(client, monkeypatch):
    def boom(df):
        raise ValueError("no scored daylight hours")
    monkeypatch.setattr(today_mod.planner, "plan", boom)
    assert client.get("/today").status_code == 503


def test_eval_404_then_ok(client):
    assert client.get("/eval").status_code == 404
    (main.DATA / "eval.json").write_text('{"results": []}')
    assert client.get("/eval").json() == {"results": []}


def test_health(client):
    assert client.get("/health").json() == {"ok": True}
