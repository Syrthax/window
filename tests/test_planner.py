import json

import pandas as pd
import pytest
import requests

from agent import fallback, planner, tools

NOW = pd.Timestamp("2026-10-12 05:30")


def make_df(hours=range(6, 18), score_hot_at=12):
    rows = []
    for h in range(0, 24):
        day = 1 if 6 <= h < 18 else 0
        rows.append(dict(
            time=pd.Timestamp("2026-10-12") + pd.Timedelta(hours=h), location="hyderabad",
            apparent_temperature=40.0 if h == score_hot_at else (24.0 if h in (7, 8) else 30.0),
            pm2_5=20.0, precipitation_probability=0.0, precipitation=0.0,
            uv_index=3.0 if day else 0.0, is_day=day))
    return pd.DataFrame(rows)


@pytest.fixture
def good():
    fc = tools.get_forecast(make_df(), NOW)
    w = tools.best_window(fc["hours"])
    place = tools.get_places()[0]["name"]
    return {"start": w["start"], "end": w["end"], "place": place, "activity": "easy walk",
            "message": f"Go {w['start']}-{w['end']} at {place}.", "reason": "Cool and clean."}


def scripted(*replies):
    calls = []

    def chat(messages):
        calls.append(messages)
        r = replies[min(len(calls) - 1, len(replies) - 1)]
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps(r)
    chat.calls = calls
    return chat


def test_best_window_picks_coolest_consecutive_hours():
    fc = tools.get_forecast(make_df(), NOW)
    assert tools.best_window(fc["hours"])["start"] == "07:00"
    assert tools.best_window(fc["hours"])["end"] == "09:00"


def test_forecast_rolls_to_tomorrow_when_day_is_over():
    two_days = pd.concat([make_df(), make_df().assign(time=lambda d: d.time + pd.Timedelta(days=1))])
    fc = tools.get_forecast(two_days, pd.Timestamp("2026-10-12 19:10"))
    assert fc["day"] == "tomorrow" and fc["date"] == "2026-10-13"


def test_forecast_raises_with_no_future_data():
    with pytest.raises(ValueError):
        tools.get_forecast(make_df(), pd.Timestamp("2026-10-12 19:10"))


def test_places_sorted_by_distance():
    km = [p["km"] for p in tools.get_places(n=14)]
    assert km == sorted(km) and len(km) == 14


def test_valid_reply_used_first_try(good):
    chat = scripted(good)
    out = planner.plan(make_df(), NOW, chat=chat, backoff=0)
    assert out["source"] == "ollama" and out["plan"]["place"] == good["place"] and len(chat.calls) == 1


def test_invalid_json_then_valid_retries_once(good):
    chat = scripted("not json {", good)
    out = planner.plan(make_df(), NOW, chat=chat, backoff=0)
    assert out["source"] == "ollama" and len(chat.calls) == 2
    assert "Invalid" in chat.calls[1][-1]["content"]


def test_two_bad_replies_fall_back(good):
    out = planner.plan(make_df(), NOW, chat=scripted("nope", "still nope"), backoff=0)
    assert out["source"] == "fallback" and len(out["plan"]["message"]) <= 140


def test_wrong_window_is_rejected(good):
    bad = {**good, "start": "13:00"}
    out = planner.plan(make_df(), NOW, chat=scripted(bad, bad), backoff=0)
    assert out["source"] == "fallback" and out["plan"]["start"] == "07:00"


def test_unknown_place_is_rejected(good):
    bad = {**good, "place": "Narnia"}
    assert planner.plan(make_df(), NOW, chat=scripted(bad, bad), backoff=0)["source"] == "fallback"


def test_long_message_is_rejected(good):
    bad = {**good, "message": "x" * 141}
    assert planner.plan(make_df(), NOW, chat=scripted(bad, bad), backoff=0)["source"] == "fallback"


def test_ollama_down_falls_back_after_retry():
    chat = scripted(requests.ConnectionError("refused"))
    out = planner.plan(make_df(), NOW, chat=chat, backoff=0)
    assert out["source"] == "fallback" and len(chat.calls) == 2


def test_latency_logged_per_stage(good):
    out = planner.plan(make_df(), NOW, chat=scripted(good), backoff=0)
    assert {"forecast", "places", "llm1"} <= set(out["latency_ms"])


def test_fallback_message_within_limit_for_long_names():
    w = tools.best_window(tools.get_forecast(make_df(), NOW)["hours"])
    msg = fallback.build(w, [{"name": "N" * 200, "shade": True, "km": 1}], "today")["message"]
    assert len(msg) <= 140


def smoggy_df():
    return make_df().assign(pm2_5=45.0)  # AQI ~126 all day


def test_poor_air_adds_caution_flag():
    w = tools.best_window(tools.get_forecast(smoggy_df(), NOW)["hours"])
    assert w["caution"] == ["air"] and tools.best_window(tools.get_forecast(make_df(), NOW)["hours"])["caution"] == []


def test_fallback_leads_with_caution():
    out = planner.plan(smoggy_df(), NOW, chat=scripted("nope", "nope"), backoff=0)
    msg = out["plan"]["message"]
    assert out["source"] == "fallback" and msg.startswith("Best of a smoggy day") and "limit time" in msg
    assert len(msg) <= 140


def test_cheerful_message_rejected_under_caution(good):
    cheerful = {**good, "message": f"Enjoy a lovely walk {good['start']}-{good['end']} at {good['place']}!"}
    out = planner.plan(smoggy_df(), NOW, chat=scripted(cheerful, cheerful), backoff=0)
    assert out["source"] == "fallback"


def test_cautious_message_accepted(good):
    w = tools.best_window(tools.get_forecast(smoggy_df(), NOW)["hours"])
    msg = f"Best of a smoggy day: {w['start']}-{w['end']} at {good['place']}. Air is poor; sensitive people limit time."
    out = planner.plan(smoggy_df(), NOW, chat=scripted({**good, "message": msg}), backoff=0)
    assert out["source"] == "ollama" and out["plan"]["message"] == msg


def test_field_names_in_message_are_rejected(good):
    bad = {**good, "message": "Go at 07:00-09:00, the max_aqi is low."}
    out = planner.plan(make_df(), NOW, chat=scripted(bad, bad), backoff=0)
    assert out["source"] == "fallback"


def test_reason_is_computed_not_taken_from_model(good):
    out = planner.plan(make_df(), NOW, chat=scripted({**good, "reason": "lowest max_aqi, trust me"}), backoff=0)
    assert out["source"] == "ollama"
    r = out["plan"]["reason"]
    assert r.startswith("Highest outdoor score of the day") and "trust me" not in r and "max_aqi" not in r
    assert "feels like" in r and "on average" in r


def test_fallback_reason_matches_model_path_reason(good):
    a = planner.plan(make_df(), NOW, chat=scripted(good), backoff=0)["plan"]["reason"]
    b = planner.plan(make_df(), NOW, chat=scripted("x", "y"), backoff=0)["plan"]["reason"]
    assert a == b


def test_activity_must_be_short(good):
    bad = {**good, "activity": f"{good['place']}, 07:00-09:00"}
    assert planner.plan(make_df(), NOW, chat=scripted(bad, bad), backoff=0)["source"] == "fallback"
