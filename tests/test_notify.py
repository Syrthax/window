import pandas as pd
import pytest
import requests

from model.predict import blend
from notify import telegram


class Resp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body or {}
        self.ok = status < 400

    def json(self):
        return self._body


@pytest.fixture(autouse=True)
def creds(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.setattr(telegram.time, "sleep", lambda s: None)


def script(monkeypatch, *replies):
    calls = []

    def post(url, json=None, timeout=None):
        calls.append(json)
        r = replies[min(len(calls) - 1, len(replies) - 1)]
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(telegram.requests, "post", post)
    return calls


def test_send_success(monkeypatch):
    calls = script(monkeypatch, Resp())
    telegram.send("hi")
    assert calls == [{"chat_id": "42", "text": "hi"}]


def test_retries_network_error_then_succeeds(monkeypatch):
    calls = script(monkeypatch, requests.ConnectionError("x"), Resp())
    telegram.send("hi")
    assert len(calls) == 2


def test_retries_429_and_5xx(monkeypatch):
    calls = script(monkeypatch, Resp(429, {"parameters": {"retry_after": 1}}), Resp(502), Resp())
    telegram.send("hi")
    assert len(calls) == 3


def test_gives_up_after_retries(monkeypatch):
    calls = script(monkeypatch, Resp(503))
    with pytest.raises(telegram.NotifyError):
        telegram.send("hi", retries=3)
    assert len(calls) == 3


def test_4xx_fails_immediately(monkeypatch):
    calls = script(monkeypatch, Resp(400, {"description": "chat not found"}))
    with pytest.raises(telegram.NotifyError, match="chat not found"):
        telegram.send("hi")
    assert len(calls) == 1


def test_token_never_in_error(monkeypatch):
    script(monkeypatch, requests.ConnectionError("https://api.telegram.org/bot123:abc/sendMessage"))
    with pytest.raises(telegram.NotifyError) as e:
        telegram.send("hi", retries=1)
    assert "123:abc" not in str(e.value)


def test_missing_credentials(monkeypatch):
    monkeypatch.delenv("TELEGRAM_CHAT_ID")
    with pytest.raises(telegram.NotifyError, match="CHAT_ID"):
        telegram.send("hi")


def test_blend_only_touches_predicted_hours():
    t = pd.date_range("2026-10-12", periods=4, freq="h")
    df = pd.DataFrame({"time": t, "pm2_5": [10.0, 20.0, 30.0, 40.0]})
    out = blend(df, {t[1]: 40.0})
    assert out.pm2_5.tolist() == [10.0, 30.0, 30.0, 40.0]
    assert df.pm2_5.tolist() == [10.0, 20.0, 30.0, 40.0]  # input untouched
