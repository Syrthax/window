import pytest

from notify import listen

PLAN = {"day": "tomorrow", "plan": {"message": "Best of a smoggy day: ...", "place": "KBR National Park",
        "activity": "easy walk", "start": "06:00", "end": "08:00", "reason": "Highest score of the day."},
        "window": {}}
OWNER = "42"


def upd(text, chat=42, key="message"):
    return {"update_id": 1, key: {"chat": {"id": chat}, "text": text}}


def ok():
    return PLAN


def test_today_returns_the_message():
    assert listen.handle(upd("/today"), OWNER, ok) == ("42", "Best of a smoggy day: ...")


def test_why_has_place_window_and_reason():
    _, text = listen.handle(upd("/why"), OWNER, ok)
    assert "KBR National Park" in text and "06:00-08:00" in text and "Highest score" in text


@pytest.mark.parametrize("cmd", ["/start", "/help", "/today@ClimateHelp_bot", "/TODAY"])
def test_commands_work_with_bot_suffix_and_case(cmd):
    assert listen.handle(upd(cmd), OWNER, ok) is not None


def test_plain_text_gets_a_hint():
    _, text = listen.handle(upd("hi"), OWNER, ok)
    assert "/today" in text


def test_other_chats_are_ignored_and_plan_not_computed():
    def boom():
        raise AssertionError("must not plan for strangers")
    assert listen.handle(upd("/today", chat=999), OWNER, boom) is None


def test_non_text_and_non_message_updates_ignored():
    assert listen.handle({"update_id": 2, "message": {"chat": {"id": 42}, "photo": []}}, OWNER, ok) is None
    assert listen.handle(upd("/today", key="edited_message"), OWNER, ok) is None


def test_failure_still_answers():
    def fail():
        raise ValueError("no data")
    _, text = listen.handle(upd("/today"), OWNER, fail)
    assert "could not" in text and "ValueError" in text


def test_help_never_plans():
    def boom():
        raise AssertionError("help must not plan")
    assert listen.handle(upd("/help"), OWNER, boom) is not None
    assert listen.handle(upd("hello there"), OWNER, boom) is not None
