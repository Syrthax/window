import json
import math

from model.features import V0
from model.predict import load_config, stacked


def test_stacked_uses_weights():
    assert stacked({"model": 0.5, "persistence": 0.5, "yesterday": 0.0},
                   {"model": 10.0, "persistence": 20.0, "yesterday": 99.0}) == 15.0


def test_stacked_renormalises_when_member_missing():
    assert stacked({"model": 0.5, "persistence": 0.0, "yesterday": 0.5},
                   {"model": 10.0, "persistence": 20.0, "yesterday": math.nan}) == 10.0


def test_load_config_falls_back_without_eval(tmp_path):
    spec, weights = load_config(tmp_path / "missing.json")
    assert spec == V0 and weights == {}


def test_load_config_reads_selection(tmp_path):
    p = tmp_path / "eval.json"
    p.write_text(json.dumps({"selected": {"name": "x", "extended": True, "delta": True, "keep_missing_weather": False},
                             "results": [{"horizon_h": 3, "weights": {"model": 0.7, "persistence": 0.3, "yesterday": 0.0}}]}))
    spec, weights = load_config(p)
    assert spec.delta and spec.extended and weights[3]["model"] == 0.7
