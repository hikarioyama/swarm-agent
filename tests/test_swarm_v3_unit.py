import os
from types import SimpleNamespace

import pytest

from fleet import v3


@pytest.fixture(autouse=True)
def _clean_v3_flags(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SWARM_V3"):
            monkeypatch.delenv(key, raising=False)
    v3.reset_flags_cache()
    yield
    for key in list(os.environ):
        if key.startswith("SWARM_V3"):
            monkeypatch.delenv(key, raising=False)
    v3.reset_flags_cache()


def _task(tid, meta=None):
    return SimpleNamespace(id=tid, meta=meta or {})


def test_parse_chem_strict_minimum_fields_and_clamps_confidence():
    valid = v3.parse_chem(
        'ok\n```json\n{"hypothesis":"alpha wins","confidence":0.75}\n```'
    )
    assert valid is not None
    assert valid["hypothesis"] == "alpha wins"
    assert valid["confidence"] == 0.75
    assert valid["stance_hash"].startswith("h:")

    assert v3.parse_chem('```json\n{"foo":"bar"}\n```') is None
    assert v3.parse_chem("not json") is None
    assert v3.parse_chem("plain {\"hypothesis\":\"x\",\"confidence\":0.4}") is None

    high = v3.parse_chem('```json\n{"stance_hash":"s1","confidence":2}\n```')
    low = v3.parse_chem('```json\n{"stance_hash":"s1","confidence":-1}\n```')
    assert high is not None and high["confidence"] == 1.0
    assert low is not None and low["confidence"] == 0.0


def test_stance_diversity_expected_values():
    assert v3.stance_diversity([]) == 0.0
    assert v3.stance_diversity([{"stance_hash": "a"}] * 3) == 0.0
    assert v3.stance_diversity([
        {"stance_hash": "a"},
        {"stance_hash": "a"},
        {"stance_hash": "b"},
    ]) == pytest.approx(4 / 9, rel=1e-4)
    assert v3.stance_diversity([
        {"stance_hash": "a"},
        {"stance_hash": "b"},
    ]) == 0.5


def test_quorum_decision_paths():
    assert v3.quorum_decision(
        [{"stance_hash": "a", "confidence": 0.9}],
        min_diversity=0.34,
        accept_conf=0.6,
        max_rounds=2,
        rounds_done=0,
    ) == "need_diversity"
    assert v3.quorum_decision(
        [
            {"stance_hash": "a", "confidence": 0.9},
            {"stance_hash": "b", "confidence": 0.8},
        ],
        min_diversity=0.34,
        accept_conf=0.6,
        max_rounds=2,
        rounds_done=0,
    ) == "accept"
    assert v3.quorum_decision(
        [{"stance_hash": "a", "confidence": 0.9}],
        min_diversity=0.34,
        accept_conf=0.6,
        max_rounds=2,
        rounds_done=2,
    ) == "insufficient"
    assert v3.quorum_decision([], max_rounds=2, rounds_done=2) == "insufficient"


def test_order_by_priority_identity_priority_boost_and_stable_ties():
    equal = [_task("a"), _task("b"), _task("c")]
    assert v3.order_by_priority(equal, now=100.0) == equal

    low = _task("low", {"chem": {"stance_hash": "s1"}, "strength": 1.0, "uncertainty": 0.0})
    high = _task("high", {"chem": {"stance_hash": "s2"}, "strength": 2.0, "uncertainty": 1.0})
    assert v3.order_by_priority([low, high], now=100.0)[0] is high

    first = _task("first", {"strength": 2.0})
    second = _task("second", {"strength": 2.0})
    assert v3.order_by_priority([first, second], now=100.0) == [first, second]


def test_master_flag_forces_subflags_off(monkeypatch):
    monkeypatch.setenv("SWARM_V3", "0")
    monkeypatch.setenv("SWARM_V3_CHEMICAL", "1")
    monkeypatch.setenv("SWARM_V3_DIVERSITY", "1")
    v3.reset_flags_cache()

    assert v3.enabled("chemical") is False
    assert v3.enabled("diversity") is False
    assert v3.enabled("reflex") is False
    assert v3.enabled("hebbian") is False
    assert v3.enabled("sleep") is False
    assert v3.any_on() is False
