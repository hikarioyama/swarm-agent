import os

import pytest

from fleet import v3, v3_sleep


@pytest.fixture(autouse=True)
def _clean_v3(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SWARM_V3"):
            monkeypatch.delenv(key, raising=False)
    v3.reset_flags_cache()
    yield
    for key in list(os.environ):
        if key.startswith("SWARM_V3"):
            monkeypatch.delenv(key, raising=False)
    v3.reset_flags_cache()


def _enable(monkeypatch, tmp_path, *, sleep: bool) -> None:
    for key in list(os.environ):
        if key.startswith("SWARM_V3"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SWARM_V3", "1")
    monkeypatch.setenv("SWARM_V3_SLEEP_PATH", str(tmp_path / "sleep.json"))
    if sleep:
        monkeypatch.setenv("SWARM_V3_SLEEP", "1")
    v3.reset_flags_cache()


def _signal(stance: str) -> dict:
    return {
        "hypothesis": stance,
        "stance_hash": stance,
        "evidence_ids": [],
        "confidence": 0.9,
        "contradictions": [],
        "toxins": [],
    }


def _reduce_stances(domain: str, decoy_stance: str, truth_stance: str) -> str:
    signals = [_signal(decoy_stance), _signal(decoy_stance), _signal(decoy_stance), _signal(truth_stance)]
    weights = v3.weighted_stance_counts(
        signals,
        domain,
        is_suppressed=v3_sleep.is_suppressed,
    )
    return max(weights.items(), key=lambda kv: (kv[1], kv[0]))[0]


def test_sleep_unit_consolidates_trap_rules_and_off_fallback(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path, sleep=False)
    v3_sleep.record_trap("billing", "decoy")
    assert v3_sleep.consolidate() == 0
    assert v3_sleep.is_suppressed("billing", "decoy") is False

    _enable(monkeypatch, tmp_path, sleep=True)
    v3_sleep.reset()
    v3_sleep.record_trap("billing", "decoy")
    assert v3_sleep.consolidate() == 0
    assert v3_sleep.is_suppressed("billing", "decoy") is False
    v3_sleep.record_trap("billing", "decoy")
    assert v3_sleep.consolidate() == 1
    assert v3_sleep.is_suppressed("billing", "decoy") is True
    assert v3_sleep.is_suppressed("billing", "truth") is False


def test_sleep_epoch_suppresses_replayed_decoy_majorities(monkeypatch, tmp_path):
    scenarios = [
        ("billing", "stance:billing-decoy", "stance:billing-truth"),
        ("billing", "stance:billing-decoy", "stance:billing-truth"),
        ("scheduler", "stance:scheduler-decoy", "stance:scheduler-truth"),
        ("scheduler", "stance:scheduler-decoy", "stance:scheduler-truth"),
        ("cache", "stance:cache-decoy", "stance:cache-truth"),
        ("cache", "stance:cache-decoy", "stance:cache-truth"),
        ("parser", "stance:parser-decoy", "stance:parser-truth"),
        ("parser", "stance:parser-decoy", "stance:parser-truth"),
    ]

    _enable(monkeypatch, tmp_path, sleep=False)
    off_decoys = sum(
        _reduce_stances(domain, decoy, truth) == decoy
        for domain, decoy, truth in scenarios
    )
    off_rate = off_decoys / len(scenarios)

    _enable(monkeypatch, tmp_path, sleep=True)
    v3_sleep.reset()
    for domain, decoy, _truth in scenarios:
        v3_sleep.record_trap(domain, decoy)
    assert v3_sleep.consolidate() == 4

    on_decoys = sum(
        _reduce_stances(domain, decoy, truth) == decoy
        for domain, decoy, truth in scenarios
    )
    on_rate = on_decoys / len(scenarios)

    assert off_rate == 1.0
    assert on_rate == 0.0
    assert on_rate <= 0.25
    assert on_rate < off_rate


def test_sleep_off_is_not_suppressed_and_weighting_is_identity(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path, sleep=False)
    signals = [_signal("decoy"), _signal("decoy"), _signal("truth")]
    weights = v3.weighted_stance_counts(signals, "billing", is_suppressed=v3_sleep.is_suppressed)
    assert v3_sleep.is_suppressed("billing", "decoy") is False
    assert weights == {"decoy": 2.0, "truth": 1.0}
