import json
import os

import pytest

from fleet import v3, v3_credit
from fleet.board import Board, State, Task


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


def _enable(monkeypatch, tmp_path, *, hebbian: bool) -> None:
    for key in list(os.environ):
        if key.startswith("SWARM_V3"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SWARM_V3", "1")
    monkeypatch.setenv("SWARM_V3_CHEMICAL", "1")
    monkeypatch.setenv("SWARM_V3_DIVERSITY", "1")
    monkeypatch.setenv("SWARM_V3_CREDIT_PATH", str(tmp_path / "credit.json"))
    if hebbian:
        monkeypatch.setenv("SWARM_V3_HEBBIAN", "1")
    v3.reset_flags_cache()


def _chem(stance: str, confidence: float = 0.9) -> dict:
    return {
        "hypothesis": stance,
        "stance_hash": stance,
        "evidence_ids": [],
        "confidence": confidence,
        "contradictions": [],
        "toxins": [],
    }


def _trap_board(domain: str) -> Board:
    board = Board()
    for idx in range(2):
        board.add(Task(
            id=f"{domain}:w{idx}",
            prompt="worker",
            state=State.DONE,
            result=f"ANSWER:decoy-{domain}\n```json\n{json.dumps(_chem('decoy-' + domain))}\n```",
            meta={"chem": _chem(f"decoy-{domain}")},
        ))
    board.add(Task(
        id=f"{domain}:reduce",
        prompt="reduce",
        deps=[f"{domain}:w0", f"{domain}:w1"],
        lane="reducer",
        meta={"domain": domain},
    ))
    return board


def _spawned_kind(domain: str) -> str:
    board = _trap_board(domain)
    reducer_id = f"{domain}:reduce"
    spawned = board.spawn_referee(reducer_id)
    assert spawned is not None
    return board.results()[spawned].meta["v3_kind"]


def test_credit_unit_stable_order_and_off_fallback(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path, hebbian=False)
    candidates = ["contrarian", "domain_expert", "skeptic"]
    v3_credit.bump_credit("domain_expert", "db", amount=5.0)
    assert v3_credit.credit_score("domain_expert", "db") == 0.0
    assert v3_credit.order_profiles(candidates, "db") == candidates

    _enable(monkeypatch, tmp_path, hebbian=True)
    v3_credit.reset()
    assert v3_credit.order_profiles(candidates, "db") == candidates
    v3_credit.bump_credit("domain_expert", "db", amount=2.0)
    v3_credit.bump_credit("contrarian", "api", amount=1.0)
    assert v3_credit.credit_score("domain_expert", "db") == pytest.approx(2.0)
    assert v3_credit.order_profiles(candidates, "db") == [
        "domain_expert", "contrarian", "skeptic",
    ]
    assert v3_credit.order_profiles(candidates, "api") == [
        "contrarian", "domain_expert", "skeptic",
    ]


def test_spawn_referee_uses_credit_only_when_hebbian_on(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path, hebbian=True)
    v3_credit.reset()
    v3_credit.bump_credit("domain_expert", "db", amount=3.0)
    assert _spawned_kind("db") == "domain_expert"

    _enable(monkeypatch, tmp_path, hebbian=False)
    assert _spawned_kind("db") == "contrarian"


def test_hebbian_epoch_routes_effective_profile_first_and_saves_calls(monkeypatch, tmp_path):
    effective = {"api": "contrarian", "db": "domain_expert"}
    epoch2_domains = ["api", "db", "db", "api", "db", "api", "db", "api"]

    _enable(monkeypatch, tmp_path, hebbian=True)
    v3_credit.reset()
    for domain, winner in effective.items():
        board = _trap_board(domain)
        board.credit_outcome(f"{domain}:reduce", winner, domain)

    learned_calls = 0
    learned_first = {}
    for domain in epoch2_domains:
        first = _spawned_kind(domain)
        learned_first.setdefault(domain, first)
        learned_calls += 1 if first == effective[domain] else 2

    _enable(monkeypatch, tmp_path, hebbian=False)
    control_calls = 0
    control_first = {}
    for domain in epoch2_domains:
        first = _spawned_kind(domain)
        control_first.setdefault(domain, first)
        control_calls += 1 if first == effective[domain] else 2

    assert learned_first == effective
    assert control_first == {"api": "contrarian", "db": "contrarian"}
    assert learned_calls == 8
    assert control_calls == 12
    assert learned_calls < control_calls


def test_hebbian_off_order_profiles_is_identity_and_scores_zero(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path, hebbian=False)
    candidates = ["a", "b", "c"]
    assert v3_credit.credit_score("a", "x") == 0.0
    assert v3_credit.order_profiles(candidates, "x") == candidates
