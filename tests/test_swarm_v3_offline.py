import json
import os
import random

import pytest

from fleet import v3
from fleet.board import Board, State, Task


THRESHOLDS = {
    "accuracy_v3": 0.80,
    "accuracy_v2": 0.55,
    "accuracy_delta": 0.30,
    "herding_v3_truth_rate": 0.70,
    "herding_v2_truth_rate": 0.35,
    "escalation_recall": 0.875,
    "false_escalation": 1 / 8,
    "quorum_fired": 20,
    "overhead": 1.8,
}


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


def _chem(hypothesis, stance, confidence=0.8, contradictions=None):
    return json.dumps({
        "hypothesis": hypothesis,
        "stance_hash": stance,
        "evidence_ids": ["e1"],
        "confidence": confidence,
        "contradictions": contradictions or [],
        "toxins": [],
    })


def _answer(answer, *, chem=None):
    block = f"\n```json\n{chem}\n```" if chem else ""
    return f"ANSWER:{answer}{block}"


def deceptive_parallel_bughunt(seed=1337):
    scenarios = []
    for i in range(24):
        scenarios.append({
            "id": f"herd-{i}",
            "kind": "herding",
            "truth_bug": f"tenant-cache-{i}",
            "decoy_bug": f"parser-regression-{i}",
            "evidence_shards": ["surface-a", "surface-b", "surface-c", "decisive"],
            "misleading_majority": True,
            "decisive_evidence": True,
            "requires_escalation": False,
            # 21/24 herding cases force one-stance consensus; 21 >= the 20/24 quorum threshold.
            "low_diversity_trap": i < 21,
        })
    for i in range(8):
        scenarios.append({
            "id": f"clean-{i}",
            "kind": "clean",
            "truth_bug": f"bounds-check-{i}",
            "decoy_bug": f"formatting-noise-{i}",
            "evidence_shards": ["direct", "support", "negative-control"],
            "misleading_majority": False,
            "decisive_evidence": True,
            "requires_escalation": False,
            "low_diversity_trap": False,
        })
    for i in range(8):
        scenarios.append({
            "id": f"esc-{i}",
            "kind": "escalation",
            "truth_bug": "ESCALATE",
            "decoy_bug": f"deadlock-decoy-{i}",
            "evidence_shards": ["surface-a", "surface-b", "surface-c", "contradiction"],
            "misleading_majority": True,
            "decisive_evidence": True,
            "requires_escalation": True,
            "low_diversity_trap": True,
        })
    random.Random(seed).shuffle(scenarios)
    return scenarios


class SeededWorkers:
    def __init__(self):
        self.calls = 0

    def run(self, task):
        self.calls += 1
        meta = task["meta"]
        scenario = meta["scenario"]
        profile = meta.get("profile")
        lane = task["lane"]
        if lane == "reducer":
            return {"text": self._reduce(task)}
        if lane == "referee":
            return {"text": self._referee(scenario)}
        if profile == "contrarian":
            return {"text": self._contrarian(scenario)}
        return {"text": self._same_prior(scenario, profile)}

    def _same_prior(self, scenario, profile):
        if scenario["kind"] == "clean":
            if profile == "same_prior_2":
                ans = f"negative-control-{scenario['id']}"
            else:
                ans = scenario["truth_bug"]
        elif scenario["kind"] == "herding" and not scenario["low_diversity_trap"]:
            ans = scenario["truth_bug"]
        else:
            ans = scenario["decoy_bug"]
        return _answer(ans, chem=_chem(ans, v3.canonical_stance(ans), 0.82))

    def _contrarian(self, scenario):
        ans = scenario["truth_bug"] if scenario["decisive_evidence"] else scenario["decoy_bug"]
        return _answer(ans, chem=_chem(ans, v3.canonical_stance(ans), 0.9))

    def _referee(self, scenario):
        ans = "ESCALATE" if scenario["requires_escalation"] else scenario["truth_bug"]
        contradictions = [v3.canonical_stance(scenario["decoy_bug"])]
        return _answer(ans, chem=_chem(ans, v3.canonical_stance(ans), 0.95, contradictions))

    def _reduce(self, task):
        counts = {}
        meta = task["meta"]
        for text in (meta.get("dep_results") or {}).values():
            ans = text.split("ANSWER:", 1)[1].splitlines()[0].strip()
            if ans == "ESCALATE":
                return "FINAL:ESCALATE"
            counts[ans] = counts.get(ans, 0) + 1
        answer = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        if any(
            "ANSWER:" + meta["scenario"]["truth_bug"] in text
            and ":v3_referee:" in dep
            for dep, text in (meta.get("dep_results") or {}).items()
        ):
            answer = meta["scenario"]["truth_bug"]
        return f"FINAL:{answer}"


def _after_complete(board, tid, text):
    if not v3.any_on():
        return None
    chem = v3.parse_chem(text)
    board.record_signal(tid, chem)
    spawned = None
    if v3.enabled("diversity"):
        snapshot = board.results()
        for reducer_tid, task in snapshot.items():
            if task.lane != "reducer" or tid not in task.deps:
                continue
            if all(snapshot[d].state == State.DONE for d in task.deps):
                spawned = board.spawn_referee(reducer_tid) or spawned
    return spawned


class SwarmSim:
    def __init__(self, *, use_v3):
        self.use_v3 = use_v3
        self.workers = SeededWorkers()
        self.quorum_fires = 0

    def run_one(self, scenario):
        board = Board()
        profiles = ["same_prior_0", "same_prior_1", "same_prior_2"]
        if scenario["kind"] == "clean":
            profiles = ["same_prior_0", "same_prior_1", "same_prior_2"]
        for idx, profile in enumerate(profiles):
            board.add(Task(
                id=f"{scenario['id']}:w{idx}",
                prompt=f"{profile} inspect {scenario['id']}",
                lane="worker",
                meta={"scenario": scenario, "profile": profile},
            ))
        deps = [f"{scenario['id']}:w{i}" for i in range(len(profiles))]
        reducer_id = f"{scenario['id']}:reduce"
        board.add(Task(
            id=reducer_id,
            prompt=f"reduce {scenario['id']}",
            deps=deps,
            lane="reducer",
            meta={"scenario": scenario},
        ))

        while board.unfinished() > 0:
            ready = board.claim_ready(1)
            assert ready, board.counts()
            task = ready[0]
            result = self.workers.run(task.spec())["text"]
            board.complete(task.id, result)
            spawned = _after_complete(board, task.id, result)
            if spawned:
                ref = board.results()[spawned]
                ref.meta = dict(ref.meta, scenario=scenario, profile="referee")
                board._tasks[spawned] = ref
                self.quorum_fires += 1
        final = board.results()[reducer_id].result
        return final.split("FINAL:", 1)[1]

    def run(self, scenarios):
        answers = {}
        for scenario in scenarios:
            answers[scenario["id"]] = self.run_one(scenario)
        return answers


def _set_flags(monkeypatch, *, use_v3):
    for key in list(os.environ):
        if key.startswith("SWARM_V3"):
            monkeypatch.delenv(key, raising=False)
    if use_v3:
        monkeypatch.setenv("SWARM_V3", "1")
        monkeypatch.setenv("SWARM_V3_CHEMICAL", "1")
        monkeypatch.setenv("SWARM_V3_DIVERSITY", "1")
    v3.reset_flags_cache()


def test_swarm_v3_beats_v2_on_deceptive_parallel_bughunt(monkeypatch):
    scenarios = deceptive_parallel_bughunt(seed=1337)

    _set_flags(monkeypatch, use_v3=False)
    v2 = SwarmSim(use_v3=False)
    answers_v2 = v2.run(scenarios)

    _set_flags(monkeypatch, use_v3=True)
    v3sim = SwarmSim(use_v3=True)
    answers_v3 = v3sim.run(scenarios)

    def correct(answers, subset):
        return sum(answers[s["id"]] == s["truth_bug"] for s in subset) / len(subset)

    herding = [s for s in scenarios if s["kind"] == "herding"]
    clean = [s for s in scenarios if s["kind"] == "clean"]
    escalation = [s for s in scenarios if s["kind"] == "escalation"]
    accuracy_v2 = correct(answers_v2, scenarios)
    accuracy_v3 = correct(answers_v3, scenarios)
    v2_truth_rate = correct(answers_v2, herding)
    v3_truth_rate = correct(answers_v3, herding)
    escalation_recall = correct(answers_v3, escalation)
    false_escalation = sum(answers_v3[s["id"]] == "ESCALATE" for s in clean) / len(clean)

    assert accuracy_v3 >= THRESHOLDS["accuracy_v3"]
    assert accuracy_v2 <= THRESHOLDS["accuracy_v2"]
    assert accuracy_v3 - accuracy_v2 >= THRESHOLDS["accuracy_delta"]
    assert v3_truth_rate >= THRESHOLDS["herding_v3_truth_rate"]
    assert v2_truth_rate <= THRESHOLDS["herding_v2_truth_rate"]
    assert escalation_recall >= THRESHOLDS["escalation_recall"]
    assert false_escalation <= THRESHOLDS["false_escalation"]
    assert v3sim.quorum_fires >= THRESHOLDS["quorum_fired"]
    assert v3sim.workers.calls <= THRESHOLDS["overhead"] * v2.workers.calls
