import json
import os

import pytest

from fleet import v3
from fleet.board import Board, SqliteBoard, Task


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


def _chem(answer):
    return json.dumps({
        "hypothesis": answer,
        "stance_hash": v3.canonical_stance(answer),
        "evidence_ids": [],
        "confidence": 0.8,
        "contradictions": [],
        "toxins": [],
    })


def _worker(task):
    if task.lane == "reducer":
        vals = []
        for text in (task.meta.get("dep_results") or {}).values():
            vals.append(text.split("ANSWER:", 1)[1].splitlines()[0].strip())
        return f"FINAL:{'+'.join(sorted(vals))}"
    answer = f"ok-{task.id}"
    return f"ANSWER:{answer}\n```json\n{_chem(answer)}\n```"


def _run_all_off(board):
    board.add_many([
        Task(id="a", prompt="A"),
        Task(id="b", prompt="B"),
        Task(id="reduce", prompt="R", deps=["a", "b"], lane="reducer"),
    ])
    while board.unfinished() > 0:
        ready = board.claim_ready(1)
        assert ready, board.counts()
        task = ready[0]
        text = _worker(task)
        board.complete(task.id, text)
        if v3.any_on():
            chem = v3.parse_chem(text)
            board.record_signal(task.id, chem)
    results = board.results()
    reducer = results["reduce"].result
    serial = {
        tid: {
            "state": task.state.value,
            "result": task.result,
            "error": task.error,
            "retries": task.retries,
            "meta": task.meta,
            "deps": task.deps,
            "lane": task.lane,
        }
        for tid, task in results.items()
    }
    return serial, reducer


def _assert_no_v3_meta(serial):
    for task in serial.values():
        for key in task["meta"]:
            assert key != "chem"
            assert not key.startswith("v3_")


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_all_off_claim_ready_identity_order(monkeypatch, tmp_path, backend):
    monkeypatch.setenv("SWARM_V3", "0")
    v3.reset_flags_cache()
    board = Board() if backend == "memory" else SqliteBoard(str(tmp_path / "order.db"))
    try:
        ids = [f"t{i}" for i in range(8)]
        board.add_many([Task(id=tid, prompt=tid) for tid in ids])
        assert [t.id for t in board.claim_ready(8)] == ids
    finally:
        board.close()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_all_off_full_run_has_no_chem_or_v3_meta_and_matches_baseline(monkeypatch, tmp_path, backend):
    monkeypatch.setenv("SWARM_V3", "0")
    v3.reset_flags_cache()
    first = Board() if backend == "memory" else SqliteBoard(str(tmp_path / f"{backend}-1.db"))
    second = Board() if backend == "memory" else SqliteBoard(str(tmp_path / f"{backend}-2.db"))
    try:
        baseline_serial, baseline_reducer = _run_all_off(first)
        serial, reducer = _run_all_off(second)
        assert serial == baseline_serial
        assert reducer == baseline_reducer == "FINAL:ok-a+ok-b"
        _assert_no_v3_meta(serial)
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_chemical_on_without_chem_written_keeps_identity_order(monkeypatch, tmp_path, backend):
    monkeypatch.setenv("SWARM_V3", "1")
    monkeypatch.setenv("SWARM_V3_CHEMICAL", "1")
    v3.reset_flags_cache()
    board = Board() if backend == "memory" else SqliteBoard(str(tmp_path / "chemical.db"))
    try:
        ids = [f"z{i}" for i in range(10)]
        board.add_many([Task(id=tid, prompt=tid) for tid in ids])
        assert [t.id for t in board.claim_ready(10)] == ids
    finally:
        board.close()
