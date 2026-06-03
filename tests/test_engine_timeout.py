"""ThreadFleet must abandon blocked workers without wedging the run loop."""

from __future__ import annotations

import time
from types import SimpleNamespace

from fleet.board import Board, Task
from fleet.engine import ThreadFleet


def _cfg(task_timeout_s: float) -> SimpleNamespace:
    return SimpleNamespace(
        ENROLL_MAX=1,
        TARGET_INFLIGHT=1,
        OVERSUB_FACTOR=1,
        MAX_RETRIES=0,
        METRICS_URL="http://127.0.0.1:1/metrics",
        TASK_TIMEOUT_S=task_timeout_s,
    )


def _result(spec) -> dict:
    return {"id": spec["id"], "text": f"done: {spec['id']}"}


def test_blocked_worker_is_abandoned_without_wedging_run() -> None:
    board = Board()
    board.add_many([
        Task(id="blocked", prompt="simulate a foreground server"),
        Task(id="reduce", prompt="reduce", deps=["blocked"], lane="reducer"),
    ])

    def worker(spec):
        if spec["id"] == "blocked":
            time.sleep(10)
        return _result(spec)

    started = time.monotonic()
    summary = ThreadFleet(board, None, cfg=_cfg(0.5), worker_fn=worker).run()
    elapsed = time.monotonic() - started

    assert elapsed < 3
    assert summary["counts"]["failed"] >= 1 or summary["stranded"] >= 1
    assert board.results()["blocked"].error == "timeout: exceeded FLEET_TASK_TIMEOUT=0.5s"


def test_fast_workers_do_not_false_timeout() -> None:
    board = Board()
    board.add_many([
        Task(id="leaf", prompt="quick"),
        Task(id="reduce", prompt="reduce", deps=["leaf"], lane="reducer"),
    ])

    summary = ThreadFleet(board, None, cfg=_cfg(0.5), worker_fn=_result).run()

    assert summary["counts"]["done"] == 2
    assert summary["counts"]["failed"] == 0
    assert summary["stranded"] == 0


def test_disabled_timeout_preserves_normal_completion() -> None:
    board = Board()
    board.add(Task(id="leaf", prompt="quick"))

    summary = ThreadFleet(board, None, cfg=_cfg(0), worker_fn=_result).run()

    assert summary["counts"]["done"] == 1
    assert summary["counts"]["failed"] == 0
