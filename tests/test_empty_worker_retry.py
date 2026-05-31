"""Empty model output must not be recorded as a successful board result."""

from __future__ import annotations

from types import SimpleNamespace

import fleet.engine as engine
from fleet.board import Board, Task


def test_empty_visible_response_fails_task() -> None:
    board = Board()
    board.add(Task(id="empty", prompt="return text"))
    original = engine.run_task_local
    engine.run_task_local = lambda spec: {"id": spec["id"], "text": "(empty)"}
    cfg = SimpleNamespace(
        ENROLL_MAX=1, TARGET_INFLIGHT=1, OVERSUB_FACTOR=1, MAX_RETRIES=0,
        METRICS_URL="http://127.0.0.1:1/metrics",
    )
    try:
        result = engine.ThreadFleet(board, None, cfg=cfg).run()
    finally:
        engine.run_task_local = original
    assert result["counts"]["failed"] == 1
    assert "empty visible response" in board.results()["empty"].error


if __name__ == "__main__":
    test_empty_visible_response_fails_task()
    print("empty worker retry smoke passed")
