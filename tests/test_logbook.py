"""Offline tests for persistent swarm event logging."""

from __future__ import annotations

import json

from swarm_agent import logbook
from swarm_agent.logbook import SwarmLogger
from swarm_agent.runner import SwarmRunner


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_logger_writes_timestamped_sequenced_json_lines(tmp_path) -> None:
    logger = SwarmLogger(path=str(tmp_path / "events.jsonl"))
    logger.log({"kind": "planning", "goal_id": "g1"})
    logger.event("final", text="done")
    logger.close()

    rows = _read(logger.path)
    assert [row["seq"] for row in rows] == [1, 2]
    assert all(row["ts"] and row["sid"] == logger.sid for row in rows)
    assert rows[0]["kind"] == "planning"
    assert rows[0]["goal_id"] == "g1"
    assert rows[1]["kind"] == "final"
    assert rows[1]["text"] == "done"


def test_logger_caps_regular_fields_but_not_error_details(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(logbook, "_FIELD_CAP", 5)
    logger = SwarmLogger(path=str(tmp_path / "events.jsonl"))
    logger.log({"kind": "error", "text": "abcdefgh", "error": "abcdefgh",
                "detail": "abcdefgh"})
    logger.close()

    row = _read(logger.path)[0]
    assert row["text"] == "abcde…[+3 chars]"
    assert row["error"] == "abcdefgh"
    assert row["detail"] == "abcdefgh"


def test_logger_can_be_disabled_without_creating_a_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SWARM_EVENT_LOG", "0")
    logger = SwarmLogger(path=str(tmp_path / "events.jsonl"))
    assert not logger.enabled
    logger.log({"kind": "planning"})
    logger.close()
    assert not logger.path.exists()


def test_logger_stringifies_non_json_values_without_raising(tmp_path) -> None:
    logger = SwarmLogger(path=str(tmp_path / "events.jsonl"))
    logger.log({"kind": "object", "value": object()})
    logger.close()
    assert isinstance(_read(logger.path)[0]["value"], str)


def test_runner_logs_session_lifecycle_and_emitted_events(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SWARM_LOG_DIR", str(tmp_path))
    runner = SwarmRunner(warm=False, admission="static")
    runner.emit("planning", goal_id="g1")
    rows = _read(runner.log.path)
    assert any(row["kind"] == "session_start" for row in rows)
    assert any(row["kind"] == "planning" and row["goal_id"] == "g1" for row in rows)

    runner.shutdown()
    rows = _read(runner.log.path)
    assert any(row["kind"] == "session_end" for row in rows)


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__]))
