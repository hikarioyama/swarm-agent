"""Offline tests for planner DAG validation and sink extraction."""

from __future__ import annotations

from swarm_agent.goal import _extract_json, final_result, tasks_from_json


def test_fenced_json_and_sink() -> None:
    tasks = tasks_from_json(_extract_json("""```json
[
  {"id":"a","prompt":"A","deps":[],"lane":"worker"},
  {"id":"b","prompt":"B","deps":[],"lane":"worker"},
  {"id":"r","prompt":"R","deps":["a","b"],"lane":"reducer"}
]
```"""))
    tasks[-1].result = "done"
    assert final_result({task.id: task for task in tasks}) == "done"


def test_cycle_rejected() -> None:
    try:
        tasks_from_json([
            {"id": "a", "prompt": "A", "deps": ["b"]},
            {"id": "b", "prompt": "B", "deps": ["a"]},
            {"id": "r", "prompt": "R", "deps": ["a"], "lane": "reducer"},
        ])
    except ValueError as exc:
        assert "cycle" in str(exc)
    else:
        raise AssertionError("cycle accepted")


if __name__ == "__main__":
    test_fenced_json_and_sink()
    test_cycle_rejected()
    print("goal front door offline smoke passed")
