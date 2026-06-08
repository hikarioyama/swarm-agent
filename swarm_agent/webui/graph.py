"""Protocol graph projection for the read-only swarm web sidecar."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from swarm_agent.dashboard import SwarmView


@dataclass
class _Goal:
    view: SwarmView = field(default_factory=SwarmView)
    emitted_nodes: set[str] = field(default_factory=set)
    emitted_edges: set[tuple[str, str]] = field(default_factory=set)
    last_states: dict[str, str] = field(default_factory=dict)
    planner_added: bool = False
    emitted_goal: bool = False
    label: str = ""
    state: str = "active"


class GraphState:
    """Maintains per-goal SwarmView state and emits protocol patches."""

    def __init__(self) -> None:
        self.goals: dict[str, _Goal] = {}
        self._interactive_turn = 0
        self._current_interactive_key: str | None = None
        self.latest_gate: int | None = None
        self.latest_running: int | None = None

    def reset(self) -> None:
        """Clear all graph and interactive-turn state."""
        self.goals.clear()
        self._interactive_turn = 0
        self._current_interactive_key = None
        self.latest_gate = None
        self.latest_running = None

    def apply(self, ev: dict[str, Any]) -> list[dict[str, Any]]:
        """Ingest one event and return zero or more websocket messages."""
        key = self._goal_key(ev)
        if key is None:
            return []

        goal = self.goals.setdefault(key, _Goal(label=key))
        self._update_label(goal, key, ev)

        if ev.get("kind") == "planning":
            goal.emitted_nodes.clear()
            goal.emitted_edges.clear()
            goal.last_states.clear()
            goal.planner_added = False
            goal.state = "active"

        if ev.get("kind") == "task" and ev.get("event") == "dispatch":
            self.latest_gate = self._maybe_int(ev.get("target"), self.latest_gate)
            self.latest_running = self._maybe_int(ev.get("running"), self.latest_running)
        elif ev.get("kind") == "task" and isinstance(ev.get("counts"), dict):
            self.latest_running = self._maybe_int(
                ev["counts"].get("running"), self.latest_running)

        goal.view.ingest(ev)

        ops: list[dict[str, Any]] = []
        flows: list[dict[str, Any]] = []

        if goal.view.order and not goal.emitted_goal:
            goal.emitted_goal = True
            ops.append({"op": "add_goal", "id": key,
                        "label": goal.label or key, "state": goal.state})

        planned_with_tasks = ev.get("kind") == "planned" and goal.view.order
        if planned_with_tasks:
            self._ensure_planner_node(key, goal, ops)

        for tid in list(goal.view.order):
            node_id = self._ns(key, tid)
            task = goal.view.tasks.get(tid)
            if not task:
                continue
            if node_id not in goal.emitted_nodes:
                goal.emitted_nodes.add(node_id)
                ops.append({"op": "add_node",
                            "node": self._node(key, tid, task)})
                for dep in task.get("deps") or []:
                    self._add_edge(key, goal, self._ns(key, str(dep)), node_id, ops)
            state = str(task.get("state") or "pending")
            if goal.last_states.get(node_id) != state:
                goal.last_states[node_id] = state
                ops.append({"op": "set_state", "id": node_id, "state": state,
                            "wall_s": task.get("wall_s"),
                            "turns": task.get("turns")})

        if planned_with_tasks:
            self._add_planner_roots(key, goal, ops, flows)

        if ev.get("kind") == "task" and ev.get("event") == "done":
            done_id = ev.get("id")
            if done_id:
                done = str(done_id)
                src = goal.view.tasks.get(done) or {}
                lane = str(src.get("lane") or "worker")
                for tid in goal.view.order:
                    task = goal.view.tasks.get(tid) or {}
                    if done in [str(dep) for dep in (task.get("deps") or [])]:
                        flows.append({"from": self._ns(key, done),
                                      "to": self._ns(key, tid),
                                      "lane": lane})

        if ev.get("kind") in {"final", "idle"}:
            goal.state = "complete"
            if goal.emitted_goal:
                ops.append({"op": "goal_state", "id": key, "state": "complete",
                            "summary": goal.view.summary})

        messages: list[dict[str, Any]] = []
        if ops:
            messages.append({"type": "patch", "ops": ops})
        if flows:
            messages.append({"type": "flow", "flows": flows})
        return messages

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        """Return the protocol graph snapshot."""
        out_goals: list[dict[str, Any]] = []
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        for key, goal in self.goals.items():
            if not goal.view.order:
                continue
            out_goals.append({"id": key, "label": goal.label or key,
                              "state": goal.state})
            if goal.planner_added:
                nodes.append(self._planner_node(key))
                for tid in goal.view.order:
                    task = goal.view.tasks.get(tid) or {}
                    if not task.get("deps"):
                        edges.append({"from": self._planner_id(key),
                                      "to": self._ns(key, tid), "goal": key})
            for tid in goal.view.order:
                task = goal.view.tasks.get(tid)
                if not task:
                    continue
                nodes.append(self._node(key, tid, task))
                for dep in task.get("deps") or []:
                    edges.append({"from": self._ns(key, str(dep)),
                                  "to": self._ns(key, tid), "goal": key})
        return {"goals": out_goals, "nodes": nodes, "edges": edges}

    def _goal_key(self, ev: dict[str, Any]) -> str | None:
        goal_id = ev.get("goal_id")
        if goal_id:
            return str(goal_id)
        if ev.get("kind") == "user":
            self._interactive_turn += 1
            self._current_interactive_key = f"turn-{self._interactive_turn}"
            return self._current_interactive_key
        if self._current_interactive_key is None:
            self._interactive_turn += 1
            self._current_interactive_key = f"turn-{self._interactive_turn}"
        return self._current_interactive_key

    def _update_label(self, goal: _Goal, key: str, ev: dict[str, Any]) -> None:
        if ev.get("kind") in {"user", "queued"}:
            text = ev.get("text") or ev.get("goal") or ev.get("prompt")
            if text:
                goal.label = str(text)
        elif not goal.label:
            goal.label = key

    def _ensure_planner_node(self, key: str, goal: _Goal,
                             ops: list[dict[str, Any]]) -> None:
        planner_id = self._planner_id(key)
        if not goal.planner_added:
            goal.planner_added = True
            goal.emitted_nodes.add(planner_id)
            goal.last_states[planner_id] = "done"
            ops.append({"op": "add_node", "node": self._planner_node(key)})

    def _add_planner_roots(self, key: str, goal: _Goal, ops: list[dict[str, Any]],
                           flows: list[dict[str, Any]]) -> None:
        planner_id = self._planner_id(key)
        for tid in goal.view.order:
            task = goal.view.tasks.get(tid) or {}
            if task.get("deps"):
                continue
            to_id = self._ns(key, tid)
            self._add_edge(key, goal, planner_id, to_id, ops)
            flows.append({"from": planner_id, "to": to_id, "lane": "planner"})

    def _add_edge(self, key: str, goal: _Goal, src: str, dst: str,
                  ops: list[dict[str, Any]]) -> None:
        edge_key = (src, dst)
        if edge_key in goal.emitted_edges:
            return
        goal.emitted_edges.add(edge_key)
        ops.append({"op": "add_edge", "edge": {"from": src, "to": dst, "goal": key}})

    def _node(self, key: str, tid: str, task: dict[str, Any]) -> dict[str, Any]:
        deps = [self._ns(key, str(dep)) for dep in (task.get("deps") or [])]
        return {"id": self._ns(key, tid), "goal": key,
                "lane": str(task.get("lane") or "worker"),
                "state": str(task.get("state") or "pending"),
                "prompt": str(task.get("prompt") or ""),
                "wall_s": task.get("wall_s"), "turns": task.get("turns"),
                "deps": deps}

    def _planner_node(self, key: str) -> dict[str, Any]:
        return {"id": self._planner_id(key), "goal": key, "lane": "planner",
                "state": "done", "prompt": "", "wall_s": None,
                "turns": None, "deps": []}

    def _planner_id(self, key: str) -> str:
        return self._ns(key, "__planner__")

    def _ns(self, key: str, tid: str) -> str:
        return f"{key}::{tid}"

    def _maybe_int(self, value: Any, fallback: int | None) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback
