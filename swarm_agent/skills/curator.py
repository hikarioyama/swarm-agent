"""Skill curator — VENDORED concept from HermesAgent agent/curator.py, decoupled.

Two tiers:
  1. apply_automatic_transitions() — PURE state machine (no LLM): walk agent-created skills,
     mark stale (unused > stale_after_days), archive (> archive_after_days), reactivate when
     used again. Skips pinned. Cheap; safe to run on a timer.
  2. run_llm_consolidation(propose_fn) — the self-improving loop: a PROPOSER (a swarm agent,
     injected) returns a structured plan {consolidate/prune/patch}; the harness APPLIES it via
     swarm_agent.skills.manager. No model-callable skill tools, no HermesAgent imports.

State persisted to <skills_dir>/.curator_state (JSON), same shape as the original.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional

from . import _env, usage, manager

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


# ── state ─────────────────────────────────────────────────────────────────────
def load_state() -> Dict[str, Any]:
    path = _env.curator_state_file()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(data: Dict[str, Any]) -> None:
    path = _env.curator_state_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".curator_", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        logger.debug("save_state failed: %s", e)


def should_run_now(now: Optional[datetime] = None) -> bool:
    """True iff curator is enabled and >= interval since last run. First call seeds the
    timestamp and defers (so a fresh install doesn't immediately run)."""
    if not _env.curator_enabled():
        return False
    now = now or _now()
    state = load_state()
    if state.get("paused"):
        return False
    last = _parse(state.get("last_run_at"))
    if last is None:
        state["last_run_at"] = now.isoformat()
        save_state(state)
        return False
    return (now - last) >= timedelta(hours=_env.curator_interval_hours())


# ── tier 1: pure auto-transitions ──────────────────────────────────────────────
def apply_automatic_transitions(now: Optional[datetime] = None) -> Dict[str, int]:
    """Move agent-created skills active⇄stale→archived by inactivity. Pure-ish (mutates the
    telemetry sidecar + may move dirs on archive); no LLM. Returns counts."""
    now = now or _now()
    stale_cut = timedelta(days=_env.stale_after_days())
    arch_cut = timedelta(days=_env.archive_after_days())
    counts = {"archived": 0, "marked_stale": 0, "reactivated": 0, "checked": 0}
    data = usage.load_usage()
    for name in usage.list_agent_created_skill_names():
        rec = data.get(name) or {}
        counts["checked"] += 1
        if rec.get("pinned"):
            continue
        anchor = usage.latest_activity_at(rec) or rec.get("created_at")
        last = _parse(anchor)
        if last is None:
            continue
        idle = now - last
        state = rec.get("state", usage.STATE_ACTIVE)
        if idle > arch_cut:
            ok, _ = usage.archive_skill(name)
            if ok:
                counts["archived"] += 1
        elif idle > stale_cut:
            if state == usage.STATE_ACTIVE:
                usage.set_state(name, usage.STATE_STALE)
                counts["marked_stale"] += 1
        else:
            if state == usage.STATE_STALE:
                usage.set_state(name, usage.STATE_ACTIVE)
                counts["reactivated"] += 1
    return counts


# ── tier 2: LLM consolidation (proposer injected; harness applies) ──────────────
# The proposer is any callable(prompt:str)->str that returns the model's JSON plan text.
# In production this is swarm_agent.skills.llm.run_agent_json bound to fleet.compat; tests
# pass a stub. We APPLY the plan through manager.* — the model never calls a tool directly.

CONSOLIDATION_PROMPT = """You are the skill curator for swarm-agent. Below is the current set \
of agent-created skills (name, state, activity, description). Your job is to keep the \
collection SMALL, GENERAL, and USEFUL — consolidate narrow, overlapping skills into broader \
"umbrella" skills, and prune ones that are redundant or obsolete. Do NOT merely deduplicate; \
prefer class-level instructions over many one-off skills.

Skills:
{skills}

Reply with ONLY a JSON object (no prose), of the form:
{{"actions": [
  {{"op": "patch",  "name": "<skill>", "old_string": "...", "new_string": "..."}},
  {{"op": "create", "name": "<new-umbrella>", "content": "---\\nname: ...\\ndescription: ...\\n---\\n# ..."}},
  {{"op": "delete", "name": "<skill>", "absorbed_into": "<umbrella-or-empty>"}}
]}}
Only include actions you are confident about. An empty list ("actions": []) is valid."""


def _render_candidates() -> str:
    rows = usage.agent_created_report()
    if not rows:
        return "(none)"
    lines = []
    for r in rows:
        lines.append(f"- {r['name']} [state={r.get('state')}, uses={r.get('activity_count')}, "
                     f"last={r.get('last_activity_at')}]")
    return "\n".join(lines)


def _apply_action(act: Dict[str, Any]) -> Dict[str, Any]:
    op = (act.get("op") or "").lower()
    name = act.get("name", "")
    if op == "create":
        return manager.create(name, act.get("content", ""), act.get("category"))
    if op == "patch":
        return manager.patch(name, act.get("old_string", ""), act.get("new_string", ""),
                             bool(act.get("replace_all", False)), act.get("file_path"))
    if op == "edit":
        return manager.edit(name, act.get("content", ""))
    if op == "delete":
        return manager.delete(name, act.get("absorbed_into"))
    return {"success": False, "error": f"unknown op '{op}'"}


def run_llm_consolidation(propose_fn: Callable[[str], str],
                          dry_run: bool = False) -> Dict[str, Any]:
    """Run one consolidation pass. propose_fn(prompt)->json_text. Applies via manager.*."""
    candidates = _render_candidates()
    if candidates == "(none)":
        return {"applied": [], "skipped": "no agent-created skills"}
    prompt = CONSOLIDATION_PROMPT.format(skills=candidates)
    try:
        raw = propose_fn(prompt) or ""
    except Exception as e:
        return {"error": f"proposer failed: {e}", "applied": []}
    plan = _extract_json(raw)
    actions = (plan or {}).get("actions") or []
    applied = []
    for act in actions:
        if not isinstance(act, dict):
            continue
        if dry_run:
            applied.append({"op": act.get("op"), "name": act.get("name"), "dry_run": True})
            continue
        applied.append({"op": act.get("op"), "name": act.get("name"),
                        "result": _apply_action(act)})
    return {"applied": applied, "n_actions": len(actions)}


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if "```" in text:  # strip ``` fences
        import re
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
        if m:
            text = m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def run_curator(propose_fn: Optional[Callable[[str], str]] = None,
                dry_run: bool = False) -> Dict[str, Any]:
    """Full pass: auto-transitions, then (if a proposer is given) LLM consolidation.
    Updates .curator_state. Returns a summary dict."""
    started = _now()
    auto = apply_automatic_transitions(started)
    llm: Dict[str, Any] = {"skipped": "no proposer"}
    if propose_fn is not None:
        llm = run_llm_consolidation(propose_fn, dry_run=dry_run)
    summary = (f"auto: {auto['archived']} archived, {auto['marked_stale']} stale, "
               f"{auto['reactivated']} reactivated; llm: {len(llm.get('applied', []))} actions")
    if not dry_run:
        state = load_state()
        state.update({"last_run_at": started.isoformat(),
                      "last_run_summary": summary,
                      "run_count": int(state.get("run_count", 0)) + 1})
        save_state(state)
    return {"auto": auto, "llm": llm, "summary": summary}
