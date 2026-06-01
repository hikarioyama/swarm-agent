"""Ephemeral worker: runs ONE task through a fresh HermesAgent and dies.

Stateless and narrow-context by design — no long-lived conversation, no shared
state. State lives on the Board, not in any worker's context. This is what keeps
many workers cheap to run concurrently (short context => fits the KV budget and
sits on the favourable part of the throughput curve).

Two entry points, ONE behaviour:
  * `run_task(spec)`        — picklable, builds the AIAgent INSIDE the (child)
                              process. Used by the ProcessPool `Scheduler`
                              (fallback / A-B). Imports run_agent lazily so the
                              child runs under the HermesAgent venv.
  * `run_task_local(spec)`  — in-process (thread) path used by `ThreadFleet`.
                              Builds the agent via `compat.make_agent`, so every
                              thread-safety mitigation (unique session_id, lean
                              toolset, no persistence, bounded turns, the decode
                              gate + decode_s/gatewait_s timers) is applied
                              uniformly. Reads the per-instance fleet counters
                              that `compat.apply` populated during generation.

Both return the SAME shaped dict so the engine / report code is path-agnostic.
"""
from __future__ import annotations
import sys
import time
import uuid
from typing import Any, Dict

from . import config


def has_visible_text(text: str) -> bool:
    """Reject Hermes placeholders produced after exhausted empty-response retries."""
    return str(text or "").strip().lower() not in {"", "(empty)", "[empty]", "<empty>"}


def _final_text(messages, final_response=None) -> str:
    """Best plain-text answer: prefer the loop's ``final_response`` if present,
    else the last assistant message content (string or content-block list)."""
    if isinstance(final_response, str) and final_response.strip():
        return final_response
    for m in reversed(messages or []):
        if m.get("role") == "assistant":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):  # content blocks
                return "".join(b.get("text", "") for b in c if isinstance(b, dict))
    return ""


def _build_prompt(spec: Dict[str, Any]) -> str:
    """Inject upstream task results so data flows along DAG edges (a reducer must
    SEE its children's outputs, not merely run after them)."""
    prompt = spec["prompt"]
    deps = (spec.get("meta") or {}).get("dep_results") or {}
    if deps:
        ctx = "\n\n".join(f"[Result of task {k}]\n{v}" for k, v in deps.items() if v)
        if ctx:
            prompt = f"Context from upstream tasks:\n{ctx}\n\n---\nYour task:\n{prompt}"
    return prompt


def _result_dict(spec, result, t0, *, decode_s=0.0, gatewait_s=0.0) -> Dict[str, Any]:
    """Assemble the common (picklable) worker result. ``tool_s`` is wall time not
    spent generating or waiting on the gate — i.e. in-process tool execution."""
    wall_s = time.time() - t0
    turns = result.get("api_calls") or 0  # each LLM round-trip is a turn
    return {
        "id": spec["id"],
        "completed": bool(result.get("completed")),
        "text": _final_text(result.get("messages"), result.get("final_response")),
        "api_calls": result.get("api_calls"),
        "wall_s": round(wall_s, 2),
        "decode_s": round(decode_s, 2),
        "gatewait_s": round(gatewait_s, 2),
        "tool_s": round(max(0.0, wall_s - decode_s - gatewait_s), 2),
        "turns": turns,
    }


def run_task(spec: Dict[str, Any]) -> Dict[str, Any]:
    """ProcessPool path. Execute one task in this (child) process, returning a
    PICKLABLE result (never the AIAgent itself).

    Passes ``max_iterations`` (BUG FIX — v0.1 never did, so workers ran the
    AIAgent default of 90) and a unique ``session_id`` (TS1 — avoid the
    auto-gen collision that would share sandbox/cwd/process_registry across
    workers). No decode-gate instrumentation here: the gate lives in the
    single-process thread engine; the ProcessPool path is the un-gated fallback.
    """
    if config.HERMES_DIR not in sys.path:
        sys.path.insert(0, config.HERMES_DIR)
    from run_agent import AIAgent  # noqa: E402  (imported in child for env isolation)
    try:
        from fleet import compat
        compat.install_noninteractive_approval()
    except Exception:
        pass

    t0 = time.time()
    prompt = _build_prompt(spec)
    # FIX #8: int(t0) collides on a fast requeue (two attempts of the same id within
    # one second share a session_id → shared sandbox/cwd/registry). Add a uuid so each
    # attempt is unique (ThreadFleet parity with compat.new_session_id).
    sid = f"fleet-{spec['id']}-{int(t0)}-{uuid.uuid4().hex[:8]}"
    # FIX #7: read the bounded-generation cap defensively (config.MAX_TOKENS authored by
    # the config agent; None = model/server default → AIAgent omits the param).
    max_tokens = getattr(config, "MAX_TOKENS", None)

    try:
        from fleet import prompts as _prompts
        _eph = _prompts.lane_system_prompt(spec.get("lane", "worker"))
    except Exception:
        _eph = None

    agent = AIAgent(
        base_url=config.BASE_URL, api_key=config.API_KEY, model=config.MODEL,
        enabled_toolsets=config.toolsets_for(spec.get("lane", "worker")),  # role-minimal tools
        skip_context_files=True,        # no SOUL.md/AGENTS.md in the prefix
        skip_memory=True,               # no persistent-memory injection
        save_trajectories=False,        # no per-turn trajectory disk writes
        quiet_mode=True,                # cut logging noise
        max_iterations=config.MAX_ITERATIONS,  # BUG FIX: bound the turn budget
        tool_delay=0.0,                 # FIX #4: drop the default 1.0s inter-tool sleep
        max_tokens=max_tokens,          # FIX #7: bounded-generation cap (None=default)
        session_id=sid,                 # TS1: unique sandbox/cwd/registry namespace
        ephemeral_system_prompt=_eph,
    )
    result = agent.run_conversation(prompt, task_id=spec["id"])
    return _result_dict(spec, result, t0)


def run_task_local(spec: Dict[str, Any]) -> Dict[str, Any]:
    """In-process (thread) path used by ThreadFleet.

    Builds the AIAgent via ``compat.make_agent`` (all TS mitigations + the
    decode-gate / decode_s wrappers already installed by ``compat.apply``), runs
    one conversation, and reads back the per-instance fleet timers that the
    forwarder wrappers populated:
      ``agent._fleet_decode_s``   — pure generation time (gate wait excluded)
      ``agent._fleet_gatewait_s`` — time blocked on the decode gate
    so ``tool_s`` = wall − decode − gatewait is the in-process tool time.
    """
    from . import compat  # local import: keeps run_task picklable for ProcessPool

    t0 = time.time()
    prompt = _build_prompt(spec)
    agent = compat.make_agent(spec.get("lane", "worker"), task_id=spec["id"])
    # FIX #3: register a per-worker isolated cwd/bash keyed by spec["id"] for the
    # duration of the conversation (terminal/file tools otherwise collapse every
    # task_id to the shared "default" sandbox → concurrent tool-using workers
    # cross-contaminate). run_conversation passes task_id=spec["id"] to the tools, so
    # the override is keyed correctly. No-op fallback when isolation is disabled or the
    # hermes override API is absent (so the lead's no-tool sweep is unaffected).
    with compat.worker_sandbox(spec["id"]):
        result = agent.run_conversation(prompt, task_id=spec["id"])
    return _result_dict(
        spec, result, t0,
        decode_s=getattr(agent, "_fleet_decode_s", 0.0),
        gatewait_s=getattr(agent, "_fleet_gatewait_s", 0.0),
    )
