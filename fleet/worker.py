"""Ephemeral worker: runs ONE task through a fresh HermesAgent and dies.

Stateless and narrow-context by design — no long-lived conversation, no shared
state. State lives on the Board, not in any worker's context. This is what keeps
many workers cheap to run concurrently (short context => fits the KV budget and
sits on the favourable part of the throughput curve).

`run_task` is a module-level function so it is picklable for ProcessPoolExecutor;
it imports HermesAgent INSIDE the child process (the child must run under the
HermesAgent venv so `run_agent` and its deps import).
"""
from __future__ import annotations
import sys
import time
from typing import Any, Dict

from . import config


def _final_text(messages) -> str:
    """Last assistant message content as plain text."""
    for m in reversed(messages or []):
        if m.get("role") == "assistant":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):  # content blocks
                return "".join(b.get("text", "") for b in c if isinstance(b, dict))
    return ""


def run_task(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one task. Returns a PICKLABLE result (never the AIAgent itself)."""
    if config.HERMES_DIR not in sys.path:
        sys.path.insert(0, config.HERMES_DIR)
    from run_agent import AIAgent  # noqa: E402  (imported in child for env isolation)

    t0 = time.time()
    # Inject upstream task results (data flow along DAG edges, not just ordering).
    prompt = spec["prompt"]
    deps = (spec.get("meta") or {}).get("dep_results") or {}
    if deps:
        ctx = "\n\n".join(f"[Result of task {k}]\n{v}" for k, v in deps.items() if v)
        prompt = f"Context from upstream tasks:\n{ctx}\n\n---\nYour task:\n{prompt}"

    agent = AIAgent(base_url=config.BASE_URL, api_key=config.API_KEY, model=config.MODEL)
    result = agent.run_conversation(prompt, task_id=spec["id"])
    return {
        "id": spec["id"],
        "completed": bool(result.get("completed")),
        "text": _final_text(result.get("messages")),
        "api_calls": result.get("api_calls"),
        "wall_s": round(time.time() - t0, 2),
    }
