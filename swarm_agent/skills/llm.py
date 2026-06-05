"""Proposer adapter — runs a single-shot swarm agent and returns its text.

This is the ONLY bridge between the standalone skill system and swarm-agent's runtime: it
drives `fleet.compat.make_agent` (swarm-agent's own agent driver, the same one workers use),
NOT HermesAgent's curator/auxiliary client. fleet is imported lazily so the skills package
stays importable (and unit-testable with stub proposers) without fleet present.

Pattern: the LLM PROPOSES a JSON plan; the harness APPLIES it via manager.* — so no
model-callable skill tool is ever added to HermesAgent.
"""
from __future__ import annotations

from typing import Callable


def make_proposer(lane: str = "writer", max_tokens: int = 4096,
                  max_iterations: int = 2) -> Callable[[str], str]:
    """Return propose(prompt)->text using a tool-free swarm lane (writer = 0 tools).

    'writer' has an empty toolset, so the agent just reasons and emits the JSON plan — it
    cannot (and must not) touch the filesystem; the harness applies the plan."""
    def propose(prompt: str) -> str:
        from fleet import compat
        from fleet.worker import _final_text
        agent = compat.make_agent(lane, task_id=f"skill-{lane}",
                                  max_iterations=max_iterations, max_tokens=max_tokens)
        result = agent.run_conversation(prompt, task_id=lane)
        if result.get("failed") or result.get("error"):
            raise RuntimeError(str(result.get("error") or "agent failed"))
        return _final_text(result.get("messages"), result.get("final_response")) or ""
    return propose
