"""Inbound routing: a Telegram message is treated IDENTICALLY to TUI typing (SWARM_V2 §C.2).

A bare message becomes a turn via ``submit`` — or, if a turn is already in flight, it STEERS
the running turn instead of being dropped (§C.3). Slash commands mirror the TUI's: ``/task`` →
``enqueue_task``, ``/stop`` → ``interrupt``, ``/tasks`` → a queue snapshot. The bridge does the
chat-id allowlist check BEFORE calling this (security-critical, §C.5/§4.5); routing assumes the
sender is already authorised."""
from __future__ import annotations

from typing import Optional


def route_inbound(runner, text: str) -> Optional[str]:
    """Route one authorised inbound message through the SAME runner methods a TUI keypress
    uses, and return a short reply to send back (or None). Never raises."""
    text = (text or "").strip()
    if not text:
        return None

    if text.startswith("/task"):
        goal = text[len("/task"):].strip()
        if not goal:
            return "usage: /task <goal>"
        runner.enqueue_task(goal)
        return f"queued: {goal[:80]}"

    if text == "/stop":
        n = runner.interrupt()
        return f"interrupted {n} agent(s)"

    if text == "/tasks":
        try:
            counts = runner.tasks.counts()
        except Exception:
            counts = {}
        return f"queue: {counts}"

    # Bare text → a turn. If the single interactive slot is busy, submit() returns None; rather
    # than drop the message we STEER it into the running turn (§C.3) and say so.
    thread = runner.submit(text)
    if thread is None:
        runner.steer(text)
        return "still working — steered your message into the running turn"
    return "on it"
