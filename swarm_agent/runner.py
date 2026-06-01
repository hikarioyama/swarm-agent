"""In-process front door for swarm-agent — the spine.

This is the architecture fix. The old TUI launched ``python -m swarm_agent.cli``
as a SUBPROCESS, merged its stdout+stderr into the transcript (the "junk logs"),
and wrote the reducer deliverable to ``/tmp/swarm-tui-final.txt`` — which it then
NEVER read back, so every answer was invisible ("応答が帰ってこない").

The fleet engine is SYNC+THREADED and single-process by design (BUILD_SPEC §3), so
there is no reason to shell out. ``SwarmRunner`` drives the whole thing IN-PROCESS:

  message ──▶ router (1 call) ──▶ chat?  ──▶ direct reply  (HermesAgent-like)
                               └─▶ swarm? ──▶ planner ──▶ Board + ThreadFleet
                                                              │ on_event
                                                              ▼
                                          structured events ──▶ events queue
                                          final_result(board) ─▶ "final" event

Everything the UI needs flows through ``self.events`` (a thread-safe queue of
dicts). NOTHING is printed. The TUI drains the queue: ``user``/``reply``/``final``/
``error`` paint the chat pane; ``task`` events drive the live swarm dashboard; the
status line pulls ``self.gate.get_limit()`` + ``/metrics`` directly.

Throughput: the runner owns a resizable ``DecodeGate`` started at the measured knee
(``DECODE_GATE_START`` ≈ 40, region C32–64) with the AIMD controller tracking it —
not the old static ``--gate 4`` the TUI used to pass (1/8 of the operating point).
"""
from __future__ import annotations

import json
import queue
import threading
import time
import traceback
import uuid
from typing import Optional

from fleet import compat, config, metrics
from fleet.board import Board
from fleet.engine import ThreadFleet
from fleet.worker import _final_text
from . import goal as goal_mod
from .taskstore import TaskStore


ROUTER_PROMPT = """You are the front door of a high-concurrency software agent swarm.
Decide whether the user's latest message is:
  - "chat": light conversation, a greeting, a clarification, or a question that ONE
    direct reply handles well; or
  - "swarm": a real task worth decomposing into many parallel sub-tasks across files,
    modules, areas, or sources (audit a repo, implement a multi-file feature, research
    several topics, refactor across a package, "do X for every Y").

Lean toward "chat" unless the work is genuinely multi-step or spans many units.

Reply with ONLY a JSON object, nothing else:
  {{"mode":"chat","reply":"<your full reply to the user, in their language>"}}
or
  {{"mode":"swarm"}}

Recent conversation (oldest first):
{history}

User message:
{message}
"""

CHAT_PROMPT = """You are swarm-agent, chatting with Hikari. Be warm, friendly, and
genuinely helpful — a natural, personable tone, concise but never curt. Speak openly and
never refuse to talk about how you work. Reply in the user's own language. Plain text,
no preamble.

Recent conversation (oldest first):
{history}

User message:
{message}
"""

# How many prior turns to feed the router/chat agent for continuity.
_HISTORY_TURNS = 6
# R3: hard cap on retained turns so the resent-every-turn history never grows unbounded
# across a long session. Tail kept comfortably larger than the snippet window.
_HISTORY_CAP = 2 * _HISTORY_TURNS
# Lanes the runtime warms at boot (the roles the front door actually spawns).
_WARM_LANES = ["router", "planner", "worker", "reducer"]


def _short_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _parse_route(text: str) -> Optional[dict]:
    """Tolerant JSON-object parse of a router reply (bare or with surrounding prose)."""
    candidates = [text]
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{"): text.rfind("}") + 1])
    for cand in candidates:
        cand = (cand or "").strip()
        if not cand:
            continue
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


class SwarmRunner:
    """Owns the in-process swarm runtime and turns user messages into events.

    Thread model: ``submit()`` spawns one daemon turn-thread; ``_run_lock`` serialises
    turns (one conversation turn at a time). The fleet engine fans out its OWN worker
    threads underneath a single swarm turn — that is where the concurrency lives. The
    ``DecodeGate`` (created once in ``setup``) bounds concurrent generations == server
    KV; the AIMD controller resizes it toward the throughput knee from live /metrics.
    """

    def __init__(self, *, gate_start: Optional[int] = None,
                 admission: str = "aimd", warm: bool = True) -> None:
        self.events: "queue.Queue[dict]" = queue.Queue()
        self.gate: Optional[compat.DecodeGate] = None
        self.history: list[tuple[str, str]] = []   # (role, text)
        self.busy = False
        self._busy_lock = threading.Lock()   # R1: non-blocking guard around the busy flag

        self._controller = None
        self._setup_done = False
        self._setup_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._gate_start = int(gate_start or config.DECODE_GATE_START)
        self._admission = admission
        self._warm = warm
        self.tasks = TaskStore()
        self._wake = threading.Event()
        # The completion manager is CREATED here but only STARTED by start_manager()
        # (the TUI calls it). Merely constructing a SwarmRunner — e.g. in a test — must
        # NOT spin up a thread that could dispatch persisted goals against a live server.
        from .manager import CompletionManager
        self._manager = CompletionManager(self)

    # ── events ────────────────────────────────────────────────────────────────
    def emit(self, kind: str, **kw) -> None:
        self.events.put({"kind": kind, **kw})

    # ── one-time runtime setup (lazy: paid on the first turn) ───────────────────
    def setup(self) -> None:
        with self._setup_lock:
            if self._setup_done:
                return
            self.emit("boot", text="warming swarm runtime…")
            self.gate = compat.DecodeGate(self._gate_start)
            # Install the gate-aware forwarder + decode timers ONCE, process-global.
            compat.apply(self.gate)
            # Kill the cold-start cache stampede before any fan-out.
            try:
                compat.prewarm(list(config.TOOL_PROFILES.values()))
            except Exception:
                pass
            if self._admission == "aimd":
                from fleet.admission import AIMDController
                self._controller = AIMDController(self.gate, config.METRICS_URL, config)
                self._controller.start()
            if self._warm:
                try:
                    from fleet import warm as warm_mod
                    warm_mod.warm_profiles(_WARM_LANES, base_url=config.BASE_URL,
                                           model=config.MODEL, api_key=config.API_KEY)
                except Exception:
                    pass
            self._setup_done = True
            # R2 preflight: a fast server-health probe so a down inference server is
            # reported up-front (clean error bubble) instead of only failing mid-turn.
            if metrics.scrape(config.METRICS_URL, timeout=0.2) is None:
                self.emit("error", text=(
                    "inference server unreachable — no response from "
                    f"{config.METRICS_URL}. Start the Step-3.7 endpoint, then retry."))
            self.emit("ready", gate=self.gate.get_limit())

    def start_manager(self) -> None:
        """Start the completion manager (TUI session only); idempotent.

        Sets the wake event once so any goals PERSISTED from a previous session resume
        promptly on launch instead of waiting out the first ``interval_s`` heartbeat.
        """
        if self._manager is not None:
            self._manager.start()
            self._wake.set()

    def shutdown(self) -> None:
        if self._manager is not None:
            try:
                self._manager.stop()
            except Exception:
                pass
        if self._controller is not None:
            try:
                self._controller.stop()
            except Exception:
                pass
            self._controller = None

    # ── public API ──────────────────────────────────────────────────────────────
    def enqueue_task(self, goal: str) -> dict:
        """Add a goal to the persistent queue and wake the completion manager."""
        rec = self.tasks.add(goal)
        self.emit("queued", goal=goal, queue=self.tasks.counts())
        self._wake.set()
        return rec

    def submit(self, message: str, force_mode: Optional[str] = None,
               record=None) -> Optional[threading.Thread]:
        """Run one user turn on a background daemon thread; returns it (None if busy).

        R1 (TOCTOU): claim the busy flag SYNCHRONOUSLY here under a non-blocking lock
        BEFORE spawning the thread. If already busy, return None (caller treats that as
        'rejected, still working') instead of racing a second turn-thread past the old
        check-then-act. ``force_mode`` ∈ {None, "chat", "swarm"} skips the router.
        """
        with self._busy_lock:
            if self.busy:
                return None
            self.busy = True
        t = threading.Thread(target=self._run_turn, args=(message, force_mode, record),
                             name="swarm-turn", daemon=True)
        t.start()
        return t

    # ── turn execution ───────────────────────────────────────────────────────────
    def _run_turn(self, message: str, force_mode: Optional[str], record=None) -> None:
        # R1: busy already claimed in submit(); here we only clear it (finally).
        with self._run_lock:
            if record is not None:
                self.tasks.mark_running(record["id"])
            self.emit("user", text=message)
            self._append_history("user", message)
            try:
                if not self._setup_done:
                    self.setup()
                if record is not None:
                    ok, out = self._run_swarm(message)
                    if ok:
                        self.tasks.complete(record["id"], out)
                    else:
                        self.tasks.fail(record["id"], out)
                elif force_mode == "swarm":
                    self._run_swarm(message)
                elif force_mode == "chat":
                    self._record_reply(self._chat(message))
                else:
                    mode, reply = self._route(message)
                    self.emit("route", mode=("swarm" if mode.startswith("swarm") else mode))
                    if mode == "swarm":
                        self._run_swarm(message)
                    elif mode == "swarm-fallback":
                        self.emit("status", text="router unsure — routing to the swarm")
                        self._run_swarm(message)
                    else:
                        self._record_reply(reply)
            except self._ServerDown as e:                # R2: typed server-down error
                if record is not None:
                    # An unreachable server is an INFRA failure, not a task failure —
                    # do NOT burn the attempt budget (which would permanently fail the
                    # goal after a few transient outages, defeating "always complete").
                    # Requeue it; the manager only re-dispatches once the server is back
                    # (it probes before dispatch), so this never tight-loops.
                    self.tasks.requeue(record["id"])
                self.emit("error", text=(
                    "inference server unreachable / API failed — "
                    f"{e}. Check the Step-3.7 endpoint and retry."))
            except Exception as e:                       # never crash the front door
                if record is not None:
                    self.tasks.fail(record["id"], str(e))
                self.emit("error", text=f"{type(e).__name__}: {e}",
                          detail=traceback.format_exc())
            finally:
                with self._busy_lock:                    # R1: clear under the same guard
                    self.busy = False
                self.emit("idle")
                self._wake.set()

    def _record_reply(self, text: str) -> None:
        text = (text or "").strip() or "…"
        self.emit("reply", text=text)
        self._append_history("assistant", text)

    def _append_history(self, role: str, text: str) -> None:
        """R3: append a (role, text) turn and trim to a bounded tail."""
        self.history.append((role, text))
        if len(self.history) > _HISTORY_CAP:
            del self.history[:-_HISTORY_CAP]

    class _ServerDown(Exception):
        """Raised when run_conversation reports the inference server unreachable / API failed."""

    def _run_agent(self, lane: str, prompt: str, task_id: str, *,
                   max_iterations: int = 1, max_tokens: int = 1024) -> str:
        """Run one conversation and return its final text, OR raise _ServerDown.

        R2: run_conversation RETURNS completed=False with error/failed set when the
        inference server is unreachable / the API fails after retries — it does not raise.
        Echoing final_response in that case reads as 'the model answered with an error'
        (wrong), so detect it and surface a clean server-down error instead.
        """
        agent = compat.make_agent(lane, task_id=_short_id(task_id),
                                  max_iterations=max_iterations, max_tokens=max_tokens)
        result = agent.run_conversation(prompt, task_id=lane)
        # NB: `completed` is False even for a NORMAL bounded single-turn call (the agent
        # loop is cut at max_iterations before it self-declares done), so it is NOT a
        # failure signal — only `failed`/`error` flag a real API/server failure (a
        # server-down return sets failed=True + error). Gating on `completed` here
        # false-positived every healthy router/chat/plan call as 'server unreachable'.
        if result.get("failed") or result.get("error"):
            raise self._ServerDown(
                str(result.get("error") or "inference server unreachable / API failed"))
        return _final_text(result.get("messages"), result.get("final_response"))

    # ── router: one cheap call returns (mode, reply-if-chat) ─────────────────────
    def _route(self, message: str) -> tuple[str, str]:
        prompt = ROUTER_PROMPT.format(history=self._history_snippet(), message=message)
        text = self._run_agent("router", prompt, "route", max_iterations=1, max_tokens=1024)
        data = _parse_route(text)
        if data and data.get("mode") == "swarm":
            return ("swarm", "")
        if data and data.get("mode") == "chat":
            return ("chat", str(data.get("reply") or "").strip())
        # R4: router returned non-dict/empty JSON — do NOT echo raw text. Route to the
        # swarm as the safe default (a real task gets decomposed; a stray greeting gets
        # a tiny plan), with a clean status, never raw JSON.
        return ("swarm-fallback", "")

    def _chat(self, message: str) -> str:
        prompt = CHAT_PROMPT.format(history=self._history_snippet(), message=message)
        return self._run_agent("router", prompt, "chat", max_iterations=1, max_tokens=2048)

    # ── swarm: plan → board → ThreadFleet → final deliverable ────────────────────
    def _run_swarm(self, goal_text: str) -> tuple[bool, str]:
        self.emit("planning")
        try:
            tasks = self._plan(goal_text)
        except ValueError as e:
            # G1: parse_plan / validate_tasks raises ValueError on a malformed plan.
            # Log the raw reason; show the user a friendly, actionable message instead
            # of planner jargon. (_ServerDown is NOT caught here — it propagates to
            # _run_turn's typed handler.)
            self.emit("error", text=(
                "couldn't turn that into a task plan — try rephrasing, narrowing it, "
                "or use /chat for a direct answer."), detail=f"plan parse failed: {e}")
            return (False, "plan parse failed")
        self.emit("planned", n=len(tasks), tasks=[
            {"id": t.id, "lane": t.lane, "deps": list(t.deps), "prompt": t.prompt[:160]}
            for t in tasks
        ])

        board = Board()
        board.add_many(tasks)

        def push(kind, tid, counts=None, **extra):
            self.events.put({"kind": "task", "event": kind, "id": tid,
                             "counts": counts or {}, **extra})

        out = ThreadFleet(board, self.gate, cfg=config, on_event=push).run()
        final = goal_mod.final_result(out.get("board_results", {}))
        counts = out.get("counts") or {}
        stats = {
            "wall_s": out.get("wall_s"),
            "counts": counts,
            "mean_running": out.get("mean_running"),
            "peak_running": out.get("peak_running"),
            "unfinished": out.get("unfinished"),
        }
        if final and final.strip():
            self.emit("final", text=final, stats=stats)
            self._append_history("assistant", final)
            return (True, final)
        else:
            # Reducer failed / produced nothing — surface a clean error, never an
            # empty assistant bubble (which read as "no response came back").
            self.emit("error", text=(
                "swarm produced no final deliverable "
                f"(done={counts.get('done')}, failed={counts.get('failed')}, "
                f"unfinished={out.get('unfinished')})"), stats=stats)
            return (False, "swarm produced no final deliverable")

    def _plan(self, goal_text: str):
        text = self._run_agent("planner",
                               goal_mod.PLANNER_PROMPT.format(goal=goal_text),
                               "plan", max_iterations=2, max_tokens=4096)
        return goal_mod.parse_plan(text)

    # ── helpers ──────────────────────────────────────────────────────────────────
    def _history_snippet(self) -> str:
        turns = self.history[-_HISTORY_TURNS:]
        if not turns:
            return "(none)"
        return "\n".join(f"{role}: {text}" for role, text in turns)
