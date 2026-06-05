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
import os
import queue
import threading
import time
import traceback
import uuid
from typing import Optional

from fleet import compat, config, metrics, prompts as _prompts
from fleet.board import Board
from fleet.engine import ThreadFleet
from fleet.worker import _final_text
from . import goal as goal_mod
from . import recall as _recall
from .skills import synth as _skills_synth
from .taskstore import TaskStore


ROUTER_PROMPT = """You are the front door of a high-concurrency software agent swarm.
Decide whether the user's latest message is:
  - "chat": light conversation, a greeting, a clarification, or a question that ONE
    direct reply handles well; or
  - "swarm": a real task worth decomposing into many parallel sub-tasks across files,
    modules, areas, or sources (audit a repo, implement a multi-file feature, research
    several topics, refactor across a package, "do X for every Y").

Lean toward "chat" unless the work is genuinely multi-step or spans many units.

""" + _prompts.LANGUAGE_DIRECTIVE + """

Reply with ONLY a JSON object, nothing else (the JSON keys stay in English; only the
reply text follows the language rule above):
  {{"mode":"chat","reply":"<your full reply to the user>"}}
or
  {{"mode":"swarm"}}

{recall}Recent conversation (oldest first):
{history}

User message:
{message}
"""

CHAT_PROMPT = """You are swarm-agent, chatting with Hikari. Be warm, friendly, and
genuinely helpful — a natural, personable tone, concise but never curt. Speak openly and
never refuse to talk about how you work. Plain text, no preamble.

""" + _prompts.LANGUAGE_DIRECTIVE + """

{recall}Recent conversation (oldest first):
{history}

User message:
{message}
"""

BTW_PROMPT = """You are swarm-agent answering a quick side question ("by the way") about
your CURRENT situation — asked while you may be in the middle of other work. You and the
parallel workers are one swarm mind, so report the whole swarm's state honestly, warmly,
and briefly.

""" + _prompts.LANGUAGE_DIRECTIVE + """

Current situation:
{situation}

The user's side question:
{question}

Answer directly and concisely, grounded in the situation above. If the situation shows
nothing running, just say so."""

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
        from .logbook import SwarmLogger
        self.log = SwarmLogger()
        self.log.event("session_start",
                       max_goals=int(config.MAX_CONCURRENT_GOALS),
                       model=config.MODEL, base_url=config.BASE_URL,
                       gate_start=int(gate_start or config.DECODE_GATE_START),
                       admission=admission,
                       sandbox_isolate=getattr(compat, "_SANDBOX_ISOLATE", None),
                       auto_approve=os.environ.get("FLEET_AUTO_APPROVE", "1"))
        self.gate: Optional[compat.DecodeGate] = None
        self.history: list[tuple[str, str]] = []   # (role, text)
        # Persistent conversation recall (LanceDB hybrid): the in-process history is
        # trimmed to _HISTORY_CAP, but every turn is also indexed here so the front door /
        # planner can REFERENCE older turns on demand instead of forgetting them.
        self._recall = _recall.get_store()
        self._turn_idx = 0                          # monotonic id source for the recall index
        self._active: dict[str, object] = {}   # in-flight turns: rec_id -> Thread; "_interactive" sentinel for a typed turn
        self._busy_lock = threading.RLock()
        self._history_lock = threading.Lock()  # guard shared history across concurrent goal turns
        self._max_goals = int(config.MAX_CONCURRENT_GOALS)
        from .scheduler import GoalScheduler
        self._goals = GoalScheduler(self._max_goals)

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
    def _publish(self, ev: dict) -> None:
        """Single publish path: persist the event to the logbook, then enqueue it for the
        UI. Every emit() and every fleet task event goes through here so the JSONL log is a
        complete record (PARALLEL_GOALS / debugging)."""
        self.log.log(ev)
        self.events.put(ev)

    def emit(self, kind: str, **kw) -> None:
        self._publish({"kind": kind, **kw})

    @property
    def busy(self) -> bool:
        """True iff ANY turn (interactive or queued goal) is in flight. Derived from the
        in-flight registry so the UI badge + reentry guard keep working unchanged."""
        return len(self._active) > 0

    @busy.setter
    def busy(self, value: bool) -> None:
        # Compat shim for callers/tests that set busy directly: True injects the interactive
        # sentinel, False clears it. Internal code manipulates self._active directly under
        # _busy_lock and does NOT go through this setter.
        with self._busy_lock:
            if value:
                self._active.setdefault("_interactive", None)
            else:
                self._active.pop("_interactive", None)

    # ── one-time runtime setup (lazy: paid on the first turn) ───────────────────
    def setup(self) -> None:
        with self._setup_lock:
            if self._setup_done:
                return
            self.emit("boot", text="warming swarm runtime…")
            self.gate = compat.DecodeGate(self._gate_start)
            # Warm the recall store (load the CPU embedder + open LanceDB + backfill from
            # prior session logs) off the hot path so the first turn isn't blocked.
            self._recall.warm_async()
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
        try:
            self.log.event("session_end")
            self.log.close()
        except Exception:
            pass

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

        R1 (TOCTOU): claim the interactive slot SYNCHRONOUSLY here under a non-blocking
        lock BEFORE spawning the thread. If already occupied, return None (caller treats
        that as 'rejected, still working') instead of racing a second turn-thread past the
        old check-then-act. ``force_mode`` ∈ {None, "chat", "swarm"} skips the router.
        """
        with self._busy_lock:
            if "_interactive" in self._active:
                return None                          # one typed turn at a time, always
            # Respect the shared K cap too: an interactive turn occupies one of the K slots
            # (everything shares one policy — §4.5), so at K=1 it must NOT start alongside a
            # running queued goal. ATOMIC under _busy_lock, restoring the old "return None
            # while at capacity" contract (the TUI also gates on busy, this is the backstop).
            if len(self._active) >= self._max_goals:
                return None
            self._active["_interactive"] = None
        t = threading.Thread(target=self._run_turn, args=(message, force_mode, record),
                             name="swarm-turn", daemon=True)
        with self._busy_lock:
            self._active["_interactive"] = t
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
                    # Distinguish a genuinely UNREACHABLE server (infra → requeue WITHOUT
                    # burning the attempt budget; the manager retries once it is back) from
                    # a request-level API error that merely surfaces as _ServerDown while
                    # the server is UP. The latter must burn the budget — otherwise a
                    # persistently-failing goal requeues forever with attempts stuck at 0
                    # (an infinite re-dispatch loop). Probe to tell them apart.
                    if metrics.scrape(config.METRICS_URL, timeout=0.3) is None:
                        self.tasks.requeue(record["id"])
                    else:
                        self.tasks.fail(record["id"], str(e))
                self.emit("error", text=(
                    "inference server unreachable / API failed — "
                    f"{e}. Check the Step-3.7 endpoint and retry."))
            except Exception as e:                       # never crash the front door
                if record is not None:
                    self.tasks.fail(record["id"], str(e))
                self.emit("error", text=f"{type(e).__name__}: {e}",
                          detail=traceback.format_exc())
            finally:
                with self._busy_lock:
                    self._active.pop("_interactive", None)
                    idle = not self._active
                if idle:
                    compat.reset_interject()     # no steer leaks into the next turn
                self.emit("idle")
                self._wake.set()

    def active_goal_ids(self) -> set[str]:
        """Ids of in-flight QUEUED goal turns (excludes the interactive sentinel)."""
        with self._busy_lock:
            return {k for k in self._active if k != "_interactive"}

    def can_admit_goal(self) -> bool:
        """True iff the manager may dispatch another queued goal: under the K cap AND no
        writer is executing or waiting (writers drain everything first — §4.4).

        The cap counts EVERY in-flight turn — queued goals AND an interactive turn (the
        ``_interactive`` sentinel). Everything shares one K-slot policy (§4.5), so at K=1
        a typed turn occupies the single slot and no queued goal starts alongside it:
        byte-for-byte the old "dispatch only when not busy" behaviour (DoD §9). At K>1 an
        interactive turn is simply one of the K concurrent slots."""
        with self._busy_lock:
            n = len(self._active)
        return (n < self._max_goals
                and not self._goals.writer_active
                and not self._goals.writer_pending)

    def submit_goal(self, rec) -> Optional[threading.Thread]:
        """Spawn a daemon thread running one QUEUED goal turn. The rec is already in
        'running' state (claim_next did the atomic transition). Returns the thread, or None
        if this goal id is already in flight OR admitting it would exceed the K cap (the
        manager then re-queues it). Concurrency-safe; does NOT take _run_lock.

        The cap is enforced ATOMICALLY here under _busy_lock — not just via the manager's
        pre-check — so a turn that starts during a slow setup()/warmup window (between
        claim_next and here) cannot push the active set past FLEET_MAX_CONCURRENT_GOALS."""
        rid = rec["id"]
        with self._busy_lock:
            if rid in self._active:
                return None
            if len(self._active) >= self._max_goals:
                return None
            self._active[rid] = None
        t = threading.Thread(target=self._run_goal_turn, args=(rec,),
                             name=f"swarm-goal-{rid}", daemon=True)
        with self._busy_lock:
            self._active[rid] = t
        t.start()
        return t

    def _run_goal_turn(self, rec) -> None:
        rid = rec["id"]
        self.emit("user", text=rec["goal"], goal_id=rid)
        self._append_history("user", rec["goal"])
        try:
            if not self._setup_done:
                self.setup()
            ok, out = self._run_swarm(rec["goal"], goal_id=rid)
            if ok:
                self.tasks.complete(rid, out)
            else:
                self.tasks.fail(rid, out)
        except self._ServerDown as e:
            # Same infra-vs-API distinction as _run_turn: unreachable server -> requeue
            # without burning budget; API error while server is up -> fail (burn budget).
            if metrics.scrape(config.METRICS_URL, timeout=0.3) is None:
                self.tasks.requeue(rid)
            else:
                self.tasks.fail(rid, str(e))
            self.emit("error", text=(
                "inference server unreachable / API failed — "
                f"{e}. Check the Step-3.7 endpoint and retry."), goal_id=rid)
        except Exception as e:
            self.tasks.fail(rid, str(e))
            self.emit("error", text=f"{type(e).__name__}: {e}",
                      detail=traceback.format_exc(), goal_id=rid)
        finally:
            with self._busy_lock:
                self._active.pop(rid, None)
                idle = not self._active
            if idle:
                compat.reset_interject()         # no steer leaks into the next turn
            self.emit("idle", goal_id=rid)
            self._wake.set()

    def _record_reply(self, text: str) -> None:
        text = (text or "").strip() or "…"
        self.emit("reply", text=text)
        self._append_history("assistant", text)

    def _append_history(self, role: str, text: str) -> None:
        """R3: append a (role, text) turn and trim to a bounded tail. Also index the turn
        into the persistent recall store (fire-and-forget) so the trimmed tail is never
        truly forgotten — older turns stay retrievable via hybrid search."""
        with self._history_lock:
            self.history.append((role, text))
            if len(self.history) > _HISTORY_CAP:
                del self.history[:-_HISTORY_CAP]
            idx = self._turn_idx
            self._turn_idx += 1
        self._recall.add_async(role, text, getattr(self.log, "sid", "default"), idx)

    def _recall_prefix(self, query: str) -> str:
        """A prompt-ready block of relevant EARLIER turns (excluding the recent snippet
        already shown), or "". Fail-soft: any error → "" (never breaks the front door)."""
        try:
            recent = [t for _, t in self.history[-_HISTORY_TURNS:]]
            blk = self._recall.block(query, exclude_texts=recent)
            return (blk + "\n\n") if blk else ""
        except Exception:
            return ""

    class _ServerDown(Exception):
        """Raised when run_conversation reports the inference server unreachable / API failed."""

    def ask_status(self, question: str, situation: str) -> None:
        """Answer a side ("btw") question about the CURRENT situation via an INDEPENDENT
        worker. Runs on its OWN daemon thread and does NOT take ``_run_lock`` or set
        ``busy`` — so it answers even while a swarm turn is mid-flight. Emits a ``btw``
        event with the answer (or a clean note on failure); never raises to the caller."""
        def _run():
            try:
                prompt = BTW_PROMPT.format(
                    situation=(situation or "(idle — nothing is running right now)"),
                    question=question)
                ans = self._run_agent("router", prompt, "btw",
                                      max_iterations=1, max_tokens=1024)
                self.emit("btw", text=ans, question=question)
            except self._ServerDown as e:
                self.emit("btw", text=f"(couldn't reach the model: {e})",
                          question=question)
            except Exception as e:
                self.emit("btw", text=f"(btw failed: {type(e).__name__}: {e})",
                          question=question)
        threading.Thread(target=_run, name="swarm-btw", daemon=True).start()

    # ── mid-flight interject (deliver a typed message INTO a running turn) ────────
    def steer(self, text: str) -> int:
        """Inject a user message into the turn(s) currently in flight WITHOUT stopping
        them (HermesAgent /steer). Fans out to every live agent — planner, fleet workers,
        reducer, persona — and is stashed so agents that start later this turn (e.g. the
        reducer) also see it. Returns how many live agents it reached right now."""
        n = compat.steer_all(text)
        self.emit("steer", text=text, reached=n)
        return n

    def interrupt(self, message: Optional[str] = None) -> int:
        """Hard-interrupt every live agent's tool-calling loop (HermesAgent interrupt).
        Stops in-flight generation/tools so the turn unwinds quickly. Returns how many
        agents were signalled."""
        n = compat.interrupt_all(message)
        self.emit("interrupt", reached=n)
        return n

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
        prompt = ROUTER_PROMPT.format(history=self._history_snippet(), message=message,
                                      recall=self._recall_prefix(message))
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
        prompt = CHAT_PROMPT.format(history=self._history_snippet(), message=message,
                                    recall=self._recall_prefix(message))
        return self._run_agent("router", prompt, "chat", max_iterations=1, max_tokens=2048)

    def _await_abandoned_writers(self, goal_id) -> None:
        """Before a WRITING goal starts, wait (bounded) for any still-alive ABANDONED write
        worker from an earlier timed-out goal to finish — its subprocess can't be killed, so
        running now could race it and corrupt the workspace. Proceeds with a loud warning if it
        outlives the bound (never a permanent stall)."""
        from fleet import engine as _engine
        wait_s = float(getattr(config, "ABANDONED_WRITER_WAIT_S", 0) or 0)
        if wait_s <= 0 or not _engine.abandoned_writers_alive():
            return
        self.emit("status", text="waiting for an abandoned write worker to finish before "
                  "starting this writing goal", goal_id=goal_id)
        deadline = time.monotonic() + wait_s
        while _engine.abandoned_writers_alive():
            if time.monotonic() >= deadline:
                msg = ("proceeding with a writing goal while an abandoned write worker is "
                       "STILL alive (waited %.0fs) — possible workspace race" % wait_s)
                self.emit("error", text=msg, goal_id=goal_id)
                self.log.event("abandoned_writer_wait_timeout", goal_id=goal_id, waited_s=wait_s)
                return
            time.sleep(0.5)

    # ── swarm: plan → board → ThreadFleet → final deliverable ────────────────────
    def _run_swarm(self, goal_text: str, *, goal_id: Optional[str] = None) -> tuple[bool, str]:
        self.emit("planning", goal_id=goal_id)
        try:
            tasks = self._plan(goal_text)
        except ValueError as e:
            # G1: parse_plan / validate_tasks raises ValueError on a malformed plan.
            # Log the raw reason; show the user a friendly, actionable message instead
            # of planner jargon. (_ServerDown is NOT caught here — it propagates to
            # _run_turn's typed handler.)
            self.emit("error", text=(
                "couldn't turn that into a task plan — try rephrasing, narrowing it, "
                "or use /chat for a direct answer."), detail=f"plan parse failed: {e}",
                goal_id=goal_id)
            return (False, "plan parse failed")

        # Classify BEFORE acquiring an execution permit (planning ran un-permitted /
        # concurrent). read-only -> shared reader slot; writing -> exclusive (§4.2).
        kind = goal_mod.classify_plan(tasks)
        readonly = (kind == "read-only")

        # Namespace a QUEUED goal's task ids by its record id so concurrent goals never
        # share a task id (per-worker sandbox cwd collision) (§4.3). Interactive turns
        # (goal_id is None) keep their ids — at most one interactive turn runs at a time.
        if goal_id is not None:
            tasks = goal_mod.namespace_tasks(tasks, goal_id)

        self.emit("planned", n=len(tasks), goal_id=goal_id, tasks=[
            {"id": t.id, "lane": t.lane, "deps": list(t.deps), "prompt": t.prompt[:160]}
            for t in tasks
        ])

        board = Board()
        board.add_many(tasks)

        def push(kind, tid, counts=None, **extra):
            self._publish({"kind": "task", "event": kind, "id": tid,
                           "counts": counts or {}, "goal_id": goal_id, **extra})

        # Acquire the shared/exclusive permit for the fleet execution ONLY (not planning),
        # then run, then release (context manager). This is what bounds write goals to
        # exclusive and read-only goals to ≤K concurrent (§4.1/4.2).
        # Writing goals (exclusive) also get a PRIVATE sandbox root so their coder workers'
        # ephemeral cwd cannot clobber an interactive turn's or another goal's (§4.3).
        import contextlib as _contextlib
        import tempfile as _tempfile
        if readonly:
            sb_ctx = _contextlib.nullcontext()
        else:
            sb_ctx = compat.sandbox_root(
                _tempfile.mkdtemp(prefix=f"swarm-goal-{(goal_id or 'turn')}-"))
        with self._goals.permit(readonly=readonly), sb_ctx:
            if not readonly:
                self._await_abandoned_writers(goal_id)
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
            self.emit("final", text=final, stats=stats, goal_id=goal_id)
            self._append_history("assistant", final)
            # Auto-generate a skill from this outcome if it taught something reusable
            # (fire-and-forget; conservative; never blocks or breaks the turn).
            try:
                _skills_synth.synthesize_async(goal_text, final)
            except Exception:
                pass
            return (True, final)
        else:
            # Reducer failed / produced nothing — surface a clean error, never an
            # empty assistant bubble (which read as "no response came back").
            self.emit("error", text=(
                "swarm produced no final deliverable "
                f"(done={counts.get('done')}, failed={counts.get('failed')}, "
                f"unfinished={out.get('unfinished')})"), stats=stats, goal_id=goal_id)
            return (False, "swarm produced no final deliverable")

    def _plan(self, goal_text: str):
        """Plan the goal into a task DAG, RETRYING on a malformed plan.

        Planning runs at reasoning_effort='none' (fast), which occasionally emits invalid
        JSON or a mis-wired DAG (e.g. a leaf left unconnected, though validate_tasks now
        auto-wires those). A couple of cheap retries recover such a slip instead of failing
        the whole goal with "plan parse failed". ``_ServerDown`` is NOT caught here — it
        propagates to the typed handler. Raises the last ValueError if every attempt fails.
        """
        prompt = self._recall_prefix(goal_text) + goal_mod.PLANNER_PROMPT.format(goal=goal_text)
        last: Optional[ValueError] = None
        for _ in range(3):
            # Generous output cap: a whole task-DAG JSON must land in ONE generation. A small
            # cap truncated big plans → truncation→continuation→max-iteration churn ("planning
            # loops for minutes"). See config.PLANNER_MAX_TOKENS.
            text = self._run_agent("planner", prompt, "plan",
                                   max_iterations=config.PLANNER_MAX_ITERATIONS,
                                   max_tokens=config.PLANNER_MAX_TOKENS)
            try:
                return goal_mod.parse_plan(text)
            except ValueError as e:
                last = e
        raise last

    # ── helpers ──────────────────────────────────────────────────────────────────
    def _history_snippet(self) -> str:
        turns = self.history[-_HISTORY_TURNS:]
        if not turns:
            return "(none)"
        return "\n".join(f"{role}: {text}" for role, text in turns)
