"""Runtime adaptation of HermesAgent for single-process fleet use.

EVERYTHING here is applied at runtime from the harness — the hermes-agent repo is
NEVER modified on disk (git-pull-safe). See BUILD_SPEC.md §1 for the recon citations
behind each patch.

Three jobs:
  1. DecodeGate — a resizable, lane-priority semaphore that bounds *concurrent
     generations*. Because HermesAgent is stateless (full-history resend), the server
     holds KV only while generating, so pinning generations == pinning num_requests_running
     == pinning KV. This is the decode-batch admission of DESIGN §3.6.
  2. apply(gate) — wrap the two LLM forwarders (run_agent.py:3277 / :3448, the single
     chokepoint for streaming + non-streaming) to (a) acquire the gate and (b) time
     decode_s per AIAgent instance; plus thread-safety + env hygiene fixups.
  3. make_agent(lane, …) — build a fleet-safe AIAgent (unique session_id, lean toolset,
     no memory/context/trajectory persistence, bounded turns), with all the thread-safety
     mitigations in one place.
"""
from __future__ import annotations

import contextlib
import heapq
import itertools
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from typing import List, Optional

from . import config

# ───────────────────────────── DecodeGate ──────────────────────────────────────

class DecodeGate:
    """Resizable counting semaphore with lane-priority admission.

    Permits held == concurrent LLM generations. Workers executing tools (between turns,
    on the stateless completions path) hold NO permit and NO server KV, so `enrolled`
    can far exceed `limit`. Higher-priority lanes (director > planner > … > router) are
    served first when a permit frees, so reserved roles never starve behind the swarm.

    Liveness: a waiter re-checks every 0.5 s, so a limit increase or a missed wakeup
    self-heals; never deadlocks while limit >= 1. Callers MUST keep limit >= 1.

    Concurrency design (v0.2 review fixes #1, #2):
      * Admission is strictly priority-ordered — only the single highest-priority
        waiter (heap top) is ever eligible to take a freed permit. So broadcasting to
        ALL ~150 waiters on every release/acquire (the old ``notify_all``) was an O(N)
        GIL-serialised thundering herd: 149 of them wake, fail the predicate, sleep
        again. Instead, every waiter carries its OWN ``threading.Event`` and we wake
        ONLY the current heap top when a permit is actually free. The 0.5 s timed wait
        on each Event is the lost-wakeup safety net (covers set_limit races etc.).
      * acquire is interrupt-safe: the ticket is pushed under the lock and removed in a
        ``finally`` if we leave WITHOUT a permit (BaseException: KeyboardInterrupt /
        async exc / GeneratorExit). A leaked higher-priority ticket would pin the heap
        top forever and wedge the gate permanently — the bug this guards against.
    """

    def __init__(self, limit: int):
        self._lock = threading.Lock()
        self._limit = max(1, int(limit))
        self._in = 0                                  # permits in use == live generations
        # heap of (-priority, seq, event) tickets; each waiter sleeps on its own event
        # so a release can wake JUST the eligible top waiter (no thundering herd).
        self._waiters: list = []
        self._seq = itertools.count()
        # observability
        self.acquired_total = 0
        self.wait_s_total = 0.0
        self.peak_in = 0

    # -- internal: must hold self._lock --------------------------------------
    def _wake_top(self) -> None:
        """Wake exactly the highest-priority waiter IFF a permit is free.

        Only the heap top can satisfy the admission predicate, so signalling
        anyone else is wasted work. No-op when no permit is free (the waiter
        will be woken by the eventual release) or when there are no waiters.
        """
        if self._in < self._limit and self._waiters:
            self._waiters[0][2].set()

    def set_limit(self, n: int) -> None:
        with self._lock:
            self._limit = max(1, int(n))
            # A limit increase may newly admit the top waiter; wake it (only it can
            # proceed). A decrease frees no one. Either way this is O(1), not O(N).
            self._wake_top()

    def get_limit(self) -> int:
        with self._lock:
            return self._limit

    def stats(self) -> dict:
        with self._lock:
            return {"limit": self._limit, "in_flight": self._in,
                    "waiting": len(self._waiters), "peak_in": self.peak_in,
                    "acquired_total": self.acquired_total,
                    "wait_s_total": round(self.wait_s_total, 2)}

    @contextlib.contextmanager
    def acquire(self, lane: str = "worker"):
        ev = threading.Event()
        ticket = (-config.lane_priority(lane), next(self._seq), ev)
        t0 = time.perf_counter()
        got_permit = False
        with self._lock:
            heapq.heappush(self._waiters, ticket)
            # If WE are already the eligible top (permit free + heap top), short-circuit;
            # otherwise loop on our own event. Wrapped in try/finally so an interrupt
            # while waiting (BaseException, not just Exception) cannot leak the ticket.
            try:
                while True:
                    if self._in < self._limit and self._waiters and self._waiters[0] is ticket:
                        heapq.heappop(self._waiters)        # pop our ticket (it is the top)
                        self._in += 1
                        got_permit = True
                        self.acquired_total += 1
                        self.peak_in = max(self.peak_in, self._in)
                        self.wait_s_total += time.perf_counter() - t0
                        # If permits AND waiters still remain (e.g. a set_limit bumped
                        # the limit by >1, freeing several slots at once), chain-wake
                        # the NEW top so the admission cascades one-by-one instead of
                        # stalling until the next release. Still O(1) per handoff.
                        self._wake_top()
                        break
                    # Not eligible yet: sleep on OUR event with the 0.5 s liveness net.
                    # Release the lock around the wait so others can make progress.
                    ev.clear()
                    self._lock.release()
                    try:
                        ev.wait(timeout=0.5)
                    finally:
                        self._lock.acquire()
            finally:
                if not got_permit:
                    # FIX #1: left WITHOUT a permit (interrupt/exception while waiting).
                    # Remove our ticket so a higher-priority leak can't pin the heap top
                    # forever, then re-wake the NEW top in case we were blocking it.
                    try:
                        self._waiters.remove(ticket)
                        heapq.heapify(self._waiters)
                    except ValueError:
                        pass  # already popped (shouldn't happen on the no-permit path)
                    self._wake_top()
        try:
            yield self
        finally:
            with self._lock:
                self._in -= 1
                # One permit freed → wake exactly the eligible top waiter (FIX #2:
                # was notify_all == wake all ~150). Correctness: every distinct
                # release wakes a distinct top; the 0.5 s Event timeout backstops
                # any lost wakeup.
                self._wake_top()


@contextlib.contextmanager
def _null_gate(lane: str = "worker"):
    yield None


# ──────────────────────── runtime monkeypatch (apply) ──────────────────────────

_GATE: Optional[DecodeGate] = None
_apply_lock = threading.Lock()
_UNSET = object()  # sentinel: apply() without a gate arg must NOT clobber the installed gate


def _ensure_hermes_on_path() -> None:
    if config.HERMES_DIR not in sys.path:
        sys.path.insert(0, config.HERMES_DIR)


def _env_hygiene() -> None:
    """TS5: keep workers headless; stop kanban tools leaking into empty profiles."""
    # HERMES_KANBAN_TASK injects kanban tools into *every* profile incl. router/reducer []
    os.environ.pop("HERMES_KANBAN_TASK", None)
    # FIX #5: the ONLY env knob hermes actually honours for interactivity is
    # HERMES_INTERACTIVE (tools/terminal_tool.py:809 `env_var_enabled("HERMES_INTERACTIVE")`;
    # acp approval-isolation tests confirm: unset => non-interactive auto-approve path).
    # Pop it so a worker thread (no TTY) can NEVER block on a sudo/approval prompt.
    os.environ.pop("HERMES_INTERACTIVE", None)
    # NOTE: HERMES_NONINTERACTIVE / HERMES_DISABLE_APPROVALS are NOT read anywhere in
    # hermes as a behaviour gate (verified) — they were placebos. Setting them is a
    # harmless no-op; popping HERMES_INTERACTIVE above is the real fix. Left documented
    # in case a future hermes version starts honouring them.


def _patch_last_resolved_tool_names() -> None:
    """TS2 — DOCUMENTED KNOWN LIMITATION, no longer patched (review fix #6).

    ``model_tools._last_resolved_tool_names`` is a process-global list that every
    ``get_tool_definitions()`` reassigns and ``execute_code`` reads as its sandbox
    allow-list. The v0.1 mitigation made that ONE attribute thread-local via a module
    ``__class__`` swap installing a custom ``__getattribute__``/``__setattr__`` — but
    that intercepts EVERY attribute access on the ``model_tools`` module process-wide
    (~3x slower per access), taxing all lanes to protect a value only the rarely-used
    concurrent `code` lane reads. Net-negative for the fleet.

    DECISION: remove the global tax. Known gap: if two `code`-lane agents resolve tool
    defs concurrently, ``execute_code`` may observe the other's toolset list (the
    ``sandbox_enabled`` fallback at model_tools.py:835). Acceptable: the fleet rarely
    runs many concurrent code-lane agents, and the bulk worker/router/reducer lanes
    never call ``execute_code``. Kept as a no-op so ``apply`` and any future re-enable
    point stay structurally intact."""
    return  # intentional no-op — see docstring


# Bounded-generation mode for the throughput sweep (env FLEET_NO_CONTINUE). Step-3.7 is a
# very verbose reasoner that does not stop under a max_tokens cap (it truncates at the cap,
# finish_reason='length'), and HermesAgent then auto-requests up to 3 *continuations*
# (conversation_loop.py:1664), each re-prefilling the full ~8K transcript — turning one
# intended generation into 4 with huge re-prefill gaps, which both stalls the run and ruins a
# clean decode-throughput measurement. When FLEET_NO_CONTINUE is set, the forwarder rewrites a
# 'length' finish_reason to 'stop' on the returned response so the loop treats the single
# max_tokens-capped generation as complete (no continuation). OFF by default (real fleets want
# continuation); ON only for the controlled bounded sweep.
_NO_CONTINUE = os.environ.get("FLEET_NO_CONTINUE", "0") not in ("0", "false", "False")

# Optional sampling override injected into every forwarder's api_kwargs. Step-3.7 is a
# reasoning model that emits a large <think> block by default; `reasoning_effort='none'`
# makes it answer DIRECTLY (verified live: finish_reason='stop', a real bounded answer),
# giving clean single-generation tasks for the throughput sweep instead of multi-thousand-
# token reasoning that never fits a cap. Unset by default (real fleets keep reasoning).
_REASONING_EFFORT = os.environ.get("FLEET_REASONING_EFFORT") or None

# Per-lane reasoning effort. Step-3.7 emits a multi-thousand-token <think> block by
# DEFAULT, which overflows each call's max_tokens and forces a truncation→continuation
# re-prefill loop + max-iteration summaries — the "research/planning seems to loop" symptom
# (measured: a single research goal took ~115 s, ~42 s of it just the planner churning).
# Bounding the reasoning effort per lane cut that SAME goal to ~11 s with NO quality loss:
#   * planner / router / manager → 'none'  — structured/short output (JSON DAG, route,
#     assessment); they do not need to ruminate, so answer directly in one generation.
#   * every other lane (workers + reducer) → 'low' — keep SOME reasoning for tool use and
#     synthesis quality, but bounded so a sub-task can't churn for 90 s.
# Precedence: FLEET_REASONING_<LANE> env (e.g. FLEET_REASONING_CODER=high) > the pinned
# 'none' lanes > global FLEET_REASONING_EFFORT (workers/reducer) > the 'low' default.
_DIRECT_LANES = {"planner": "none", "router": "none", "manager": "none"}


def _lane_reasoning(lane: str):
    env = os.environ.get(f"FLEET_REASONING_{lane.upper()}")
    if env:
        return env
    if lane in _DIRECT_LANES:
        return _DIRECT_LANES[lane]
    return _REASONING_EFFORT or "low"


def _neutralize_length(result) -> None:
    """Rewrite finish_reason 'length' -> 'stop' on a forwarder return (streaming mock
    SimpleNamespace or a non-streaming response/dict). Best-effort; never raises."""
    try:
        resp = result.get("response") if isinstance(result, dict) else result
        for ch in (getattr(resp, "choices", None) or []):
            if getattr(ch, "finish_reason", None) == "length":
                ch.finish_reason = "stop"
    except Exception:
        pass


def apply(gate=_UNSET) -> None:
    """Idempotent. Wrap the two LLM forwarders with gate-acquire + decode_s timing,
    apply thread-safety + env fixups.

    Gate semantics (order-independent, so prefix-warm can't clobber the real gate):
      apply(gate)  → install `gate` (the live run).
      apply(None)  → explicitly DISABLE gating (timing only).
      apply()      → ensure forwarders are patched but DO NOT touch the installed gate
                     (used by warm/prewarm so calling them after the engine installed a
                     gate does not silently turn gating off — the bug this guards against).

    KNOWN GAP (review fix #9 — DEFERRED, ungated path):
      The two forwarders below are the ONLY generation chokepoints we wrap, so every
      normal turn (streaming + non-streaming, all api_modes) is gated + timed. BUT the
      iteration-limit *summary* generation calls the OpenAI client DIRECTLY —
      `agent._ensure_primary_openai_client(...).chat.completions.create(**summary_kwargs)`
      (agent/chat_completion_helpers.py:1433 and :1476 retry) — bypassing these wrappers,
      so it is UNGATED and its decode time is NOT counted in `_fleet_decode_s`.
      Why deferred (not fixed): wrapping it safely means intercepting the per-call OpenAI
      client (built/closed per call, owner_tid-disciplined — TS6) or monkeypatching
      `_ensure_primary_openai_client`, both of which touch the client/socket lifecycle the
      recon explicitly warned us NOT to perturb under high concurrency. Risk > reward:
      this path fires ONLY when a worker hits `max_iterations` AND needs a summary — never
      on the no-tool single-turn sweep, and rarely otherwise. It is one short, bounded
      (`agent.max_tokens`-capped) generation. Net effect: at most a handful of brief
      ungated generations near max_iterations; acceptable. Revisit if many workers
      routinely saturate max_iterations.
    """
    global _GATE
    with _apply_lock:
        _ensure_hermes_on_path()
        if gate is not _UNSET:
            _GATE = gate
        _env_hygiene()
        _patch_last_resolved_tool_names()

        import run_agent
        A = run_agent.AIAgent
        if getattr(A, "_fleet_patched", False):
            return

        def _wrap(orig):
            def inner(self, *args, **kwargs):
                lane = getattr(self, "_fleet_lane", "worker")
                eff = _lane_reasoning(lane)
                if eff and args and isinstance(args[0], dict):
                    # inject into a COPY of api_kwargs (don't mutate the loop's dict, which
                    # it may reuse on retry); flows into both streaming and non-streaming.
                    # Per-lane: planner/router/manager → 'none' (no truncation→continuation
                    # churn); other lanes use the global FLEET_REASONING_EFFORT default.
                    args = (dict(args[0], reasoning_effort=eff),) + args[1:]
                ctx = _GATE.acquire(lane) if _GATE is not None else _null_gate(lane)
                gw0 = time.perf_counter()
                with ctx:
                    self._fleet_gatewait_s = getattr(self, "_fleet_gatewait_s", 0.0) + \
                        (time.perf_counter() - gw0)
                    t0 = time.perf_counter()
                    try:
                        result = orig(self, *args, **kwargs)
                        if _NO_CONTINUE:
                            _neutralize_length(result)  # bounded-generation sweep: no continuation
                        return result
                    finally:
                        self._fleet_decode_s = getattr(self, "_fleet_decode_s", 0.0) + \
                            (time.perf_counter() - t0)
            inner.__name__ = getattr(orig, "__name__", "wrapped")
            inner.__wrapped__ = orig
            return inner

        A._interruptible_api_call = _wrap(A._interruptible_api_call)
        A._interruptible_streaming_api_call = _wrap(A._interruptible_streaming_api_call)
        A._fleet_patched = True


def prewarm(profiles: List[List[str]]) -> None:
    """TS3: resolve tool defs for every distinct role profile + warm the OpenAI class
    import, ONCE single-threaded, to avoid a 120-thread cold-start cache stampede."""
    _ensure_hermes_on_path()
    try:
        import run_agent
        _ = run_agent.OpenAI  # trigger the lazy `from openai import OpenAI`
    except Exception:
        pass
    try:
        import model_tools
        seen = set()
        for prof in profiles:
            key = tuple(sorted(prof or []))
            if key in seen:
                continue
            seen.add(key)
            try:
                model_tools.get_tool_definitions(enabled_toolsets=list(prof or []), quiet_mode=True)
            except Exception:
                pass
    except Exception:
        pass


# ──────────────────────────── make_agent ───────────────────────────────────────

_session_seq = itertools.count()


def new_session_id(task_id: str = "") -> str:
    """Globally-unique session id (TS1: avoid the ~0.07% auto-gen collision that would
    share a sandbox / cwd file / process_registry namespace across workers)."""
    return f"fleet-{task_id}-{next(_session_seq)}-{uuid.uuid4().hex[:8]}"


# ─────────────────────── per-worker sandbox isolation (FIX #3) ──────────────────

# FIX #3 recon (verified in venv, tools/terminal_tool.py:944-996, :1783-1801, :1917):
#   Terminal/file tools key their long-lived sandbox (cwd, bash, LocalEnvironment) off
#   the *tool-call* task_id via `_resolve_container_task_id`, which COLLAPSES every
#   ordinary task_id to the literal "default". So a unique AIAgent session_id does NOT
#   isolate sandboxes — concurrent tool-using workers would share one cwd/bash and
#   cross-contaminate (one worker's `cd`/files visible to another).
#   THE ESCAPE HATCH: if a task_id has an entry registered via
#   `register_task_env_overrides(task_id, {...})`, `_resolve_container_task_id` returns
#   that task_id UNCHANGED (its own `_active_environments[task_id]` entry) and honours
#   `overrides["cwd"]`. Registering a unique per-worker `cwd` therefore gives each
#   worker an isolated working directory + its own bash/LocalEnvironment.
#
# Env-overridable: FLEET_SANDBOX_ISOLATE=0 disables (then it's a documented no-op so the
# no-tool throughput sweep is unaffected; tool-using fleets want it ON).
_SANDBOX_ISOLATE = os.environ.get("FLEET_SANDBOX_ISOLATE", "1") not in ("0", "false", "False")
_SANDBOX_ROOT = os.environ.get("FLEET_SANDBOX_ROOT") or None  # None => system tempdir


@contextlib.contextmanager
def sandbox_root(path):
    """Temporarily point per-worker sandboxes at ``path`` (a private build dir). A WRITING
    goal uses this so its coder workers' ephemeral cwd lives under a private root, isolated
    from interactive turns / other goals' artifacts (PARALLEL_GOALS_PLAN §4.3). Safe because
    writing goals run EXCLUSIVELY (no other fleet runs concurrently), so mutating this
    module-global for the writer's fleet cannot race another fleet. No-op when path is falsy.
    NOTE: only RELATIVE cwd is isolated — absolute-path writes the planner emits still land
    where told; exclusivity is what actually prevents cross-goal write races."""
    global _SANDBOX_ROOT
    if not path:
        yield
        return
    prev = _SANDBOX_ROOT
    _SANDBOX_ROOT = path
    try:
        yield
    finally:
        _SANDBOX_ROOT = prev


def _hermes_sandbox_api():
    """Return (register, clear) from hermes terminal_tool, or (None, None) if the
    symbols are absent (older/newer hermes). DEFENSIVE: never raise."""
    try:
        _ensure_hermes_on_path()
        from tools.terminal_tool import (
            register_task_env_overrides as _reg,
            clear_task_env_overrides as _clr,
        )
        return _reg, _clr
    except Exception:
        return None, None


def _noninteractive_approval(command, description="", *, allow_permanent=True):
    """Return a dangerous-command approval decision without reading worker stdin.

    Worker threads have no TTY, so the only non-blocking choices are auto-approve or
    auto-deny. DEFAULT = auto-approve ("once"): a DELIBERATE autonomy choice so the
    completion manager can run goals to done unattended (the project's "always finish
    the task" requirement). The tradeoff is that dangerous commands (rm -rf, sudo,
    destructive/SSH) from a mistaken or prompt-injected worker also run unconfirmed.
    Set FLEET_AUTO_APPROVE=0 (or "deny") to flip the default to deny — benign file
    reads/commands still work; only guard-flagged dangerous commands are blocked.
    Read at call time so the env can be toggled without reimporting."""
    enabled = os.environ.get("FLEET_AUTO_APPROVE", "1") not in (
        "0", "false", "False", "deny")
    return "once" if enabled else "deny"


def install_noninteractive_approval() -> None:
    """Register the non-interactive approval callback in this worker thread."""
    try:
        _ensure_hermes_on_path()
        from tools.terminal_tool import set_approval_callback
        set_approval_callback(_noninteractive_approval)
    except Exception:
        pass


@contextlib.contextmanager
def worker_sandbox(task_id: str):
    """Give the worker whose tool calls run under ``task_id`` its OWN isolated cwd/bash.

    Registers a per-worker env override (unique workdir under a tempdir) keyed by
    ``task_id`` BEFORE the conversation runs, and tears it down (override entry +
    workdir + the live LocalEnvironment) after. Keyed by the *tool-call* task_id —
    which the worker passes as ``run_conversation(prompt, task_id=spec["id"])`` and
    which flows to the terminal/file tools as their `task_id` (verified
    agent/conversation_loop.py:442 `effective_task_id = task_id or uuid4`).

    Safe-by-default: if isolation is disabled (FLEET_SANDBOX_ISOLATE=0) or the hermes
    sandbox API is unavailable, this is a NON-FATAL no-op (the lead's no-tool sweep is
    unaffected; tool-using workers fall back to the shared "default" sandbox as before).
    """
    if not _SANDBOX_ISOLATE or not task_id:
        yield None
        return
    register, clear = _hermes_sandbox_api()
    if register is None:
        # Documented no-op fallback: hermes lacks the override API on this tree.
        yield None
        return

    workdir = None
    try:
        workdir = tempfile.mkdtemp(prefix=f"fleet-{task_id}-", dir=_SANDBOX_ROOT)
        # Registering ANY override entry for task_id is what makes
        # _resolve_container_task_id stop collapsing it to "default"; the cwd key
        # then points the worker's bash/LocalEnvironment at its private dir.
        register(task_id, {"cwd": workdir})
    except Exception:
        # Could not isolate (tempdir or API failure) — fall back to shared sandbox,
        # never break the run. Clean up a half-made workdir.
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)
        yield None
        return

    try:
        yield workdir
    finally:
        # Tear down: kill the live sandbox for this task_id (frees its bash/FDs), drop
        # the override entry, then remove the private workdir. All best-effort.
        # cleanup_vm(task_id, force_remove=True) pops _active_environments[task_id],
        # clears the creation lock + file-ops cache, and calls env.cleanup()
        # (verified tools/terminal_tool.py:1429). force_remove=True == user-initiated
        # teardown (our per-worker sandbox is ephemeral, unlike a long-lived session).
        try:
            _ensure_hermes_on_path()
            from tools.terminal_tool import cleanup_vm  # type: ignore
            cleanup_vm(task_id, force_remove=True)
        except Exception:
            pass
        try:
            clear(task_id)
        except Exception:
            pass
        shutil.rmtree(workdir, ignore_errors=True)


def make_agent(lane: str, *, base_url: str = None, api_key: str = None, model: str = None,
               max_iterations: int = None, session_id: str = None, task_id: str = "",
               max_tokens=_UNSET):
    """Construct a fleet-safe AIAgent for `lane` with every thread-safety mitigation:
    unique session_id, lean per-role toolset, no memory/context/trajectory persistence,
    bounded turns, quiet logging. Stamps `_fleet_lane` (read by the gate) and binds the
    session ContextVar in the *calling thread*.

    ``max_tokens`` (FIX #7): caps each generation for the bounded-generation sweep.
    Default ``_UNSET`` => read ``getattr(config, "MAX_TOKENS", None)`` (env FLEET_MAX_TOKENS);
    ``None`` => model/server default (AIAgent omits the param). Pass an int to override.
    """
    _ensure_hermes_on_path()
    from run_agent import AIAgent

    sid = session_id or new_session_id(task_id)
    # TS1: bind the session ContextVar in THIS worker thread so tools that read
    # get_session_env see the right id (ContextVars don't propagate to bare threads).
    try:
        from gateway.session_context import set_current_session_id
        set_current_session_id(sid)
    except Exception:
        pass
    install_noninteractive_approval()

    # FIX #7: resolve max_tokens defensively (config.MAX_TOKENS is authored by the
    # config agent; tolerate its absence on older trees → None = model default).
    mt = getattr(config, "MAX_TOKENS", None) if max_tokens is _UNSET else max_tokens

    # PERSONA front door: router/planner/reducer speak AS the user's HermesAgent —
    # give them SOUL.md identity + persistent MEMORY/USER memory; the worker swarm
    # stays lean. Resolved defensively so an older config.py (no is_persona_lane)
    # falls back to the original all-lean behaviour.
    try:
        _persona = config.is_persona_lane(lane)
    except Exception:
        _persona = False

    try:
        from . import prompts as _prompts
        _eph = _prompts.lane_system_prompt(lane)
    except Exception:
        _eph = None

    agent = AIAgent(
        base_url=base_url or config.BASE_URL,
        api_key=api_key or config.API_KEY,
        model=model or config.MODEL,
        enabled_toolsets=config.toolsets_for(lane),     # role-minimal tools
        # PERSONA lanes: inject SOUL.md (load_soul_identity=True) while KEEPING
        # skip_context_files=True, so the persona loads but the harness repo's cwd
        # AGENTS.md/.cursorrules never enter the prefix (system_prompt.py:90 loads
        # SOUL via the load_soul_identity branch; :263 gates cwd files on
        # not skip_context_files). WORKERS: both off → byte-identical lean prefix.
        skip_context_files=True,                        # NEVER inject cwd AGENTS.md/.cursorrules
        load_soul_identity=_persona,                    # True for persona → SOUL.md identity
        skip_memory=not _persona,                       # False for persona → MEMORY/USER injected
        save_trajectories=False,                        # no per-turn trajectory disk writes
        quiet_mode=True,                                # cut logging contention under 120 threads
        max_iterations=config.MAX_ITERATIONS if max_iterations is None else max_iterations,
        tool_delay=0.0,                                 # FIX #4: kill the default 1.0s inter-tool sleep
        max_tokens=mt,                                  # FIX #7: bounded-generation cap (None=default)
        session_id=sid,                                 # TS1: unique sandbox/cwd/registry
        ephemeral_system_prompt=_eph,
    )
    agent._fleet_lane = lane                            # read by DecodeGate.acquire
    agent._fleet_decode_s = 0.0
    agent._fleet_gatewait_s = 0.0
    return agent
