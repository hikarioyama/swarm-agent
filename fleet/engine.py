"""ThreadFleet — the single-process, bounded ThreadPool engine.

This is the v0.2 hot path. HermesAgent is SYNC+THREADED (recon fact #1): each LLM
turn blocks an OS thread but releases the GIL during socket I/O, so dozens of
generations overlap inside ONE process — no asyncio, no per-worker subprocess.

Two independent bounds, by design:
  * the POOL bounds *enrolled* workers (threads in flight). Tool-executing workers
    hold no server KV (stateless resend, fact #2), so we OVERSUBSCRIBE:
        enrolled = clamp(OVERSUB_FACTOR * gate_limit, gate_limit, ENROLL_MAX)
    re-evaluated every loop because AIMD moves the gate limit underneath us.
  * the DECODE GATE (inside compat's forwarder wrapper) bounds *generations* ==
    server num_requests_running == KV-resident requests. The pool can enrol 2×
    the gate; the surplus sits in tool code or blocks on the gate, holding no KV.

The engine itself carries NO LLM and does NO reasoning — it is a dumb, fast
admission loop. Intelligence (decomposition, reduction) lives in the tasks on the
Board; coordination is stigmergic (claim ready → run → write result → unlock deps).

`run()` returns a summary dict the CLI turns into the operating-point report.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import thread as _ftypes
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable, Dict, Optional

from . import config, metrics, v3
from .worker import has_visible_text, run_task_local

# Serialises the brief ``threading.Thread`` swap in _DaemonThreadPoolExecutor (below).
_DAEMON_SPAWN_LOCK = threading.Lock()

# Abandoned (timed-out, un-killable) WRITE-capable worker futures still running in the
# background. A new writing goal waits these out (runner) before starting, so it can't race a
# stale writer. Pruned lazily as the daemons finally finish.
_ABANDONED_WRITERS: "set" = set()
_ABANDONED_LOCK = threading.Lock()


def abandoned_writers_alive() -> bool:
    """True iff any abandoned WRITE worker is still running. Prunes finished ones."""
    with _ABANDONED_LOCK:
        for f in list(_ABANDONED_WRITERS):
            if f.done():
                _ABANDONED_WRITERS.discard(f)
        return bool(_ABANDONED_WRITERS)


class _DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor whose workers are DAEMON threads, so one stuck in a never-returning
    blocking command (which we cannot interrupt) can't block interpreter exit or our own
    shutdown.

    Version-robustness: we daemonize WITHOUT copying CPython's private executor internals
    (which differ across versions — e.g. 3.14 dropped ``_initializer``/``_initargs`` and
    changed ``_worker``'s signature). Instead we briefly install a daemon-defaulting
    ``threading.Thread`` factory and delegate to whatever the running interpreter's own
    ``_adjust_thread_count`` does. If anything about that fails on some future Python, we fall
    back to the normal (non-daemon) pool — the swarm still un-wedges via timeout-abandon +
    ``shutdown(wait=False)``; only clean process-exit on a *stuck* worker degrades."""

    def _adjust_thread_count(self):
        real_thread = threading.Thread

        class _DaemonThread(real_thread):
            def __init__(self, *a, **kw):
                kw.setdefault("daemon", True)
                super().__init__(*a, **kw)

        with _DAEMON_SPAWN_LOCK:                    # serialise the brief global swap
            before = set(_ftypes._threads_queues)
            threading.Thread = _DaemonThread       # stdlib spawns workers via threading.Thread
            try:
                super()._adjust_thread_count()     # delegate to THIS interpreter's own logic
            except Exception:                      # never let daemonisation break the pool
                threading.Thread = real_thread
                super()._adjust_thread_count()     # plain (non-daemon) fallback
            finally:
                threading.Thread = real_thread
            # De-register the worker threads WE just spawned from concurrent.futures' atexit
            # hook (_python_exit joins every registered thread — daemon or not — which would
            # re-introduce the exit wedge on a stuck worker). Daemon-ness already excludes them
            # from threading._shutdown; this excludes them from the futures atexit join too.
            for t in set(_ftypes._threads_queues) - before:
                _ftypes._threads_queues.pop(t, None)


def _enrolled_target(gate, cfg) -> int:
    """enrolled = clamp(OVERSUB_FACTOR * gate_limit, gate_limit, ENROLL_MAX).

    Never fewer than min(gate_limit, ENROLL_MAX) (you want at least as many threads
    as decode slots to fill the gate) and NEVER more than ENROLL_MAX (the hard thread
    cap == the pool's max_workers).

    BUG FIX (review #1): the old form `max(limit, min(ENROLL_MAX, want))` made the
    *lower* bound win, so when `limit > ENROLL_MAX` it returned `limit` and the engine
    tried to keep MORE futures in flight than the pool has worker threads. The surplus
    futures queue un-started inside the ThreadPoolExecutor while we still count them as
    in-flight (len(futs)), and their board tasks sit RUNNING but never get a thread —
    starving the gate and stalling progress. ENROLL_MAX must be the OUTER cap:
        clamp into [1, ENROLL_MAX], floored at min(limit, ENROLL_MAX).
    """
    limit = gate.get_limit() if gate is not None else cfg.TARGET_INFLIGHT
    want = int(cfg.OVERSUB_FACTOR * limit)
    # ENROLL_MAX is the hard outer ceiling; floor at the (also-capped) gate limit so we
    # still field at least one thread per decode slot when ENROLL_MAX allows it.
    return max(1, min(cfg.ENROLL_MAX, max(limit, want)))


class ThreadFleet:
    """Single-process bounded ThreadPool over the Board.

    Args:
        board:    a Board / SqliteBoard (claim_ready/complete/fail/unfinished/...).
        gate:     compat.DecodeGate (or None for an un-gated A/B run). Drives the
                  enrolled target and is the actual KV bound via compat's wrapper.
        cfg:      config module (injectable); defaults to config.
        on_event: optional callback(kind, tid, **extra) for progress/logging.
        worker_fn: injectable task runner; defaults to worker.run_task_local.
    """

    def __init__(self, board, gate, cfg=config,
                 on_event: Optional[Callable[..., None]] = None,
                 worker_fn=None) -> None:
        self.board = board
        self.gate = gate
        self.cfg = cfg
        self.on_event = on_event or (lambda *a, **k: None)
        self.worker_fn = worker_fn or run_task_local
        self.max_retries = cfg.MAX_RETRIES
        # duty / throughput accounting from live metrics. Sampling runs OFF the hot
        # loop on a daemon thread (see _sampler_loop / review #2) so a slow /metrics
        # never blocks dispatch/harvest.
        self.duty = metrics.DutyIntegrator()
        self.t0 = time.time()
        # Cached last successful scrape (last-value fallback when /metrics is slow/down).
        self._last_scrape: Optional[dict] = None
        self._stop_sampler = threading.Event()
        # Sampler scrape timeout: short, so a wedged /metrics endpoint cannot hold the
        # sampler thread for the full default ~3s — it just reuses the last value.
        self._sample_timeout = self.cfg._envf("FLEET_SAMPLE_TIMEOUT", 1.0) \
            if hasattr(self.cfg, "_envf") else 1.0
        self._sample_period = self.cfg._envf("FLEET_SAMPLE_PERIOD", 1.0) \
            if hasattr(self.cfg, "_envf") else 1.0

    def _emit(self, kind: str, tid: Optional[str], counts=None, **extra) -> None:
        # review #4: do NOT call board.counts() (O(tasks) under the board lock) on
        # EVERY event — that contends the lock workers need. The hot loop computes
        # counts at most once per tick and passes it through here; callers that pass
        # nothing get whatever the loop cached (slightly stale ready-count is fine for
        # progress display). on_event signature is unchanged (counts kw remains).
        self.on_event(kind, tid, counts=counts, **extra)

    def _sampler_loop(self) -> None:
        """Daemon: scrape /metrics with a SHORT timeout and update duty + the cached
        last scrape. Runs OFF the engine's hot loop (review #2) so a slow/blocked
        /metrics endpoint can never stall harvest/dispatch/dep-unlock. On a failed
        scrape we keep the previous cached value (last-value fallback) rather than
        feeding None into the integrator."""
        while not self._stop_sampler.is_set():
            sc = metrics.scrape(self.cfg.METRICS_URL, timeout=self._sample_timeout)
            if sc is not None:
                self.duty.update(sc)
                self._last_scrape = sc
            # wait() returns early if stop is set, so shutdown is prompt.
            self._stop_sampler.wait(self._sample_period)

    def _v3_after_complete(self, tid: str, text: str) -> None:
        try:
            chem = v3.parse_chem(text)
            self.board.record_signal(tid, chem)
            if not v3.enabled("diversity"):
                return
            snapshot = self.board.results()
            for reducer_tid, task in snapshot.items():
                if getattr(task, "lane", "") != "reducer" or tid not in getattr(task, "deps", []):
                    continue
                deps = getattr(task, "deps", [])
                if deps and all(snapshot.get(dep) is not None and snapshot[dep].state.value == "done"
                                for dep in deps):
                    self.board.spawn_referee(reducer_tid)
        except Exception:
            return

    def run(self) -> Dict[str, object]:
        """Drive the board to completion. Loop invariant: keep `enrolled` workers
        in flight (re-evaluated each pass so AIMD limit moves take effect), claim
        lane-aware ready tasks to fill free slots, harvest completions, write
        results (unlocking deps), requeue failures. Detect deadlock (nothing in
        flight, work remains, nothing ready)."""
        # ENROLL_MAX is the hard ceiling on threads; the live target floats below it.
        results: Dict[str, dict] = {}
        stranded = 0                                            # review #3: see break path

        # review #2: spin up the off-loop metrics sampler. It owns ALL /metrics I/O so
        # the hot loop below NEVER blocks on the network (urllib timeout could be ~3s).
        sampler = threading.Thread(target=self._sampler_loop,
                                   name="fleet-metrics-sampler", daemon=True)
        sampler.start()
        ex = _DaemonThreadPoolExecutor(max_workers=self.cfg.ENROLL_MAX,
                                       thread_name_prefix="fleet-w")
        try:
            futs: Dict[object, str] = {}                        # future -> task id
            started: Dict[object, float] = {}                   # future -> monotonic start
            lane_of: Dict[object, str] = {}                     # future -> lane
            task_timeout = float(getattr(self.cfg, "TASK_TIMEOUT_S", 0) or 0)

            while self.board.unfinished() > 0 or futs:
                target = _enrolled_target(self.gate, self.cfg)
                slots = target - len(futs)
                # review #4: counts() is O(tasks) under the board lock — compute it
                # AT MOST ONCE per loop tick and reuse the snapshot for every event
                # emitted this pass, rather than per dispatch/done event.
                tick_counts = None
                if slots > 0:
                    # lane-aware: Board.claim_ready injects dep_results into spec.meta;
                    # the DecodeGate then serves the claimed tasks by lane priority.
                    claimed = self.board.claim_ready(slots)
                    if claimed:
                        tick_counts = self.board.counts()
                    for t in claimed:
                        f = ex.submit(self.worker_fn, t.spec())
                        futs[f] = t.id
                        started[f] = time.monotonic()
                        lane_of[f] = t.lane
                        self._emit("dispatch", t.id, counts=tick_counts,
                                   target=target, running=len(futs) + 1)

                if not futs:
                    # nothing in flight: either we're done, or we're wedged
                    if self.board.unfinished() > 0:
                        # review #3: a real dep-deadlock (work remains, nothing
                        # ready, nothing in flight). Record the stranded count so
                        # the summary/CLI can exit non-zero — counts['failed'] alone
                        # would report SUCCESS for tasks stuck PENDING forever.
                        stranded = self.board.unfinished()
                        self._emit("deadlock", None,
                                   counts=self.board.counts(), stranded=stranded)
                    break

                # Wake periodically even if no future completes, so we can grow
                # enrolled when AIMD widens the gate. (Duty/throughput sampling is
                # the sampler thread's job now — review #2.)
                done, _ = wait(list(futs), timeout=1.0, return_when=FIRST_COMPLETED)

                # ABANDON rather than kill: terminal timeout + cleanup_vm were measured
                # not to interrupt in-flight commands. Daemon workers and non-blocking
                # shutdown make abandonment safe; the run-app skill prevents the cause.
                if task_timeout > 0:
                    now = time.monotonic()
                    overdue = [f for f in list(futs) if f not in done
                               and now - started.get(f, now) > task_timeout]
                    for f in overdue:
                        tid = futs.pop(f)
                        started.pop(f, None)
                        lane = lane_of.pop(f, "")
                        write_capable = config.lane_writes(lane)
                        if write_capable:
                            with _ABANDONED_LOCK:
                                _ABANDONED_WRITERS.add(f)   # a new writer must wait this out (runner)
                        error = f"timeout: exceeded FLEET_TASK_TIMEOUT={task_timeout:g}s"
                        self.board.fail(tid, error, 0)
                        self._emit("fail", tid, counts=self.board.counts(),
                                   error=error + (" [write worker still alive]" if write_capable else ""))

                if done and tick_counts is None:
                    tick_counts = self.board.counts()           # one snapshot for the batch
                for f in done:
                    tid = futs.pop(f)
                    started.pop(f, None)
                    lane_of.pop(f, None)
                    try:
                        res = f.result()
                        if not has_visible_text(res.get("text", "")):
                            raise RuntimeError("worker returned an empty visible response")
                        results[tid] = res
                        complete_ok = self.board.complete(tid, res.get("text", ""))
                        if complete_ok is not False and v3.any_on():
                            self._v3_after_complete(tid, res.get("text", ""))
                        self._emit("done", tid, counts=tick_counts,
                                   wall_s=res.get("wall_s"),
                                   decode_s=res.get("decode_s"),
                                   tool_s=res.get("tool_s"),
                                   gatewait_s=res.get("gatewait_s"),
                                   turns=res.get("turns"))
                    except Exception as e:
                        requeued = self.board.fail(tid, repr(e), self.max_retries)
                        self._emit("requeue" if requeued else "fail", tid,
                                   counts=tick_counts, error=repr(e)[:160])
        finally:
            # review #2: always stop + join the sampler so no daemon thread or socket
            # leaks past run() (matters for repeated runs / A-B in one process).
            self._stop_sampler.set()
            sampler.join(timeout=self._sample_timeout + self._sample_period + 1.0)
            ex.shutdown(wait=False, cancel_futures=True)        # never await stuck workers

        wall_s = round(time.time() - self.t0, 1)
        gate_stats = self.gate.stats() if self.gate is not None else None
        mean_running = round(self.duty.mean_running(), 2)
        # NOTE (duty denominator): we divide mean(running) by the enrolled target taken
        # at the FINAL gate limit. For a STATIC gate (the lead measures the operating
        # point with a static gate) the enrolled target is constant, so this is exact.
        # Under AIMD the gate limit (hence enrolled) moves over the run, so duty should
        # instead divide by the TIME-AVERAGED enrolled count; the static-gate case is
        # the accepted measurement path, so this approximation is documented, not fixed.
        summary = {
            "counts": self.board.counts(),
            "wall_s": wall_s,
            "results": results,
            "board_results": self.board.results(),
            "gate_stats": gate_stats,
            "duty": self.duty.duty(_enrolled_target(self.gate, self.cfg)),
            "mean_running": mean_running,
            "peak_running": round(self.duty.peak_running, 1),
            # review #3: surface stranded/unfinished so the CLI can exit non-zero on a
            # deadlock even though counts['failed'] may be 0 (tasks stuck PENDING).
            "stranded": stranded,
            "unfinished": self.board.unfinished(),
        }
        return summary
