"""AIMD dynamic admission — drives the DecodeGate toward the throughput knee.

The gate (compat.DecodeGate) bounds *concurrent generations* == server
``num_requests_running`` == KV-resident requests (stateless full-history resend,
recon fact #2). This controller is the closed loop that decides how wide the gate
should be, from live ``/metrics`` ground truth — NOT from guessing duty.

Control law (BUILD_SPEC §3, classic AIMD — with the review fixes folded in):
  every AIMD_INTERVAL_S:
    scrape running / waiting / kv / preemptions, derive tok/s.
    ADDITIVE-INCREASE  limit += AIMD_STRIDE     when the gate is *saturated*
        — judged from the GATE's own in_flight (gate.stats()["in_flight"]), NOT the
          lagging async server ``running`` count: a just-released slot is not yet
          "running", so at small limits server-running is structurally < gate and
          saturation would never trip (review fix #3) — AND there is headroom
          (kv < AIMD_KV_HI) AND preemptions are stable AND tok/s is not regressing
          AND waiting is not rising (the KV-bound brake, fix #7) AND we are past the
          post-change DWELL (fix #9).
    HOLD (don't grow, don't cut) when ``waiting`` is positive and RISING — the
        earliest KV-bound "stop growing" signal before KV actually gets hot (fix #7);
        also during the settling DWELL after a limit change (fix #9).
    MULTIPLICATIVE-DECREASE limit *= AIMD_BACKOFF  on a stress signal
        (kv >= AIMD_KV_HI, a preemption RATE over threshold, a large waiting backlog,
        or tok/s dropping > AIMD_TPUT_DROP below a *decaying* recent baseline with
        hysteresis + cooldown) — back off fast, the way TCP halves cwnd on loss.
  Clamp to [DECODE_GATE_MIN, DECODE_GATE_MAX]. NEVER below MIN: a gate of 0 would
  block every worker forever (the gate only re-checks, it does not time out the
  generation). MIN is the floor that keeps the fleet alive.

Throughput attribution (fix #4): by default we difference the server-wide
``generation_tokens_total`` (labelled "server"), which mixes the fleet with any
co-tenant. The engine/CLI can instead feed fleet-only completion tokens via
``record_fleet_tokens(n)``; if any fleet tokens have been recorded we difference THAT
("fleet") and only fall back to the server-wide number otherwise.

The controller is a daemon Thread so it co-exists with the ThreadFleet in one
process. It keeps a `history` of every sample for the final report and can stream
samples to a caller via `on_sample` (the CLI uses this for its live line).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, List, Optional

from . import config, metrics


def _cfg(cfg, name: str, env: str, default):
    """Resolve a tunable: env var wins, then cfg attribute, then the default.

    The new review knobs may not yet exist on config.py (authored by another agent),
    so read them defensively and keep everything env-overridable per the harness rules.
    """
    raw = os.environ.get(env)
    if raw is not None:
        try:
            return type(default)(raw)
        except (TypeError, ValueError):
            pass
    return getattr(cfg, name, default)


class AIMDController(threading.Thread):
    """Resizes ``gate`` toward the knee from live server metrics.

    Args:
        gate:        the compat.DecodeGate to resize (set_limit/get_limit).
        metrics_url: vLLM /metrics endpoint (defaults to config.METRICS_URL).
        cfg:         the config module (injectable for tests); defaults to config.
        on_sample:   optional callback(dict) invoked once per control tick with
                     the full sample (so a CLI / logger can render it live).
    """

    def __init__(self, gate, metrics_url: Optional[str] = None, cfg=config,
                 on_sample: Optional[Callable[[dict], None]] = None) -> None:
        super().__init__(name="aimd-admission", daemon=True)
        self.gate = gate
        self.cfg = cfg
        self.metrics_url = metrics_url or cfg.METRICS_URL
        self.on_sample = on_sample
        self._stop = threading.Event()
        self._tput = metrics.ThroughputMeter()
        # ground-truth references carried across ticks
        self._prev_preempt: Optional[float] = None
        self._prev_waiting: Optional[float] = None     # for the "waiting rising" brake (#7)
        # throughput baseline (#5): a decaying EWMA, not an all-time high-water mark, so a
        # single noisy dip below an unbeatable peak can't force a spurious backoff.
        self._tps_ewma: Optional[float] = None
        self._backoff_cooldown: int = 0                # ticks left before another tput backoff (#5)
        self._dwell: int = 0                           # ticks to HOLD after a limit change (#9)
        # fleet-attributed throughput (#4): cumulative completion tokens the engine feeds us.
        self._fleet_tokens_lock = threading.Lock()
        self._fleet_tokens: float = 0.0
        self._fleet_tokens_seen: bool = False          # did anyone ever feed fleet tokens?
        # tunables (env > config > default); the review knobs may not be on config.py yet.
        self._kv_hi = float(_cfg(cfg, "AIMD_KV_HI", "FLEET_AIMD_KV_HI", 0.85))
        self._sat = float(_cfg(cfg, "AIMD_SATURATION", "FLEET_AIMD_SAT", 0.9))
        self._stride = int(_cfg(cfg, "AIMD_STRIDE", "FLEET_AIMD_STRIDE", 8))
        self._backoff = float(_cfg(cfg, "AIMD_BACKOFF", "FLEET_AIMD_BACKOFF", 0.7))
        self._tput_drop = float(_cfg(cfg, "AIMD_TPUT_DROP", "FLEET_AIMD_TPUT_DROP", 0.07))
        # EWMA smoothing for the tput baseline; small alpha = slow, noise-robust baseline.
        self._tps_alpha = float(_cfg(cfg, "AIMD_TPUT_EWMA", "FLEET_AIMD_TPUT_EWMA", 0.3))
        # cooldown: no SECOND tput backoff within this many ticks (anti-thrash, #5).
        self._tput_cooldown_ticks = int(_cfg(cfg, "AIMD_TPUT_COOLDOWN", "FLEET_AIMD_TPUT_COOLDOWN", 2))
        # dwell: HOLD this many ticks after any limit change so the next tput sample
        # reflects the NEW limit, not the transient (#9). KV/preempt emergencies override.
        self._dwell_ticks = int(_cfg(cfg, "AIMD_DWELL", "FLEET_AIMD_DWELL", 1))
        # waiting backlog that escalates from HOLD to a mild backoff (#7).
        self._waiting_backoff = int(_cfg(cfg, "AIMD_WAITING_BACKOFF", "FLEET_AIMD_WAITING_BACKOFF", 8))
        # preemption RATE (per second) over which we treat preempts as stress (#8); a
        # single +1 over a 4s tick is ~0.25/s and should not by itself halve the gate.
        self._preempt_rate_hi = float(_cfg(cfg, "AIMD_PREEMPT_RATE", "FLEET_AIMD_PREEMPT_RATE", 1.0))
        # observability — the final report walks this
        self.history: List[dict] = []

    # ── lifecycle ───────────────────────────────────────────────────────────
    def stop(self) -> None:
        """Signal the loop to exit; the controller is a daemon so this is best
        effort — the process can also just exit."""
        self._stop.set()

    def record_fleet_tokens(self, completion_tokens: float) -> None:
        """Feed FLEET-attributed completion tokens (review fix #4).

        The engine/worker layer already gets ``completion_tokens`` back from each
        finished conversation (run_conversation result). Call this as workers complete
        (e.g. ``controller.record_fleet_tokens(res["completion_tokens"])``) and the
        control loop will difference the fleet's OWN cumulative tokens for tok/s instead
        of the server-wide counter that mixes in any co-tenant. Thread-safe; if never
        called the loop transparently falls back to the server-wide number.
        """
        if not completion_tokens:
            return
        with self._fleet_tokens_lock:
            self._fleet_tokens += float(completion_tokens)
            self._fleet_tokens_seen = True

    def run(self) -> None:  # Thread entry point
        # First tick primes the throughput differencer (returns None tok/s); we
        # still record it so the history has a t0 baseline.
        interval = float(_cfg(self.cfg, "AIMD_INTERVAL_S", "FLEET_AIMD_INTERVAL", 4.0))
        while not self._stop.is_set():
            self._tick()
            # interruptible sleep so stop() is prompt
            self._stop.wait(interval)

    # ── one control step ────────────────────────────────────────────────────
    def _clamp(self, n: int) -> int:
        lo, hi = self.cfg.DECODE_GATE_MIN, self.cfg.DECODE_GATE_MAX
        return max(lo, min(hi, int(n)))

    def _fleet_token_total(self) -> Optional[float]:
        """Snapshot the fleet's cumulative completion tokens, or None if the engine has
        never fed any (then we fall back to the server-wide counter)."""
        with self._fleet_tokens_lock:
            return self._fleet_tokens if self._fleet_tokens_seen else None

    def _measure_tps(self, sc, now: float) -> Optional[float]:
        """tok/s for this tick: FLEET-attributed if the engine has fed fleet tokens
        (fix #4), else the server-wide differencer. The meter is told which counter-space
        the value belongs to and re-baselines on a source switch, so the fleet and server
        counters (different magnitudes AND timestamps) are NEVER differenced against each
        other. One source per process-lifetime in practice (fleet once fed). Each path
        timestamps with its own clock: the server counter with the scrape wall-time
        ``sc['t']``, the fleet feed with ``now`` (no scrape behind it)."""
        fleet_total = self._fleet_token_total()
        if fleet_total is not None:
            return self._tput.update_source(fleet_total, now, source="fleet")
        gen = sc.get("gen_tokens") if (sc and "gen_tokens" in sc) else None
        return self._tput.update_source(gen, (sc or {}).get("t", now), source="server")

    def _tick(self) -> None:
        limit = self.gate.get_limit()
        sc = metrics.scrape(self.metrics_url)
        now = time.time()
        tps = self._measure_tps(sc, now)

        # GATE-side saturation truth (fix #3): the server's `running` LAGS the gate (a
        # just-released slot is not yet "running"), so it is structurally <= in_flight
        # <= limit and saturation off `running` never trips at small limits. Judge from
        # the gate's own exact in_flight / peak instead. Defensive: if a custom gate lacks
        # stats(), fall back to the server count so we never crash the loop.
        try:
            gstats = self.gate.stats()
            in_flight = float(gstats.get("in_flight", 0.0))
            gate_peak = float(gstats.get("peak_in", in_flight))
        except Exception:
            in_flight = float((sc or {}).get("running", 0.0))
            gate_peak = in_flight

        sample = {
            "t": now,
            "limit_before": limit,
            "running": (sc or {}).get("running"),
            "waiting": (sc or {}).get("waiting"),
            "in_flight": in_flight,            # gate truth (fix #3)
            "kv": (sc or {}).get("kv"),
            "preemptions": (sc or {}).get("preemptions"),
            "tok_s": tps,
            "tok_src": self._tput.attribution,  # "fleet" | "server" | "none" (fix #4)
            "action": "hold",
            "reason": "",
        }

        if sc is None:
            # Server unreachable this tick: do nothing (don't starve the fleet on
            # a transient scrape failure). Keep the limit where it is. Don't burn the
            # dwell/cooldown counters on a blind tick.
            sample["reason"] = "no-metrics"
            self._record(sample, limit)
            return

        waiting = sc.get("waiting", 0.0)
        kv = sc.get("kv", 0.0)
        preempt = sc.get("preemptions")

        # Snapshot the dwell/cooldown state for THIS tick's decisions, then decrement at
        # the end (a dwell armed on tick N must block tick N+1, so we must read it before
        # consuming it). `_changed_this_tick` tracks a re-arm so we don't immediately
        # decrement a freshly-set dwell.
        in_dwell = self._dwell > 0
        in_cooldown = self._backoff_cooldown > 0
        rearmed_dwell = False

        # --- update the decaying tput baseline (#5) --------------------------
        # EWMA over real samples only — a noise-robust "recent normal", not an all-time
        # peak. tput_regress compares the current sample to THIS, with hysteresis (drop
        # threshold) + a cooldown so one dip can't trigger repeated backoffs.
        baseline = self._tps_ewma
        tput_regress = (tps is not None and baseline is not None and baseline > 0
                        and tps < baseline * (1.0 - self._tput_drop)
                        and not in_cooldown)

        # --- stress signals --------------------------------------------------
        kv_hot = kv >= self._kv_hi
        # preemption RATE over the interval, not any +1 (#8). Reset on a counter
        # regression (server restart) so a fresh-zero counter is not read as a jump.
        preempt_rate = 0.0
        if self._prev_preempt is not None and preempt is not None:
            if preempt < self._prev_preempt:
                self._prev_preempt = None          # restart → re-baseline, no stress
            else:
                dt = float(_cfg(self.cfg, "AIMD_INTERVAL_S", "FLEET_AIMD_INTERVAL", 4.0))
                preempt_rate = (preempt - self._prev_preempt) / dt if dt > 0 else 0.0
        preempt_hot = preempt_rate >= self._preempt_rate_hi
        # waiting backlog: a large queue means we're already over the KV knee → mild cut.
        waiting_overflow = waiting >= self._waiting_backoff
        # waiting positive AND rising is the gentle brake (#7): HOLD, don't grow yet.
        waiting_rising = (waiting > 0 and self._prev_waiting is not None
                          and waiting > self._prev_waiting)

        new_limit = limit
        if kv_hot or preempt_hot or waiting_overflow or tput_regress:
            # MULTIPLICATIVE DECREASE — react fast to loss/pressure.
            new_limit = self._clamp(int(limit * self._backoff))
            reasons = []
            if kv_hot:
                reasons.append(f"kv>={self._kv_hi:.2f}")
            if preempt_hot:
                reasons.append(f"preempt{preempt_rate:.1f}/s")
            if waiting_overflow:
                reasons.append(f"wait>={self._waiting_backoff}")
            if tput_regress:
                reasons.append("tput-drop")
            sample["action"], sample["reason"] = "backoff", ",".join(reasons)
            # DECAY the prior baseline (#6) — do NOT set it from this sample. If the
            # backoff tick has tps=None, setting best=current would zero it and DISABLE
            # regress detection forever. Decaying the EWMA keeps regress armed and lowers
            # the bar to match the smaller batch we just chose.
            if self._tps_ewma is not None:
                self._tps_ewma *= self._backoff
            # arm the anti-thrash cooldown + settle dwell after the change.
            self._backoff_cooldown = self._tput_cooldown_ticks
            if new_limit != limit:
                self._dwell = self._dwell_ticks
                rearmed_dwell = True
        elif in_dwell:
            # DWELL after a recent limit change (#9): let the next tput sample reflect the
            # new limit, not the transient. (We already handle hard KV/preempt above, so
            # this only blocks discretionary GROWTH.)
            sample["action"] = "hold"
            sample["reason"] = f"dwell({self._dwell})"
        elif waiting_rising:
            # gentle brake (#7): queue forming and growing → stop growing, don't cut yet.
            sample["action"] = "hold"
            sample["reason"] = f"waiting+({waiting:.0f})"
        else:
            # SATURATION from the GATE (fix #3): the batch is "full" when the gate's own
            # in_flight (peak across the interval, to ride out the lag of a momentary
            # release) is at/near the limit — independent of the lagging server count.
            sat_ref = max(in_flight, gate_peak)
            saturated = sat_ref >= limit * self._sat
            headroom = kv < self._kv_hi
            if saturated and headroom:
                # ADDITIVE INCREASE — only widen a full batch the server can take.
                new_limit = self._clamp(limit + self._stride)
                sample["action"] = "increase" if new_limit > limit else "at-max"
                sample["reason"] = f"sat({sat_ref:.0f}/{limit})"
                if new_limit != limit:
                    self._dwell = self._dwell_ticks    # settle before the next decision
                    rearmed_dwell = True
            else:
                sample["action"] = "hold"
                sample["reason"] = ("not-saturated" if not saturated else "no-kv-headroom")

        # fold a real sample into the EWMA baseline (#5) — AFTER the regress test, so a
        # dip is judged against the prior baseline, then updates it.
        if tps is not None:
            self._tps_ewma = (tps if self._tps_ewma is None
                              else (1 - self._tps_alpha) * self._tps_ewma
                                   + self._tps_alpha * tps)
        sample["tps_baseline"] = (round(self._tps_ewma, 1)
                                  if self._tps_ewma is not None else None)

        if new_limit != limit:
            self.gate.set_limit(new_limit)

        # consume the per-tick timers AT THE END so a dwell/cooldown armed on tick N is
        # still in force for the decision on tick N+1 (#5/#9). A dwell re-armed THIS tick
        # is left at its full value (don't immediately burn one of its ticks).
        if self._backoff_cooldown > 0:
            self._backoff_cooldown -= 1
        if self._dwell > 0 and not rearmed_dwell:
            self._dwell -= 1

        if preempt is not None:
            self._prev_preempt = preempt
        self._prev_waiting = waiting

        self._record(sample, new_limit)

    def _record(self, sample: dict, limit_after: int) -> None:
        sample["limit_after"] = limit_after
        self.history.append(sample)
        if self.on_sample is not None:
            try:
                self.on_sample(sample)
            except Exception:
                pass  # a logging callback must never kill the controller
