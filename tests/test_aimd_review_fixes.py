"""Unit tests for the reviewed AIMD/metrics defects (no live server needed).

Run:
  PYTHONPATH=. /home/hikari/.hermes/hermes-agent/venv/bin/python tests/test_aimd_review_fixes.py
"""
import sys

import fleet.metrics as m
import fleet.admission as a
from fleet.admission import AIMDController


# ── a fake gate exposing the same surface the controller uses ────────────────
class FakeGate:
    def __init__(self, limit, in_flight=0, peak_in=None, full=False):
        self._limit = int(limit)
        self._full = full              # if True, in_flight/peak always == limit (full batch)
        self._in = int(in_flight)
        self._peak = int(peak_in if peak_in is not None else in_flight)

    def get_limit(self):
        return self._limit

    def set_limit(self, n):
        self._limit = int(n)

    def stats(self):
        inflight = self._limit if self._full else self._in
        peak = self._limit if self._full else self._peak
        return {"limit": self._limit, "in_flight": inflight,
                "peak_in": peak, "waiting": 0}


class FakeCfg:
    DECODE_GATE_MIN = 8
    DECODE_GATE_MAX = 96
    AIMD_INTERVAL_S = 4.0
    AIMD_STRIDE = 8
    AIMD_BACKOFF = 0.7
    AIMD_KV_HI = 0.85
    AIMD_TPUT_DROP = 0.07
    AIMD_SATURATION = 0.9
    METRICS_URL = "http://x/metrics"


def _ctrl(gate, **kw):
    c = AIMDController(gate, "http://x/metrics", FakeCfg)
    # make the dwell a non-issue for tests that aren't exercising it
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _patch_scrape(monkey):
    """monkey: a list/iter of scrape dicts (or callable) returned per _tick."""
    state = {"i": 0}

    def fake(url=None, timeout=3.0):
        seq = monkey
        if callable(seq):
            return seq(state["i"])
        d = seq[min(state["i"], len(seq) - 1)]
        state["i"] += 1
        return None if d is None else dict(d)
    a.metrics.scrape = fake
    return state


_ORIG_SCRAPE = a.metrics.scrape


def _restore():
    a.metrics.scrape = _ORIG_SCRAPE


def sc(running=0, waiting=0, kv=0.0, preemptions=0, gen=0, t=0.0):
    return {"running": float(running), "waiting": float(waiting), "kv": float(kv),
            "preemptions": float(preemptions), "gen_tokens": float(gen), "t": float(t)}


checks = []


def ok(name, cond, detail=""):
    checks.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


# ── FIX #3: saturation judged from GATE in_flight, grows even though server-running lags ──
def test_saturated_from_gate_grows():
    g = FakeGate(limit=8, in_flight=8, peak_in=8)         # gate FULL
    c = _ctrl(g, _dwell=0, _dwell_ticks=0)
    # server `running`=3 LAGS far below the gate (a just-released slot etc.). kv cool.
    _patch_scrape([sc(running=3, kv=0.2, gen=0), sc(running=3, kv=0.2, gen=100, t=4)])
    c._tick(); c._tick()
    ok("fix#3 saturated-from-gate -> grow",
       g.get_limit() == 16,
       f"limit 8->{g.get_limit()} (server running=3 would NOT have saturated)")
    _restore()


# ── FIX #1+kv: kv>hi -> backoff ──────────────────────────────────────────────
def test_kv_hot_backoff():
    g = FakeGate(limit=40, in_flight=40, peak_in=40)
    c = _ctrl(g)
    _patch_scrape([sc(running=40, kv=0.90, gen=0)])
    c._tick()
    ok("kv>hi -> backoff", g.get_limit() == 28, f"40*0.7=28, got {g.get_limit()}")
    _restore()


# ── FIX #7: waiting rising -> HOLD (do not grow) ─────────────────────────────
def test_waiting_rising_holds():
    g = FakeGate(limit=40, in_flight=40, peak_in=40)      # gate full => would grow
    c = _ctrl(g, _dwell=0, _dwell_ticks=0)
    # tick1: waiting=2 (prime prev_waiting), tick2: waiting=5 (rising) -> HOLD
    _patch_scrape([sc(running=40, waiting=2, kv=0.3, gen=0),
                   sc(running=40, waiting=5, kv=0.3, gen=100, t=4)])
    c._tick()
    first = g.get_limit()
    c._tick()
    last_sample = c.history[-1]
    ok("fix#7 waiting rising -> hold",
       g.get_limit() == first and last_sample["action"] == "hold"
       and "waiting+" in last_sample["reason"],
       f"limit stayed {g.get_limit()} action={last_sample['action']} reason={last_sample['reason']}")
    _restore()


# ── FIX #5: tps EWMA dip -> backoff, with cooldown (no 2nd backoff next tick) ──
def test_tps_ewma_dip_backoff_with_cooldown():
    g = FakeGate(limit=40, in_flight=10, peak_in=10)      # NOT saturated -> no growth noise
    c = _ctrl(g, _tput_cooldown_ticks=2, _dwell=0, _dwell_ticks=0)
    # build a baseline ~1000 tok/s over several good ticks, then a sharp dip.
    seq = [
        sc(running=10, kv=0.3, gen=0,    t=0),    # prime differ
        sc(running=10, kv=0.3, gen=4000, t=4),    # 1000/s
        sc(running=10, kv=0.3, gen=8000, t=8),    # 1000/s -> ewma ~1000
        sc(running=10, kv=0.3, gen=10000, t=12),  # 500/s -> 50% drop -> BACKOFF
        sc(running=10, kv=0.3, gen=12000, t=16),  # 500/s -> would drop again, but COOLDOWN
    ]
    _patch_scrape(seq)
    c._tick(); c._tick(); c._tick()
    limit_before_dip = g.get_limit()
    c._tick()                                     # the dip tick
    after_dip = g.get_limit()
    dip_sample = c.history[-1]
    c._tick()                                     # cooldown tick: must NOT backoff again
    after_cool = g.get_limit()
    cool_sample = c.history[-1]
    ok("fix#5 tps dip -> backoff",
       after_dip < limit_before_dip and dip_sample["action"] == "backoff"
       and "tput-drop" in dip_sample["reason"],
       f"{limit_before_dip}->{after_dip} action={dip_sample['action']}")
    ok("fix#5 cooldown suppresses 2nd backoff",
       cool_sample["action"] != "backoff" and after_cool == after_dip,
       f"next-tick action={cool_sample['action']} limit={after_cool} (cooldown left={c._backoff_cooldown})")
    _restore()


# ── FIX #6: None tps on a backoff must NOT zero the baseline (regress stays armed) ──
def test_none_tps_on_backoff_keeps_regress_armed():
    g = FakeGate(limit=40, in_flight=10, peak_in=10)
    c = _ctrl(g, _tput_cooldown_ticks=0, _dwell=0, _dwell_ticks=0)
    # establish an EWMA baseline.
    _patch_scrape([
        sc(running=10, kv=0.3, gen=0,    t=0),
        sc(running=10, kv=0.3, gen=4000, t=4),    # ewma ~1000
        # kv_hot backoff tick WITH a partial (no gen_tokens) sample -> tps None
        {"running": 10.0, "waiting": 0.0, "kv": 0.90, "preemptions": 0.0, "t": 8.0},
    ])
    c._tick(); c._tick()
    base_before = c._tps_ewma
    c._tick()                                     # kv-hot backoff on a None-tps tick
    base_after = c._tps_ewma
    ok("fix#6 None-tps backoff decays prior baseline (not =0)",
       base_after is not None and base_after > 0
       and abs(base_after - base_before * c._backoff) < 1e-6,
       f"baseline {round(base_before,1)} -> {round(base_after,1)} (decayed by {c._backoff}, NOT zeroed)")
    # and regress detection is still functional afterwards: a dip now still trips.
    c._tput_cooldown_ticks = 0
    c._backoff_cooldown = 0
    # feed a low tps (must be < baseline*(1-drop)); baseline ~700 now
    _patch_scrape([
        sc(running=10, kv=0.3, gen=0,    t=12),   # re-prime the fleet differ baseline
        sc(running=10, kv=0.3, gen=400,  t=16),   # 100/s << 700 -> regress
    ])
    c._tick(); c._tick()
    ok("fix#6 regress still armed after None-tps backoff",
       c.history[-1]["action"] == "backoff" and "tput-drop" in c.history[-1]["reason"],
       f"action={c.history[-1]['action']} reason={c.history[-1]['reason']}")
    _restore()


# ── FIX #8: single +1 preempt over a 4s tick is NOT enough to backoff ────────
def test_preempt_rate_threshold():
    g = FakeGate(limit=40, in_flight=10, peak_in=10)
    c = _ctrl(g, _preempt_rate_hi=1.0, _dwell=0, _dwell_ticks=0)
    _patch_scrape([
        sc(running=10, kv=0.3, preemptions=3, gen=0, t=0),    # prime prev_preempt=3
        sc(running=10, kv=0.3, preemptions=4, gen=10, t=4),   # +1 over 4s = 0.25/s < 1.0 -> no backoff
    ])
    c._tick()
    before = g.get_limit()
    c._tick()
    ok("fix#8 single +1 preempt/4s -> NO backoff",
       g.get_limit() == before and c.history[-1]["action"] != "backoff",
       f"rate=0.25/s action={c.history[-1]['action']}")
    # restart (counter regresses) must NOT be read as a jump
    _patch_scrape([
        sc(running=10, kv=0.3, preemptions=4, gen=0, t=8),
        sc(running=10, kv=0.3, preemptions=0, gen=10, t=12),  # counter reset
    ])
    c._tick(); c._tick()
    ok("fix#8 counter regression (restart) -> no false backoff",
       c.history[-1]["action"] != "backoff",
       f"action={c.history[-1]['action']} reason={c.history[-1]['reason']}")
    _restore()


# ── FIX #9: DWELL holds growth for a tick after a limit change ───────────────
def test_dwell_after_change():
    # `full` gate: in_flight always == limit, so the batch is ALWAYS saturated and the
    # ONLY thing that can stop a grow is the dwell (isolates fix #9).
    g = FakeGate(limit=8, full=True)
    c = _ctrl(g, _dwell_ticks=1, _tput_cooldown_ticks=0)
    _patch_scrape([
        sc(running=8, kv=0.2, gen=0, t=0),       # grow 8->16, arms dwell=1
        sc(running=8, kv=0.2, gen=100, t=4),     # dwell active -> HOLD (no grow to 24)
        sc(running=8, kv=0.2, gen=200, t=8),     # dwell elapsed -> grow 16->24
    ])
    c._tick()
    ok("fix#9 first tick grows + arms dwell", g.get_limit() == 16, f"limit={g.get_limit()}")
    c._tick()
    ok("fix#9 dwell HOLDs the next tick",
       g.get_limit() == 16 and c.history[-1]["action"] == "hold"
       and "dwell" in c.history[-1]["reason"],
       f"limit={g.get_limit()} action={c.history[-1]['action']} reason={c.history[-1]['reason']}")
    c._tick()
    ok("fix#9 grows again after dwell elapses", g.get_limit() == 24, f"limit={g.get_limit()}")
    _restore()


# ── FIX #4: fleet-attributed throughput preferred over server-wide ───────────
def test_fleet_attributed_throughput():
    g = FakeGate(limit=40, in_flight=10, peak_in=10)
    c = _ctrl(g, _dwell=0, _dwell_ticks=0)
    # As workers complete the engine feeds fleet completion tokens. The server gen_tokens
    # counter jumps hugely (a co-tenant), but the fleet's OWN tokens grow modestly. Once
    # any fleet token is recorded the controller differences the FLEET counter, not the
    # server-wide one. NOTE: the first fleet tick re-baselines (source switch from the
    # server prime), so the first fleet RATE lands on the second fleet tick.
    clock = {"t": 1000.0}
    orig_time = a.time.time
    a.time.time = lambda: clock["t"]
    try:
        c.record_fleet_tokens(400)      # worker A finished -> enter fleet mode
        _patch_scrape([
            sc(running=10, kv=0.3, gen=0,      t=0),
            sc(running=10, kv=0.3, gen=999999, t=4),   # co-tenant noise (ignored in fleet)
            sc(running=10, kv=0.3, gen=999999, t=8),
        ])
        c._tick()                        # fleet=400 -> re-baseline (source switch), no rate
        clock["t"] = 1004.0
        c.record_fleet_tokens(400)       # worker B finished -> fleet cumulative now 800
        c._tick()                        # fleet 800 vs 400 over 4s -> 100 tok/s
        s = c.history[-1]
    finally:
        a.time.time = orig_time
    ok("fix#4 fleet-attributed tok/s used (not server-wide)",
       s["tok_src"] == "fleet" and s["tok_s"] is not None and abs(s["tok_s"] - 100.0) < 1.0,
       f"tok_src={s['tok_src']} tok_s={s['tok_s']} (fleet delta=400/4s=100, NOT 999999 co-tenant)")
    _restore()

    # and when NOT fed, it falls back to server-wide.
    g2 = FakeGate(limit=40, in_flight=10, peak_in=10)
    c2 = _ctrl(g2, _dwell=0, _dwell_ticks=0)
    _patch_scrape([sc(running=10, kv=0.3, gen=0, t=0), sc(running=10, kv=0.3, gen=800, t=4)])
    c2._tick(); c2._tick()
    s2 = c2.history[-1]
    ok("fix#4 falls back to server-wide when unfed",
       s2["tok_src"] == "server" and abs(s2["tok_s"] - 200.0) < 1.0,
       f"tok_src={s2['tok_src']} tok_s={s2['tok_s']} (800/4=200)")
    _restore()


# ── floor: never below MIN ───────────────────────────────────────────────────
def test_never_below_min():
    g = FakeGate(limit=9, in_flight=9, peak_in=9)
    c = _ctrl(g)
    _patch_scrape([sc(running=9, kv=0.99, gen=0)])   # hammer kv
    for _ in range(10):
        _patch_scrape([sc(running=9, kv=0.99, gen=0)])
        c._tick()
    ok("never below DECODE_GATE_MIN", g.get_limit() == FakeCfg.DECODE_GATE_MIN,
       f"limit={g.get_limit()} min={FakeCfg.DECODE_GATE_MIN}")
    _restore()


if __name__ == "__main__":
    for fn in [test_saturated_from_gate_grows, test_kv_hot_backoff,
               test_waiting_rising_holds, test_tps_ewma_dip_backoff_with_cooldown,
               test_none_tps_on_backoff_keeps_regress_armed, test_preempt_rate_threshold,
               test_dwell_after_change, test_fleet_attributed_throughput,
               test_never_below_min]:
        print(f"\n{fn.__name__}:")
        try:
            fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            checks.append((fn.__name__, False, repr(e)))
    passed = sum(1 for _, c, _ in checks if c)
    print(f"\n{'='*60}\n{passed}/{len(checks)} checks passed")
    sys.exit(0 if passed == len(checks) else 1)
