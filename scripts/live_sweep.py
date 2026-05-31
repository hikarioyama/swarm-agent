"""Live operating-point sweep for the v0.2 thread engine + DecodeGate.

For each fixed gate N, run the REAL agent fleet over a pool of ~8K-token tasks and
measure, from the server's own /metrics, whether the gate pins concurrent generations
to ≈ N and what aggregate throughput / duty that yields — reproducing FLEET_OPTIMUM §4
(C16–C64) through the real AIAgent path (not the synthetic httpx probe).

    cd ~/projects/step37-harness
    PYTHONPATH=. /home/hikari/.hermes/hermes-agent/venv/bin/python scripts/live_sweep.py \
        --tasks examples/sweep_8k.jsonl --gates 8,16,32,48,64 --mult 6 --min-pool 96

Reports per N: windowed aggregate tok/s, mean running (≈ decode batch), occupancy
(mean_running/N), KV%, fleet duty, per-worker decode_s/tool_s. Writes results/live_sweep.json.
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time

# Sweep-tuned config defaults — set BEFORE importing fleet (config reads env at import).
# These bound worker output (one clean ~200-tok generation per task, matching FLEET_OPTIMUM's
# bounded synthetic probe) and oversubscribe enough enrolled threads to keep the gate saturated
# at the real agent's duty for N up to 64 (enrolled = min(ENROLL_MAX, OVERSUB*N)). All overridable.
# Step-3.7 is a verbose reasoner; reasoning_effort='none' makes it answer DIRECTLY so each
# task is ONE bounded generation (no thousand-token <think>). max_tokens=1024 fits the direct
# answer (finish='stop'); NO_CONTINUE rewrites any residual length-truncation to 'stop' so a
# task never thrashes into multi-continuation re-prefill. This matches FLEET_OPTIMUM's bounded
# single-generation-per-request methodology.
os.environ.setdefault("FLEET_REASONING_EFFORT", "none")
os.environ.setdefault("FLEET_NO_CONTINUE", "1")
os.environ.setdefault("FLEET_MAX_TOKENS", "1024")
os.environ.setdefault("FLEET_OVERSUB", "4.0")
os.environ.setdefault("FLEET_ENROLL_MAX", "320")
# No-tool sweep tasks never touch the filesystem, so per-worker sandbox isolation is pure
# overhead here — make it a clean no-op (it stays ON by default for real tool-using fleets).
os.environ.setdefault("FLEET_SANDBOX_ISOLATE", "0")

sys.path.insert(0, ".")
from fleet import compat, config, metrics  # noqa: E402
from fleet.board import Board, Task  # noqa: E402
from fleet.engine import ThreadFleet  # noqa: E402

# No-tool lane for the sweep: guarantees a SINGLE clean generation per task (the model has
# no tools to call, so the loop ends after one assistant turn) AND keeps the prompt ≈ 8K total
# (no tool-schema prefix on top of the ~8K context), matching FLEET_OPTIMUM's ~8.1-8.2k prompt.
SWEEP_LANE = os.environ.get("FLEET_SWEEP_LANE", "reducer")


def load_pool(path, n):
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or len(out) >= n:
                continue
            d = json.loads(line)
            out.append((str(d.get("id", i)), d["prompt"], d.get("lane", "worker")))
    return out


class Sampler(threading.Thread):
    """Background /metrics sampler → timeseries for windowed throughput + duty."""

    def __init__(self, period=0.5):
        super().__init__(daemon=True)
        self.period = period
        self._stop = threading.Event()
        self.ts = []  # (t, running, kv, gen_tokens)

    def run(self):
        while not self._stop.is_set():
            sc = metrics.scrape(timeout=2.0)
            if sc:
                self.ts.append((sc["t"], sc.get("running", 0.0), sc.get("kv", 0.0),
                                sc.get("gen_tokens", 0.0)))
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()

    def window(self, t0, t1, warm=8.0, drain=2.0):
        """Steady-window aggregate tok/s + mean running + peak kv over [t0+warm, t1-drain]."""
        lo, hi = t0 + warm, t1 - drain
        pts = [p for p in self.ts if lo <= p[0] <= hi]
        if len(pts) < 2:  # run too short for a clean window: use whole run
            pts = [p for p in self.ts if t0 <= p[0] <= t1]
        if len(pts) < 2:
            return None
        tok_s = (pts[-1][3] - pts[0][3]) / (pts[-1][0] - pts[0][0]) if pts[-1][0] > pts[0][0] else None
        # time-weighted mean running
        area = span = 0.0
        for (ta, ra, _, _), (tb, rb, _, _) in zip(pts, pts[1:]):
            dt = tb - ta
            area += 0.5 * (ra + rb) * dt
            span += dt
        mean_run = area / span if span else pts[-1][1]
        kv_peak = max(p[2] for p in pts)
        return {"tok_s": tok_s, "mean_running": mean_run, "kv_peak": kv_peak,
                "samples": len(pts)}


def run_gate(pool, N, mult, min_pool, warm_s=12.0):
    pool_n = max(min_pool, mult * N)
    # override to the no-tool SWEEP_LANE for a clean single-generation-per-task measurement
    tasks = [Task(id=f"{tid}-g{N}", prompt=p, lane=SWEEP_LANE) for (tid, p, _lane) in pool[:pool_n]]
    board = Board()
    board.add_many(tasks)

    # co-tenant guard: server-wide gen_tokens is only ≈ fleet throughput if nothing else is
    # decoding. Warn (don't abort) if another client is already running requests.
    pre = metrics.scrape()
    if pre and pre.get("running", 0) > 2:
        print(f"  ⚠ co-tenant: {pre['running']} requests already running — tok/s may be polluted",
              flush=True)

    gate = compat.DecodeGate(N)
    compat.apply(gate)  # install THIS gate (idempotent; updates module-global _GATE)

    sampler = Sampler(0.5)
    sampler.start()
    sc0 = metrics.scrape()
    t0 = time.time()
    out = ThreadFleet(board, gate, cfg=config, on_event=lambda *a, **k: None).run()
    t1 = time.time()
    sc1 = metrics.scrape()
    sampler.stop(); sampler.join(1.0)

    w = sampler.window(t0, t1, warm=warm_s) or {}
    # per-worker timing from the result dicts
    res = out.get("results", {})
    decs = [r["decode_s"] for r in res.values() if r.get("decode_s") is not None]
    tools = [r["tool_s"] for r in res.values() if r.get("tool_s") is not None]
    gws = [r.get("gatewait_s", 0.0) for r in res.values()]
    row = {
        "N": N, "pool": len(tasks), "wall_s": round(t1 - t0, 1),
        "tok_s_windowed": round(w["tok_s"], 0) if w.get("tok_s") else None,
        "tok_s_whole": round((sc1["gen_tokens"] - sc0["gen_tokens"]) / (t1 - t0), 0)
        if sc0 and sc1 else None,
        "mean_running": round(w.get("mean_running", out.get("mean_running", 0)), 2),
        "occupancy": round(w.get("mean_running", 0) / N, 3) if w.get("mean_running") else None,
        "kv_peak_pct": round(w.get("kv_peak", 0) * 100, 1),
        "duty_enrolled": round(out.get("duty"), 3) if out.get("duty") is not None else None,
        "gate_peak_in": out.get("gate_stats", {}).get("peak_in"),
        "decode_s_mean": round(statistics.mean(decs), 2) if decs else None,
        "tool_s_mean": round(statistics.mean(tools), 2) if tools else None,
        "gatewait_s_mean": round(statistics.mean(gws), 2) if gws else None,
        "tok_s_per_agent": round(w["tok_s"] / w["mean_running"], 1)
        if w.get("tok_s") and w.get("mean_running") else None,
        "prefix_hit": metrics.prefix_hit_rate(sc0, sc1),
        "failed": out.get("counts", {}).get("failed", 0),
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="examples/sweep_8k.jsonl")
    ap.add_argument("--gates", default="8,16,32,48,64")
    ap.add_argument("--mult", type=int, default=6, help="pool = max(min_pool, mult*N)")
    ap.add_argument("--min-pool", type=int, default=96)
    ap.add_argument("--warm-s", type=float, default=12.0,
                    help="ramp seconds excluded from the steady throughput window")
    ap.add_argument("--out", default="results/live_sweep.json")
    a = ap.parse_args()

    gates = [int(x) for x in a.gates.split(",")]
    need = max(a.min_pool, a.mult * max(gates))
    pool = load_pool(a.tasks, need)
    assert len(pool) >= need, f"need {need} tasks in {a.tasks}, have {len(pool)} (regen with scripts/gen_tasks.py --n {need})"

    # warm the per-role prefixes + caches once, before the sweep
    compat.apply()  # ensure forwarders patched (no gate yet)
    compat.prewarm(list(config.TOOL_PROFILES.values()))

    print(f"live sweep: gates={gates} pool=max({a.min_pool},{a.mult}*N) tasks/{a.tasks} "
          f"-> {config.BASE_URL}", flush=True)
    rows = []
    for N in gates:
        print(f"\n── gate N={N} (pool {max(a.min_pool, a.mult*N)}) ──", flush=True)
        row = run_gate(pool, N, a.mult, a.min_pool, warm_s=a.warm_s)
        rows.append(row)
        print(f"  N={N:3d} | tok/s(win)={row['tok_s_windowed']} tok/s(whole)={row['tok_s_whole']} | "
              f"running={row['mean_running']} occ={row['occupancy']} | KV={row['kv_peak_pct']}% | "
              f"duty={row['duty_enrolled']} | dec={row['decode_s_mean']}s tool={row['tool_s_mean']}s | "
              f"fail={row['failed']}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"gates": gates, "tasks_file": a.tasks, "rows": rows}, open(a.out, "w"), indent=2)
    print(f"\n=== SWEEP SUMMARY (compare to FLEET_OPTIMUM §4) ===")
    print(f"{'N':>4} {'tok/s win':>10} {'tok/s all':>10} {'mean_run':>9} {'occ':>6} "
          f"{'KV%':>6} {'duty':>6} {'t/s/agt':>8} {'fail':>5}")
    for r in rows:
        print(f"{r['N']:>4} {str(r['tok_s_windowed']):>10} {str(r['tok_s_whole']):>10} "
              f"{r['mean_running']:>9} {str(r['occupancy']):>6} {r['kv_peak_pct']:>6} "
              f"{str(r['duty_enrolled']):>6} {str(r['tok_s_per_agent']):>8} {r['failed']:>5}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
