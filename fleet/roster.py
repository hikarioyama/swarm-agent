"""Print the agent roster and check it against the KV budget — TWO views.

    python -m fleet.roster

WHY two views, and why the new one is the correct one
-----------------------------------------------------
Recon fact #2 (BUILD_SPEC §0): HermesAgent is *stateless* — every turn rebuilds and
resends the ENTIRE transcript through the synchronous completions path, so the server
holds KV **only while a request is actually generating**. A worker sitting between
turns executing a tool holds **zero** server KV. Two consequences reshape the KV model:

  • "parking" is automatic: KV-resident ≈ count of *currently-decoding* requests, NOT
    the count of *enrolled* agents. There is no eviction step and no parked-but-resident
    state to budget for.
  • The thing that bounds resident KV is therefore the **DecodeGate** (compat.DecodeGate,
    config.DECODE_GATE_*), a single fleet-wide cap on concurrent generations ==
    vLLM `num_requests_running`. The roster's `enrolled` count does NOT bound KV.

So this module prints:

  RESIDENT  (correct, new)  resident KV ≈ Σ_lane (concurrent_decoding_lane × per_turn_tokens_lane).
            Because the gate caps the WHOLE fleet's concurrent decoding (not per-lane), the
            true bound is `gate_limit × per_turn_tokens`. But per_turn_tokens is NOT a single
            constant: HermesAgent is stateless and RESENDS the full transcript every turn, so
            the prompt GROWS turn over turn. We therefore show resident KV as a RANGE per gate
            limit:
              low  ≈ gate_limit × (output + per-task unique suffix)   [prefix-cache HITS: the
                     byte-identical system+tools prefix is served from cache, so the marginal
                     resident cost of an extra decode slot is ~just the fresh output + suffix]
              high ≈ gate_limit × max_turn_prompt                      [system + tools + the
                     FULL transcript at MAX_ITERATIONS — the worst, last-turn prompt]
            evaluated across the gate's operating band (config.KNEE_LO..config.KNEE_HI). We also
            show the lane decode-priority waterfall (config.LANE_PRIORITY) — who gets those
            scarce decode slots first — and note the director (1×128K, persistent) holds KV
            only while it actually decodes (duty 0.15).

  ENROLLED  (old, pessimistic)  the v0.1 enrolled×context table, kept for contrast but LABELLED
            "pessimistic (pre-stateless-insight)": it overcounts because tool-executing /
            enrolled-but-idle workers hold no server KV under the stateless model.
"""
from __future__ import annotations

from . import config

# Per-turn resident tokens for a ~8K worker is a RANGE, not a constant, because the
# stateless full-history RESEND grows the prompt every turn (BUILD_SPEC fact #2). We
# model the per-decode-slot cost between two honest bounds (all env-overridable so the
# band sweep can be re-pointed without editing code):
#
#   PER_TURN_LOW  = output + per-task unique suffix, ASSUMING the byte-identical
#                   system+tools prefix is a vLLM prefix-cache HIT (recon: same-role
#                   prefix served from cache after worker #1). This is the marginal
#                   resident cost of adding one more decode slot in the common case.
#   PER_TURN_HIGH = system + tools + FULL transcript at MAX_ITERATIONS — the worst,
#                   last-turn prompt that a decode slot can carry. For the lean ~8K
#                   worker this is ~its whole 8,192-token context window + output.
#
# Source: FLEET_OPTIMUM.md §5 (~8.2k prompt + ~2k output) decomposed into prefix vs.
# growing-suffix. Defaults: ~3,328-tok lean prefix (config note, mostly cache-served),
# ~2,048 output, a small ~256-tok unique per-task suffix; high ≈ 8K ctx + 2K output.
import os


def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


WORKER_OUTPUT_TOKENS = _envi("FLEET_OUTPUT_TOKENS", 2_048)   # tokens generated per turn
WORKER_SUFFIX_TOKENS = _envi("FLEET_SUFFIX_TOKENS", 256)     # unique-per-task non-cached prompt bytes
# low: only the fresh output + unique suffix are new KV (shared prefix is a cache hit).
PER_TURN_LOW = _envi("FLEET_PER_TURN_LOW", WORKER_OUTPUT_TOKENS + WORKER_SUFFIX_TOKENS)
# high: the heaviest worker turn = full ~8K context window resent + this turn's output.
_WORKER_CTX = config.ROSTER.get("worker", {}).get("context", 8_192)
PER_TURN_HIGH = _envi("FLEET_PER_TURN_HIGH", _WORKER_CTX + WORKER_OUTPUT_TOKENS)

# Back-compat: the old single midpoint figure some callers/labels referenced.
PER_TURN_TOKENS = _envi("FLEET_PER_TURN_TOKENS", 10_240)


def _max_role_context() -> tuple[str, int]:
    """Largest per-role context in the roster — the per-turn upper bound if a decode slot
    were occupied by the heaviest-context role (e.g. the 128K director)."""
    role, ctx = max(config.ROSTER.items(), key=lambda kv: kv[1]["context"])
    return role, ctx["context"]


# ─────────────────────────── RESIDENT view (correct) ───────────────────────────

def _resident_view() -> None:
    B = config.KV_BUDGET
    lo, hi = config.KNEE_LO, config.KNEE_HI
    start = config.DECODE_GATE_START
    gmin, gmax = config.DECODE_GATE_MIN, config.DECODE_GATE_MAX
    max_role, max_ctx = _max_role_context()

    print("=" * 96)
    print("RESIDENT KV  (correct — stateless full-history resend: KV held ONLY while decoding)")
    print("=" * 96)
    print("  resident KV ≈ Σ_lane (concurrent_decoding_lane × per_turn_tokens_lane)")
    print("  concurrent_decoding for the WHOLE fleet is bounded by the DecodeGate, NOT by enrolled.")
    print(f"  gate: start={start}  band[KNEE_LO..KNEE_HI]={lo}..{hi}  clamp[MIN..MAX]={gmin}..{gmax}")
    print(f"  KV budget = {B:,} fp8 tokens\n")

    # How much of the budget the gate consumes across its operating band, as a RANGE
    # per gate limit (the stateless full-history resend grows the prompt every turn):
    #   low  = gate_limit × PER_TURN_LOW   (output + unique suffix; prefix-cache HITS)
    #   high = gate_limit × PER_TURN_HIGH  (system+tools+FULL transcript at MAX_ITERATIONS)
    # The truth for any running fleet sits inside [low, high]; both are useful bounds.
    print(f"  worker per-turn band: low={PER_TURN_LOW:,} tok "
          f"(output {WORKER_OUTPUT_TOKENS:,} + suffix {WORKER_SUFFIX_TOKENS:,}, prefix-cache hit)")
    print(f"                        high={PER_TURN_HIGH:,} tok "
          f"(system+tools+full transcript @ MAX_ITERATIONS={config.MAX_ITERATIONS})\n")
    print(f"  {'gate limit':>11}  {'resident KV (low)':>18} {'(%budget)':>10}   "
          f"{'resident KV (high)':>19} {'(%budget)':>10}")
    print(f"  {'(decoding)':>11}  {'@'+f'{PER_TURN_LOW:,}'+'/turn':>18} {'':>10}   "
          f"{'@'+f'{PER_TURN_HIGH:,}'+'/turn':>19} {'':>10}")
    print("  " + "-" * 84)
    rows = [(lo, "KNEE_LO (latency-aware C32)"),
            (hi, "KNEE_HI (throughput-max C64)"),
            (gmax, "DECODE_GATE_MAX (AIMD ceiling)")]
    for limit, label in rows:
        kv_low = limit * PER_TURN_LOW
        kv_high = limit * PER_TURN_HIGH
        print(f"  {limit:>11}  {kv_low:>18,} {kv_low/B*100:>9.0f}%   "
              f"{kv_high:>19,} {kv_high/B*100:>9.0f}%   {label}")
    print("  " + "-" * 84)

    band_lo, band_hi = lo * PER_TURN_LOW, hi * PER_TURN_HIGH
    print(f"\n  Across the gate's whole operating band ({lo}..{hi} concurrent), resident KV ranges")
    print(f"  from {lo}×{PER_TURN_LOW:,}={band_lo:,} (low, cache-hit) to "
          f"{hi}×{PER_TURN_HIGH:,}={band_hi:,} (high, full transcript)")
    print(f"  = {band_lo/B*100:.0f}%..{band_hi/B*100:.0f}% of the {B:,}-token KV budget.")
    headroom_hi = B - band_hi
    if headroom_hi >= 0:
        print(f"  Even at the band's worst corner (KNEE_HI × full transcript) the fleet still leaves")
        print(f"  {headroom_hi:,} tokens ({headroom_hi/B*100:.0f}%) of KV free — KV is NOT the binding")
        print(f"  constraint in the measured C32–C64 region; the gate is.")
    else:
        print(f"  WARNING: the band's worst corner (KNEE_HI × full transcript) is "
              f"{-headroom_hi:,} tokens OVER budget;")
        print(f"  at that corner KV — not the gate — would bind. The realistic (low/cache-hit) "
              f"corner stays well under.")
    # Pessimistic upper bound: heaviest-context role on every decode slot at its full ctx.
    ub_max = gmax * max_ctx
    print(f"  (Hard upper bound: if every one of {gmax} decode slots held the heaviest {max_role} "
          f"{max_ctx:,}-ctx\n   role at full context, that is {ub_max:,} = {ub_max/B*100:.0f}% of budget — and that mix never occurs.)")

    # Lane decode-priority waterfall: who claims the scarce gate slots first.
    print("\n  Lane decode-priority waterfall (config.LANE_PRIORITY — higher served first when a")
    print("  gate permit frees, so reserved roles never starve behind the worker swarm):")
    print(f"    {'lane':<10} {'priority':>8}  {'per-turn (resident while decoding)':<40}")
    print("    " + "-" * 62)
    # map lanes to a representative per-turn token figure (roster context if known, else worker turn)
    ctx_by_role = {role: r["context"] for role, r in config.ROSTER.items()}
    for lane, prio in sorted(config.LANE_PRIORITY.items(), key=lambda kv: -kv[1]):
        # workers/code/research are the ~8K lean lane → per-turn BAND (resend grows the
        # prompt); reserved roles carry their roster context as the while-decoding cost.
        if lane in ("worker", "code", "research"):
            per_turn = f"{PER_TURN_LOW:,}..{PER_TURN_HIGH:,} tok (lean worker band)"
        elif lane in ctx_by_role:
            per_turn = f"{ctx_by_role[lane]:,} tok ctx (role max)"
        else:
            per_turn = "(role not in ROSTER; uses worker turn)"
        print(f"    {lane:<10} {prio:>8}  {per_turn:<40}")

    d = config.ROSTER.get("director")
    if d:
        dctx, dduty = d["context"], d["duty"]
        print(f"\n  director: 1× {dctx:,}-token (128K) persistent agent, duty {dduty}. It is PERSISTENT in")
        print(f"  enrollment but holds KV ONLY while it actually decodes — at duty {dduty} it occupies a")
        print(f"  {config.lane_priority('director')}-priority decode slot ~{dduty*100:.0f}% of the time, costing")
        print(f"  ~{dctx:,} resident tokens ({dctx/config.KV_BUDGET*100:.0f}% of KV) ONLY during those windows,")
        print(f"  not the full {dctx:,} continuously as the enrolled view below would imply.")


# ────────────────────── ENROLLED view (old, pessimistic) ───────────────────────

def _enrolled_rows():
    """The v0.1 enrolled×context model — overcounts under the stateless insight."""
    for role, r in config.ROSTER.items():
        ctx, n, duty = r["context"], r["count"], r["duty"]
        inflight_kv = ctx * n
        enrolled = n if r.get("persistent") else round(n / max(duty, 1e-6))
        enrolled_kv = ctx * enrolled
        yield role, r, ctx, n, duty, enrolled, inflight_kv, enrolled_kv


def _enrolled_view() -> None:
    B = config.KV_BUDGET
    print("\n" + "=" * 96)
    print("ENROLLED KV  — PESSIMISTIC (pre-stateless-insight) — OVERCOUNTS")
    print("=" * 96)
    print("  This is the v0.1 model: resident KV = enrolled × context. It assumes every enrolled")
    print("  (and every tool-executing / parked) agent continuously holds server KV. Under the")
    print("  stateless full-history-resend model (recon fact #2) that is FALSE: tool-executing")
    print("  workers hold NO server KV, so this table overcounts. Kept only for contrast.\n")
    print(f"  {'role':<9} {'context':>8} {'count':>6} {'duty':>5} {'enrolled':>8} "
          f"{'KV in-flight':>13} {'KV enrolled':>12}  note")
    tot_inflight = tot_enrolled = 0
    for role, r, ctx, n, duty, enr, ikv, ekv in _enrolled_rows():
        tot_inflight += ikv
        tot_enrolled += ekv
        print(f"  {role:<9} {ctx:>8,} {n:>6} {duty:>5} {enr:>8} {ikv:>13,} {ekv:>12,}  {r.get('note','')}")
    print("  " + "-" * 110)
    print(f"  {'TOTAL':<9} {'':>8} {'':>6} {'':>5} {'':>8} {tot_inflight:>13,} {tot_enrolled:>12,}")
    print(f"\n  in-flight KV  = {tot_inflight:,}  ({tot_inflight/B*100:.0f}% of budget)")
    print(f"  enrolled  KV  = {tot_enrolled:,}  ({tot_enrolled/B*100:.0f}% of budget)"
          f"  {'OK' if tot_enrolled <= B else 'OVER — but this view is pessimistic; see RESIDENT above'}")
    print("  NOTE: pessimistic — overcounts; tool-executing workers hold no server KV (recon fact #2).")


def main() -> None:
    _resident_view()
    _enrolled_view()
    print()


if __name__ == "__main__":
    main()
