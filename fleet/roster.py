"""Print the agent roster and check it against the KV budget.

    python -m fleet.roster

Two KV views:
  in-flight  = count * context        (KV held by agents actively resident)
  enrolled   = (count / duty) * ctx   (workers also need duty-cycle headroom: to keep
               `count` decoding when duty<1 you enroll more; parked agents still hold
               KV unless evicted — see FLEET_OPTIMUM.md §5 'park idle agents out of KV').
"""
from __future__ import annotations
from . import config


def _rows():
    for role, r in config.ROSTER.items():
        ctx, n, duty = r["context"], r["count"], r["duty"]
        inflight_kv = ctx * n
        enrolled = n if r.get("persistent") else round(n / max(duty, 1e-6))
        enrolled_kv = ctx * enrolled
        yield role, r, ctx, n, duty, enrolled, inflight_kv, enrolled_kv


def main() -> None:
    B = config.KV_BUDGET
    print(f"KV budget = {B:,} fp8 tokens\n")
    print(f"{'role':<9} {'context':>8} {'count':>6} {'duty':>5} {'enrolled':>8} "
          f"{'KV in-flight':>13} {'KV enrolled':>12}  note")
    tot_inflight = tot_enrolled = 0
    for role, r, ctx, n, duty, enr, ikv, ekv in _rows():
        tot_inflight += ikv
        tot_enrolled += ekv
        print(f"{role:<9} {ctx:>8,} {n:>6} {duty:>5} {enr:>8} {ikv:>13,} {ekv:>12,}  {r.get('note','')}")
    print("-" * 110)
    print(f"{'TOTAL':<9} {'':>8} {'':>6} {'':>5} {'':>8} {tot_inflight:>13,} {tot_enrolled:>12,}")
    print(f"\n  in-flight KV  = {tot_inflight:,}  ({tot_inflight/B*100:.0f}% of budget)")
    print(f"  enrolled  KV  = {tot_enrolled:,}  ({tot_enrolled/B*100:.0f}% of budget)"
          f"  {'OK' if tot_enrolled <= B else 'OVER — park idle workers out of KV'}")
    director = config.ROSTER["director"]
    print(f"\n  director: {director['context']:,}-token context = "
          f"{director['context']/B*100:.0f}% of KV, count 1, duty {director['duty']} "
          f"(rarely decodes — it steers via the board, not the hot loop).")


if __name__ == "__main__":
    main()
