"""Right-pane live swarm dashboard for the split-view TUI.

Pure model + renderer. It consumes the structured events emitted by
``swarm_agent.runner.SwarmRunner`` (``planned`` and ``task`` events) and paints the
swarm's live state: one row per task with a status glyph, an animated progress bar,
the lane, and timing — plus a footer with the decode gate / running / KV / throughput.

It knows NOTHING about curses: it draws through an injected ``add(y, x, text, attr)``
callback and a small named-attribute ``palette``, so the TUI owns all curses plumbing.
The runner already emits everything needed (engine.py's ``on_event`` → dispatch / done /
requeue / fail / deadlock with counts), so this is a thin projection of that stream.
"""
from __future__ import annotations

import unicodedata
from typing import Optional

# task state -> (glyph, palette-key)
_GLYPH = {
    "pending": ("·", "dim"),
    "running": ("▸", "run"),
    "done":    ("✓", "ok"),
    "retry":   ("↻", "warn"),
    "failed":  ("✗", "err"),
    "stranded": ("⨯", "err"),          # D2: dep-deadlocked, never ran
}
_BAR_W = 8


def _cell_width(ch: str) -> int:
    """Display cells one glyph occupies (East-Asian Wide/Fullwidth = 2)."""
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _dwidth(text: str) -> int:
    return sum(_cell_width(ch) for ch in text)


def _fit(text: str, width: int) -> str:
    """Truncate ``text`` so it occupies at most ``width`` display cells."""
    if width <= 0:
        return ""
    out, used = [], 0
    for ch in text:
        cw = _cell_width(ch)
        if used + cw > width:
            break
        out.append(ch)
        used += cw
    return "".join(out)


class SwarmView:
    """Model + renderer for the live swarm pane."""

    def __init__(self) -> None:
        self.goal = ""
        self.order: list[str] = []          # task ids in plan order
        self.tasks: dict[str, dict] = {}
        self.counts: dict = {}
        self.frame = 0
        self.active = False
        self.summary: Optional[dict] = None
        self.stranded = 0                  # D2: count of tasks stranded by a deadlock

    # ── ingest one runner event ──────────────────────────────────────────────
    def ingest(self, ev: dict) -> None:
        kind = ev.get("kind")
        if kind == "planning":
            self.active = True
            self.summary = None
            self.order = []
            self.tasks = {}
            self.counts = {}
            self.stranded = 0          # D2: clear stranded state for the new run
        elif kind == "planned":
            for t in ev.get("tasks", []):
                tid = t["id"]
                if tid not in self.tasks:
                    self.order.append(tid)
                self.tasks[tid] = {
                    "lane": t.get("lane", "worker"), "deps": t.get("deps", []),
                    "prompt": t.get("prompt", ""), "state": "pending",
                    "wall_s": None, "turns": None,
                }
        elif kind == "task":
            self._ingest_task(ev)
        elif kind in ("final", "error"):
            self.active = False
            self.summary = ev.get("stats") or self.summary
        elif kind == "idle":
            self.active = False

    def _ingest_task(self, ev: dict) -> None:
        self.counts = ev.get("counts") or self.counts
        if ev.get("event") == "deadlock":
            # D2: engine emits deadlock with id=None + stranded=N (engine.py:156-157).
            # Mark every non-terminal task stranded and record the count for the footer.
            # MUST run BEFORE the None-id early return below, which would drop it.
            self.stranded = int(ev.get("stranded") or 0)
            for t in self.tasks.values():
                if t["state"] not in ("done", "failed"):
                    t["state"] = "stranded"
            self.active = False
            return
        tid = ev.get("id")
        if not tid or tid == "None":
            return
        if tid not in self.tasks:                          # e.g. the reducer sink
            self.order.append(tid)
            self.tasks[tid] = {"lane": "reducer", "deps": [], "prompt": "",
                               "state": "pending", "wall_s": None, "turns": None}
        t = self.tasks[tid]
        e = ev.get("event")
        if e == "dispatch":
            t["state"] = "running"
        elif e == "done":
            t["state"], t["wall_s"], t["turns"] = "done", ev.get("wall_s"), ev.get("turns")
        elif e == "requeue":
            t["state"] = "retry"
        elif e == "fail":
            t["state"] = "failed"

    # ── render ───────────────────────────────────────────────────────────────
    def render(self, add, geom, *, palette, gate_limit, running, kv_pct, tok_s,
               elapsed=0, busy=False, phase="", queue=None) -> None:
        """Paint the right status panel: WORKERS, TASKS, then a metrics footer.

        Layout top→bottom: title + live/idle badge, a rule, the two stacked sections,
        a rule, and the metrics (gate/run/kv + tok-s/elapsed) pinned to the last two
        rows. Every metric lives here now — the old full-width status rule above the
        composer is gone, so this panel is the single status home.

        ``busy``/``phase`` come from the runner front door. They make the WORKERS list
        ALWAYS show the live state: in swarm mode the running sub-agents, and in
        chat/router/plan mode the single *main* worker (the front-door agent) animated
        while it composes its reply — so the panel is never dead while work is in
        flight, even when no fleet has fanned out.
        """
        top, left, bottom, right = geom
        self.frame += 1
        width = max(4, right - left)

        def put(y, text, key="fg", x=None):
            xx = left if x is None else x
            if top <= y <= bottom and right - xx > 0:
                add(y, xx, _fit(text, right - xx), palette.get(key, 0))

        def rule(y):
            if top <= y <= bottom:
                add(y, left, "─" * width, palette["dim"])

        # title + live/idle badge. "live" tracks the runner (busy), not just the swarm,
        # so the badge lights up for a chat/router/plan reply too — not only fan-out.
        live = self.active or busy
        badge = "● live" if live else "○ idle"
        if bottom >= top:
            add(top, left,
                _fit("swarm" + (f" · {self.goal}" if self.goal else ""),
                     max(0, width - _dwidth(badge) - 1)), palette["gold"])
            add(top, max(left, right - _dwidth(badge)), badge,
                palette["run" if live else "dim"])
        rule(top + 1)

        # metrics footer, pinned to the bottom two rows (+ a rule above them)
        tok = f"{tok_s:.0f}" if tok_s else "–"
        put(bottom, f"{tok} tok/s · {int(elapsed)}s", "dim")
        put(bottom - 1, f"gate {gate_limit} · run {running} · kv {kv_pct}", "dim")
        rule(bottom - 2)
        sec_end = bottom - 3                       # last row the sections may use
        if queue and queue.get("total"):
            p, r = queue.get("pending", 0), queue.get("running", 0)
            d, f = queue.get("done", 0), queue.get("failed", 0)
            text = f"QUEUE · {p} pending · {r} running · {d} done"
            if f:
                text += f" · {f} failed"
            put(sec_end, text, "err" if f else "dim")
            sec_end -= 1

        y = top + 2
        if y > sec_end:                            # panel too short for any section
            return

        # ── WORKERS: the live worker roster, ALWAYS populated. The front-door "main"
        #    worker (router / chat / planner / reducer-synthesis) is ALWAYS row 0 —
        #    animated (★, cylon bar) while the runner is busy, and shown standing by
        #    (☆, idle) otherwise, so the roster is never empty even before a turn or
        #    between turns. During a swarm the running sub-agents follow (active), then
        #    the queued-but-not-yet-dispatched ones (waiting, dimmed) — so you see who
        #    is working AND who is lined up, not just the live few. ──
        def _state(tid):
            return self.tasks.get(tid, {}).get("state")
        running_ids = [tid for tid in self.order if _state(tid) == "running"]
        waiting_ids = [tid for tid in self.order if _state(tid) in ("pending", "ready")]
        # worker row := (tag, label, active) ; main first, then running, then waiting
        workers: list[tuple[str, str, bool]] = [
            ("main", (phase or "working") if busy else "idle", busy)]
        workers += [(self.tasks[t]["lane"][:8], t, True) for t in running_ids]
        workers += [(self.tasks[t]["lane"][:8], t, False) for t in waiting_ids]
        # counts: the standby main worker is neither "active" nor a queued "waiting"
        # sub-agent, so it doesn't inflate either tally — "waiting" means lined-up subs.
        n_active = (1 if busy else 0) + len(running_ids)
        n_wait = len(waiting_ids)
        if sec_end - y + 1 >= 2:
            hdr = f"WORKERS · {n_active} active"
            if n_wait:
                hdr += f" · {n_wait} waiting"
            put(y, hdr, "head" if n_active else "dim")
            y += 1
            budget = max(0, (sec_end - y + 1) - 2)        # leave TASKS header + ≥1 row
            shown = workers[:budget]                       # head-priority: main+running first
            for i, (tag, label, active) in enumerate(shown):
                is_main = tag == "main"
                if active:
                    add(y, left, "★" if is_main else "●",
                        palette["gold"] if is_main else palette["ok"])
                    put(y, f" {tag:<8} {self._bar('running', i, 4)} {label}",
                        "run", x=left + 1)
                else:
                    add(y, left, "☆" if is_main else "○", palette["dim"])
                    put(y, f" {tag:<8} {self._bar('pending', i, 4)} {label}",
                        "dim", x=left + 1)
                y += 1
            hidden = len(workers) - len(shown)
            if hidden > 0 and y <= sec_end:
                put(y, f"  +{hidden} more queued", "dim")
                y += 1
            if (sec_end - y + 1) >= 3:                     # blank gap only if TASKS fits
                y += 1

        # ── TASKS: the full plan; on overflow keep the live/recent tail ─────────
        if y <= sec_end:
            c = self.counts or {}
            done, failed = c.get("done", 0), c.get("failed", 0)
            total = len(self.order) or sum(
                c.get(k, 0) for k in ("pending", "ready", "running", "done", "failed"))
            hdr = f"TASKS · done {done} / {total}"
            if failed:
                hdr += f" · failed {failed}"
            if self.stranded:
                hdr += f" · stranded {self.stranded}"
            put(y, hdr, "err" if (failed or self.stranded) else "head")
            y += 1

            ids = self.order
            budget = sec_end - y + 1
            if budget >= 1 and len(ids) > budget:
                hidden = len(ids) - (budget - 1)           # 1 row spent on the banner
                ids = ids[hidden:]
                put(y, f"  … {hidden} earlier task(s)", "dim")
                y += 1
            tid_w = max(6, min(18, width - 16))
            for tid in ids:
                if y > sec_end:
                    break
                t = self.tasks[tid]
                glyph, key = _GLYPH.get(t["state"], ("·", "dim"))
                meta = (f"  {t['wall_s']}s"
                        if (t["state"] == "done" and t["wall_s"] is not None) else "")
                put(y, f" {glyph} {tid[:tid_w]:<{tid_w}} {t['lane'][:7]:<7}{meta}", key)
                y += 1

    def _bar(self, state: str, row: int, w: int = _BAR_W) -> str:
        if state == "done":
            return "█" * w
        if state == "failed":
            return "─" * w
        if state == "running":                              # cylon marquee, offset per row
            cells = ["░"] * w
            span = max(1, 2 * w - 2)
            pos = (self.frame + row * 2) % span
            if pos >= w:
                pos = span - pos
            cells[pos] = "█"
            return "".join(cells)
        return "·" * w                                      # pending / retry
