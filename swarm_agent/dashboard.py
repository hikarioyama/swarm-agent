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
    def render(self, add, geom, *, palette, gate_limit, running, kv_pct, tok_s) -> None:
        top, left, bottom, right = geom
        self.frame += 1
        width = max(4, right - left)

        title = "swarm" + (f" · {self.goal}" if self.goal else "")
        add(top, left, title[:width], palette["gold"])
        sub = ("● live" if self.active else "○ idle")
        add(top, max(left, right - len(sub)), sub, palette["run" if self.active else "dim"])

        y = top + 2
        foot_h = 3
        maxrows = max(0, bottom - y - foot_h)
        ids = self.order
        more = 0
        # D1: only truncate/render the banner when there is ≥1 task row to show. At
        # maxrows==0 the old code computed more=len(ids)+1 and drew the banner at the
        # footer row, corrupting it and over-counting by one. The banner costs a row, so
        # hide (maxrows-1) rows behind it and report EXACTLY that many.
        if maxrows >= 1 and len(ids) > maxrows:
            hidden = len(ids) - (maxrows - 1)        # rows replaced by the 1-row banner
            ids = ids[hidden:]
            more = hidden
        if more:
            add(y, left, f"  … {more} earlier task(s)"[:width], palette["dim"])
            y += 1
        for i, tid in enumerate(ids):
            t = self.tasks[tid]
            active = t["state"] == "running"
            glyph, key = _GLYPH.get(t["state"], ("·", "dim"))
            # active indicator: GREEN ● while the worker is generating, WHITE ○ otherwise.
            add(y, left, "●" if active else "○", palette["ok"] if active else palette["dim"])
            bar = self._bar(t["state"], i)
            meta = f" {t['wall_s']}s" if (t["state"] == "done" and t["wall_s"] is not None) else ""
            row = f" {glyph} {bar} {tid[:16]:<16} {t['lane'][:7]}{meta}"
            add(y, left + 1, row[: max(0, width - 1)], palette.get(key, 0))
            y += 1

        c = self.counts or {}
        done, failed = c.get("done", 0), c.get("failed", 0)
        active_n = sum(1 for t in self.tasks.values() if t["state"] == "running")
        total = len(self.order) or sum(c.get(k, 0) for k in ("pending", "ready", "running", "done", "failed"))
        bar = bottom - 2
        if bar >= top:                                  # D3: only draw if the footer fits
            line = f"● {active_n} active · done {done}/{total}"
            if failed:
                line += f" · failed {failed}"
            if self.stranded:                           # D2: surface stranded count
                line += f" · stranded {self.stranded}"
            add(bar, left, line[:width],                # D4: width-clamp
                palette["err"] if (failed or self.stranded) else palette["dim"])
        tok = f"{tok_s:.0f}" if tok_s else "–"
        if bottom - 1 >= top:                            # D3: skip if pane too short
            add(bottom - 1, left,
                f"gate {gate_limit} · run {running} · kv {kv_pct} · {tok} tok/s"[:width],
                palette["dim"])

    def _bar(self, state: str, row: int) -> str:
        if state == "done":
            return "█" * _BAR_W
        if state == "failed":
            return "─" * _BAR_W
        if state == "running":                              # cylon marquee, offset per row
            cells = ["░"] * _BAR_W
            span = 2 * _BAR_W - 2
            pos = (self.frame + row * 2) % span
            if pos >= _BAR_W:
                pos = span - pos
            cells[pos] = "█"
            return "".join(cells)
        return "·" * _BAR_W                                 # pending / retry
