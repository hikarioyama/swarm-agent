"""Hermes-style split-view TUI for swarm-agent.

Left pane = the conversation (clean user/assistant turns). Right pane = the live swarm
dashboard (per-task progress, lanes, gate/running/kv/throughput). One composer at the
bottom; a plain message is auto-routed (chat vs swarm), or forced with /chat // /swarm.

Architecture (the fix): the runtime is IN-PROCESS via ``SwarmRunner`` — no subprocess,
no merged stdout. The curses loop drains ``runner.events`` (a thread-safe queue) every
tick and projects each event onto the chat log and the swarm view. The fleet engine's
own stdout (HermesAgent import banners, tool fork warnings, vLLM notices) is redirected
to a logfile so it can never corrupt the screen.
"""
from __future__ import annotations

import curses
import os
import queue
import sys
import textwrap
import time
import unicodedata
from pathlib import Path

from fleet import config, metrics
from .runner import SwarmRunner
from .dashboard import SwarmView


LOG_PATH = Path(os.environ.get(
    "SWARM_LOG", str(Path.home() / ".cache" / "swarm-agent" / "runtime.log")))

LOGO = [
    "  ▄▄▄ █  █ █▀█ █▀█ █▀▄▀█   ▄▀█ █▀▀ █▀▀ █▄ █ ▀█▀",
    "  ▄██ ▀▄▀▄▀ █▀█ █▀▄ █ ▀ █   █▀█ █▄█ ██▄ █ ▀█  █ ",
]
HELP = [
    "plain message   router auto-decides: chat reply vs swarm fan-out",
    "/swarm GOAL     force the swarm to decompose GOAL into a parallel DAG",
    "/chat  MSG      force one direct single-agent reply",
    "/clear          clear the conversation",
    "/gate N         set the decode-gate floor (throughput)",
    "/help           show this help",
    "/quit           exit  (Ctrl+D)",
    "no mid-run stop — a turn runs to completion; Ctrl+D ends the session",
]
_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def parse_command(text: str) -> tuple[str, str]:
    text = text.strip()
    if not text.startswith("/"):
        return "message", text
    command, _, argument = text[1:].partition(" ")
    return command.lower(), argument.strip()


def _disp_width(text: str) -> int:
    """Terminal display width: East-Asian Wide/Fullwidth glyphs occupy 2 cells."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


# ── curses helpers ────────────────────────────────────────────────────────────
def _init_colors() -> None:
    if not curses.has_colors():
        return
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_YELLOW, -1)   # gold
    curses.init_pair(2, curses.COLOR_GREEN, -1)    # ok
    curses.init_pair(3, curses.COLOR_RED, -1)      # err
    curses.init_pair(4, curses.COLOR_CYAN, -1)     # run / accent
    curses.init_pair(5, curses.COLOR_WHITE, -1)    # fg


def _pair(n: int, extra: int = 0) -> int:
    return (curses.color_pair(n) if curses.has_colors() else 0) | extra


def _palette() -> dict:
    return {
        "gold": _pair(1, curses.A_BOLD),
        "ok":   _pair(2),
        "err":  _pair(3, curses.A_BOLD),
        "run":  _pair(4),
        "fg":   _pair(5),
        "dim":  _pair(0, curses.A_DIM),
        "warn": _pair(1),
    }


class ChatLog:
    """The left conversation pane: a list of (kind, text) blocks, wrapped on render."""

    _STYLE = {                                  # kind -> (palette-key, first-prefix, cont-prefix)
        "user":      ("gold", "❯ ", "  "),
        "assistant": ("fg",   "  ", "  "),
        "status":    ("dim",  "· ", "  "),
        "error":     ("err",  "✗ ", "  "),
    }

    def __init__(self) -> None:
        self.blocks: list[tuple[str, str]] = []

    def add(self, kind: str, text: str) -> None:
        self.blocks.append((kind, text))

    def clear(self) -> None:
        self.blocks.clear()

    @staticmethod
    def _wrap(text: str, width: int) -> list[str]:
        out: list[str] = []
        for para in text.split("\n"):
            out.extend(textwrap.wrap(para, width) if para.strip() else [""])
        return out or [""]

    def render(self, add, geom, palette) -> None:
        top, left, bottom, right = geom
        width = max(10, right - left)
        lines: list[tuple[str, str]] = []
        for kind, text in self.blocks:
            key, first, cont = self._STYLE.get(kind, ("fg", "  ", "  "))
            for j, seg in enumerate(self._wrap(text, width - 2)):
                lines.append(((first if j == 0 else cont) + seg, key))
            if kind in ("assistant", "error"):
                lines.append(("", "dim"))
        rows = max(1, bottom - top)
        y = top
        for text, key in lines[-rows:]:
            add(y, left, text[:width], palette.get(key, 0))
            y += 1


class App:
    SCRAPE_PERIOD = 0.8

    def __init__(self, stdscr) -> None:
        self.stdscr = stdscr
        self.runner = SwarmRunner()
        self.chat = ChatLog()
        self.swarm = SwarmView()
        self.meter = metrics.ThroughputMeter()
        self.input = ""
        self.show_help = False
        self.message = "type a goal, or /help"
        self.started_at: float | None = None
        self._sc: dict = {}
        self._sc_at = 0.0
        self._tok_s = 0.0
        self._tick = 0
        self.phase = ""
        self._task_total = 0
        self.pending: list = []   # messages queued while a turn is running
        self._log = None
        self._orig_io = None

    # ── output redirect (keep the engine's stdout off the curses screen) ──────
    def _redirect_output(self) -> None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(LOG_PATH, "a", buffering=1)
        self._orig_io = (sys.stdout, sys.stderr)
        sys.stdout = sys.stderr = self._log

    def _restore_output(self) -> None:
        if self._orig_io:
            sys.stdout, sys.stderr = self._orig_io
        if self._log:
            try:
                self._log.close()
            except Exception:
                pass

    # ── event pump: runner.events -> chat + swarm ─────────────────────────────
    def pump(self) -> None:
        while True:
            try:
                ev = self.runner.events.get_nowait()
            except queue.Empty:
                break
            self.swarm.ingest(ev)
            kind = ev.get("kind")
            if kind == "user":
                self.chat.add("user", ev.get("text", ""))
                self.started_at = time.monotonic()
                self.message = "working…"
                self.phase = "thinking"
            elif kind == "boot":
                self.phase = "warming runtime"
            elif kind == "ready":
                # T2: runner finished warm-up; surface it (was silently dropped).
                self.chat.add("status", f"runtime ready · gate {ev.get('gate')}")
                self.phase = ""
            elif kind == "status":
                # R4: front-door status lines (e.g. 'router unsure — routing to the swarm').
                self.chat.add("status", ev.get("text", ""))
            elif kind == "route":
                if ev.get("mode") == "swarm":
                    self.phase = "planning"
            elif kind == "planning":
                self.phase = "planning"
            elif kind == "planned":
                self.swarm.goal = self._last_goal()
                self._task_total = int(ev.get("n") or 0)
                self.chat.add("status", f"planned → {ev.get('n')} parallel tasks")
                self.phase = f"dispatching {ev.get('n')} tasks"
            elif kind == "task":
                c = ev.get("counts") or {}
                done = c.get("done", 0)
                self.phase = (f"running · {done}/{self._task_total} done"
                              if self._task_total else "running")
            elif kind == "reply":
                self.chat.add("assistant", ev.get("text", ""))
                self.phase = ""
            elif kind == "final":
                self.chat.add("assistant", ev.get("text", ""))
                self.chat.add("status", self._summary_line(ev.get("stats")))
                self.phase = ""
            elif kind == "error":
                self.chat.add("error", ev.get("text", "error"))
                self.phase = ""
            elif kind == "idle":
                self.message = "ready"
                self.started_at = None
                self.phase = ""
                self._drain_pending()

    def _last_goal(self) -> str:
        for kind, text in reversed(self.chat.blocks):
            if kind == "user":
                return text[:48]
        return ""

    @staticmethod
    def _summary_line(stats) -> str:
        if not stats:
            return "swarm complete"
        c = stats.get("counts") or {}
        return (f"swarm: {c.get('done', 0)} tasks in {stats.get('wall_s')}s · "
                f"peak {stats.get('peak_running')} in-flight")

    # ── submit / commands ─────────────────────────────────────────────────────
    def submit(self, text: str) -> bool:
        command, argument = parse_command(text)
        self.show_help = False
        if command in {"quit", "exit"}:
            return False
        if not text.strip():
            return True
        if command == "message":
            self._dispatch(argument, None)
        elif command in {"help", "?"}:
            self.show_help = True
        elif command == "swarm":
            self._dispatch(argument, "swarm")
        elif command == "chat":
            self._dispatch(argument, "chat")
        elif command == "clear":
            self.chat.clear()
            self.swarm = SwarmView()
            self.message = "cleared"
        elif command == "gate":
            self._set_gate(argument)
        else:
            self.message = f"unknown command: /{command}"
        return True

    def _dispatch(self, text: str, force_mode) -> None:
        if not text.strip():
            self.message = "nothing to send"
            return
        if self.runner.busy:
            # A turn is running — QUEUE this message instead of dropping it, so the user
            # can fire several in a row; they run in order as each turn finishes.
            self.pending.append((text, force_mode))
            self.message = f"queued · {len(self.pending)} waiting (runs when ready)"
            return
        self.runner.submit(text, force_mode=force_mode)

    def _drain_pending(self) -> None:
        """Submit the next queued message once the runner is free (called on idle)."""
        if self.pending and not self.runner.busy:
            text, force_mode = self.pending.pop(0)
            self.runner.submit(text, force_mode=force_mode)

    def _set_gate(self, argument: str) -> None:
        try:
            n = max(config.DECODE_GATE_MIN, min(config.DECODE_GATE_MAX, int(argument)))
        except ValueError:
            self.message = "usage: /gate N"
            return
        # T4: under AIMD the controller resizes the gate every control interval from live
        # /metrics (admission.py set_limit), so a manual /gate is only a TRANSIENT seed it
        # will move — say so, don't imply it sticks. We seed the live gate for instant effect.
        aimd = getattr(self.runner, "_admission", "aimd") == "aimd"
        if self.runner.gate is not None:
            self.runner.gate.set_limit(n)
            self.message = (f"decode gate → {n} (AIMD will retune toward the knee)"
                            if aimd else f"decode gate set to {n}")
        else:
            self.runner._gate_start = n
            self.message = f"decode gate floor {n} (applied on first run)"

    # ── metrics ────────────────────────────────────────────────────────────────
    def _scrape(self) -> None:
        now = time.monotonic()
        if now - self._sc_at < self.SCRAPE_PERIOD:
            return
        sc = metrics.scrape(config.METRICS_URL, timeout=0.2) or {}
        self._sc, self._sc_at = sc, now
        self._tok_s = self.meter.update(sc) or 0.0

    # ── draw ────────────────────────────────────────────────────────────────────
    def _safe(self, y: int, x: int, text: str, attr: int = 0) -> None:
        h, w = self.stdscr.getmaxyx()
        if 0 <= y < h and 0 <= x < w and text:
            try:
                self.stdscr.addnstr(y, x, text, max(0, w - x - 1), attr)
            except curses.error:
                pass

    def draw(self) -> None:
        self.stdscr.erase()
        h, w = self.stdscr.getmaxyx()
        pal = _palette()
        self._tick += 1
        busy = self.runner.busy
        gate = self.runner.gate.get_limit() if self.runner.gate is not None else config.DECODE_GATE_START
        running = int(self._sc.get("running", 0)) if self._sc else 0
        kv = f"{self._sc.get('kv', 0) * 100:.0f}%" if self._sc else "?"

        # header (1 line)
        self._safe(0, 1, "swarm-agent", pal["gold"])
        self._safe(0, 13, f"· {config.MODEL} · high-concurrency Hermes runtime", pal["dim"])

        body_top, body_bottom = 2, h - 4
        divider = max(28, int(w * 0.58))
        # vertical divider
        for y in range(body_top, body_bottom):
            self._safe(y, divider, "│", pal["dim"])

        # left: conversation (or welcome); reserve the bottom row for a live spinner
        chat_bottom = body_bottom - 1 if (busy and not self.show_help) else body_bottom
        chat_geom = (body_top, 2, chat_bottom, divider - 1)
        if self.show_help:
            y = body_top
            self._safe(y, 2, "commands", pal["gold"]); y += 1
            for line in HELP:
                self._safe(y + 1, 2, line, pal["dim"]); y += 1
        elif not self.chat.blocks:
            self._welcome((body_top, 2, body_bottom, divider - 1), pal)
        else:
            self.chat.render(self._safe, chat_geom, pal)
        if busy and not self.show_help:
            spin = _SPIN[self._tick % len(_SPIN)]
            el = int(time.monotonic() - self.started_at) if self.started_at else 0
            self._safe(body_bottom, 2, f"{spin} {self.phase or 'working'}… {el}s"[:divider - 3],
                       pal["run"])

        # right: live swarm dashboard
        self.swarm.render(self._safe, (body_top, divider + 2, body_bottom, w - 2),
                          palette=pal, gate_limit=gate, running=running,
                          kv_pct=kv, tok_s=self._tok_s)

        # status line + composer
        elapsed = int(time.monotonic() - self.started_at) if self.started_at else 0
        status = f"{self.phase or 'working'}…" if busy else self.message
        rule = (f"─ {status} │ gate {gate} │ running {running} │ kv {kv} │ "
                f"{self._tok_s:.0f} tok/s │ {elapsed}s ")
        self._safe(h - 3, 1, rule + "─" * max(0, w - len(rule) - 2), pal["dim"])
        self._safe(h - 2, 1, "❯ ", pal["gold"])
        if self.input:
            self._safe(h - 2, 3, self.input, pal["fg"])
        else:
            hint = ("working… no mid-run stop · Ctrl+D ends the session"
                    if busy else "Type a goal or /help")
            self._safe(h - 2, 3, hint, pal["dim"])
        self._safe(h - 1, 1,
                   "Enter send · /swarm · /chat · /clear · /help · Ctrl+D quit", pal["dim"])
        try:
            curses.curs_set(1)
            # T3: position the cursor by DISPLAY width, not len(); CJK glyphs are 2 cells.
            self.stdscr.move(h - 2, min(w - 2, 3 + _disp_width(self.input)))
        except curses.error:
            pass
        self.stdscr.refresh()

    def _welcome(self, geom, pal) -> None:
        top, left, bottom, right = geom
        width = right - left
        y = top + 1
        if width >= max(len(s) for s in LOGO) and bottom - top > 8:
            for line in LOGO:
                self._safe(y, left, line, pal["gold"]); y += 1
            y += 1
        self._safe(y, left, "起動即スワーム — type a goal and the planner fans it", pal["dim"])
        self._safe(y + 1, left, "out across dozens of agents; small talk gets a", pal["dim"])
        self._safe(y + 2, left, "direct reply.  /help for commands.", pal["dim"])

    # ── main loop ────────────────────────────────────────────────────────────────
    def run(self) -> int:
        # T1: redirect FIRST, then everything else inside the try whose finally restores
        # stdio + closes the logfile. The old code did _init_colors()/curs_set BEFORE the
        # try, so a curses.error there leaked self._log + the swapped sys.stdout/err.
        self._redirect_output()
        try:
            try:
                _init_colors()
            except curses.error:
                pass                                # colors are non-essential
            try:
                curses.curs_set(1)
            except curses.error:
                pass
            self.stdscr.nodelay(True)
            self.stdscr.timeout(100)
            while True:
                self.pump()
                self._scrape()
                self.draw()
                try:
                    key = self.stdscr.get_wch()
                except KeyboardInterrupt:
                    self.message = "use Ctrl+D to quit"
                    continue
                except curses.error:
                    continue
                if key == "\x04":                       # Ctrl+D
                    return 0
                if key == "\x1b":                       # Esc
                    self.input = ""
                    self.show_help = False
                elif key in (curses.KEY_ENTER, "\n", "\r"):
                    text, self.input = self.input, ""
                    if not self.submit(text):
                        return 0
                elif key in (curses.KEY_BACKSPACE, "\b", "\x7f"):
                    self.input = self.input[:-1]
                elif key == "\x15":                      # Ctrl+U
                    self.input = ""
                elif isinstance(key, str) and key.isprintable():
                    self.input += key
        finally:
            self.runner.shutdown()
            self._restore_output()


def main() -> int:
    return curses.wrapper(lambda stdscr: App(stdscr).run())


if __name__ == "__main__":
    raise SystemExit(main())
