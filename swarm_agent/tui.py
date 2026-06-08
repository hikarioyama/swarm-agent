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
import time
import unicodedata
from pathlib import Path

from fleet import config, metrics
from .runner import SwarmRunner
from .dashboard import SwarmView, MultiSwarmView


LOG_PATH = Path(os.environ.get(
    "SWARM_LOG", str(Path.home() / ".cache" / "swarm-agent" / "runtime.log")))

# ASCII-only banner (figlet "small"). Box-drawing / block glyphs are East-Asian
# "Ambiguous" width, so a CJK-locale terminal renders them 2 cells wide and the
# logo shears apart; plain ASCII is Narrow everywhere and stays aligned.
LOGO = [
    '                                              _',
    ' ____ __ ____ _ _ _ _ __    __ _ __ _ ___ _ _| |_',
    "(_-< V  V / _` | '_| '  \\  / _` / _` / -_) ' \\  _|",
    '/__/\\_/\\_/\\__,_|_| |_|_|_| \\__,_\\__, \\___|_||_\\__|',
    '                                |___/',
]
HELP = [
    "plain message   router auto-decides: chat reply vs swarm fan-out",
    "plain (mid-run) INTERJECT into the running turn — the model sees it next iteration",
    "/stop           interrupt the running turn (stops in-flight generation/tools)",
    "/swarm GOAL     force the swarm to decompose GOAL into a parallel DAG (queues if busy)",
    "/chat  MSG      force one direct single-agent reply (queues if busy)",
    "/task GOAL      queue a goal; the completion manager runs it to done",
    "/tasks          list the persistent task queue",
    "/btw <q>        ask an INDEPENDENT worker about the current situation (works mid-run)",
    "mouse drag      select chat text and copy it (the status panel is left alone)",
    "/copy [last]    also copy the conversation (or last reply) to the clipboard",
    "Ctrl+Y          copy the last reply to the clipboard",
    "←→ Ctrl+A/E     move the caret in the composer (Del/Backspace edit at the caret)",
    "↑↓ PgUp PgDn    scroll the conversation (Home/End jump to oldest/newest)",
    "/clear          clear the conversation",
    "/gate N         set the decode-gate floor (throughput)",
    "/help           show this help",
    "/quit           exit  (Ctrl+D)",
]
_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class _EOFStdin:
    """A stdin replacement that is always at EOF.

    Any tool that falls back to input()/sys.stdin.readline() on a worker thread
    (approval prompts, clarify, oauth) gets an immediate EOF instead of fighting
    curses for the real terminal or blocking on the 60s approval-join timeout.
    """
    def read(self, *a): return ""
    def readline(self, *a): return ""
    def readlines(self, *a): return []
    def __iter__(self): return iter(())
    def isatty(self): return False
    def fileno(self): raise OSError("stdin disabled in swarm TUI")
    def close(self): pass


def parse_command(text: str) -> tuple[str, str]:
    text = text.strip()
    if not text.startswith("/"):
        return "message", text
    command, _, argument = text[1:].partition(" ")
    return command.lower(), argument.strip()


def _copy_to_clipboard(text: str) -> str:
    """Put ``text`` on the system clipboard; return the method used ("" on failure).

    Tries Wayland (wl-copy — the user's Hyprland setup), then X11 (xclip/xsel), then an
    OSC 52 escape written to /dev/tty (works over SSH and in terminals that support it,
    where no clipboard CLI is reachable). The subprocess stdin never touches the curses
    screen; OSC 52 is a non-rendering control string so it does not corrupt the display.
    """
    import shutil
    import subprocess
    for tool, cmd in (("wl-copy", ["wl-copy"]),
                      ("xclip", ["xclip", "-selection", "clipboard"]),
                      ("xsel", ["xsel", "-ib"])):
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, input=text.encode(), timeout=2, check=True)
                return tool
            except Exception:
                pass
    try:
        import base64
        seq = "\033]52;c;" + base64.b64encode(text.encode()).decode() + "\a"
        with open("/dev/tty", "w") as tty:
            tty.write(seq)
            tty.flush()
        return "osc52"
    except Exception:
        return ""


def _chat_plaintext(blocks, last_only: bool = False) -> str:
    """Serialise CHAT blocks to plain text (status-panel content is never here).

    ``last_only`` → just the most recent assistant reply. Otherwise the whole
    conversation: user turns prefixed ``> ``, assistant/error verbatim; transient
    ``status`` lines (UI chrome) are dropped so a copy is clean conversation text."""
    if last_only:
        for kind, text in reversed(blocks):
            if kind == "assistant":
                return text
        return ""
    out: list[str] = []
    for kind, text in blocks:
        if kind == "user":
            out.append(f"> {text}")
        elif kind in ("assistant", "error"):
            out.append(text)
    return "\n\n".join(out)


def _disp_width(text: str) -> int:
    """Terminal display width: East-Asian Wide/Fullwidth glyphs occupy 2 cells."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _fit(text: str, width: int) -> str:
    """Truncate ``text`` to at most ``width`` display cells (CJK-safe)."""
    if width <= 0:
        return ""
    out, used = [], 0
    for ch in text:
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if used + cw > width:
            break
        out.append(ch)
        used += cw
    return "".join(out)


def _hard_break(token: str, width: int) -> list[str]:
    """Split one space-less token into pieces of at most ``width`` cells."""
    pieces, cur, cur_w = [], "", 0
    for ch in token:
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if cur and cur_w + cw > width:
            pieces.append(cur)
            cur, cur_w = "", 0
        cur += ch
        cur_w += cw
    if cur:
        pieces.append(cur)
    return pieces or [""]


def _cursor_rowcol(text: str, cursor: int, width: int) -> tuple[int, int]:
    """Map a caret char-index into ``(row, col_cells)`` for ``_hard_break`` wrapping.

    Replays the same cell-width split so the on-screen caret lands exactly where the
    next typed glyph will appear — including the trailing empty line that ``draw``
    appends when the last wrapped line is full.
    """
    cursor = max(0, min(cursor, len(text)))
    row, col = 0, 0
    for i, ch in enumerate(text):
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if col and col + cw > width:        # piece boundary (mirrors _hard_break)
            row, col = row + 1, 0
        if i == cursor:
            return row, col
        col += cw
    if col and col >= width:                # caret past a full last line → fresh row
        return row + 1, 0
    return row, col


def _wrap_cells(text: str, width: int) -> list[str]:
    """Word-wrap ``text`` to at most ``width`` DISPLAY cells per line.

    Unlike ``textwrap`` (which counts characters), this measures East-Asian width,
    so Japanese/CJK lines — 2 cells per glyph and usually space-less — wrap at the
    pane edge instead of bleeding across the divider into the right panel. Over-long
    space-less runs (CJK, URLs) are hard-broken by cell count.
    """
    width = max(1, width)
    lines: list[str] = []
    for para in text.split("\n"):
        if not para.strip():
            lines.append("")
            continue
        cur, cur_w = "", 0
        for word in para.split():
            ww = _disp_width(word)
            if ww > width:                      # token wider than a whole line
                if cur:
                    lines.append(cur)
                    cur, cur_w = "", 0
                pieces = _hard_break(word, width)
                lines.extend(pieces[:-1])
                cur, cur_w = pieces[-1], _disp_width(pieces[-1])
            elif not cur:
                cur, cur_w = word, ww
            elif cur_w + 1 + ww <= width:
                cur += " " + word
                cur_w += 1 + ww
            else:
                lines.append(cur)
                cur, cur_w = word, ww
        if cur or not lines:
            lines.append(cur)
    return lines or [""]


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
        "head": _pair(5, curses.A_BOLD),   # section headers in the right panel
    }


class ChatLog:
    """The left conversation pane: a list of (kind, text) blocks, wrapped on render."""

    _STYLE = {                                  # kind -> (palette-key, first-prefix, cont-prefix)
        "user":      ("gold", "❯ ", "  "),
        "assistant": ("fg",   "  ", "  "),
        "steer":     ("run",  "↪ ", "  "),
        "btw":       ("run",  "btw › ", "      "),
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
        return _wrap_cells(text, width)

    def _all_lines(self, width: int) -> list[tuple[str, str]]:
        """Wrap every block into the full ordered list of (text, palette-key) lines."""
        lines: list[tuple[str, str]] = []
        for kind, text in self.blocks:
            key, first, cont = self._STYLE.get(kind, ("fg", "  ", "  "))
            for j, seg in enumerate(self._wrap(text, width - 2)):
                lines.append(((first if j == 0 else cont) + seg, key))
            if kind in ("assistant", "btw", "error"):
                lines.append(("", "dim"))
        return lines

    def render(self, add, geom, palette, scroll: int = 0) -> int:
        """Paint the conversation; ``scroll`` = lines held back from the bottom.

        Returns the maximum valid scroll (so the caller can clamp its own state). With
        ``scroll == 0`` the newest line is pinned to the bottom (terminal-style follow);
        a positive ``scroll`` reveals earlier lines — how the user reads a reply whose
        top scrolled off. ``scroll`` is clamped here so it can never blank the pane.
        """
        top, left, bottom, right = geom
        width = max(10, right - left)
        lines = self._all_lines(width)
        rows = max(1, bottom - top + 1)
        max_scroll = max(0, len(lines) - rows)
        scroll = max(0, min(scroll, max_scroll))
        end = len(lines) - scroll
        start = max(0, end - rows)
        y = top
        for text, key in lines[start:end]:
            add(y, left, _fit(text, width), palette.get(key, 0))
            y += 1
        return max_scroll


class App:
    SCRAPE_PERIOD = 0.8

    def __init__(self, stdscr) -> None:
        self.stdscr = stdscr
        self.runner = SwarmRunner()
        self.chat = ChatLog()
        self.swarm = MultiSwarmView()
        self.meter = metrics.ThroughputMeter()
        self.input = ""
        self.cursor = 0           # caret position within self.input (0..len)
        self.show_help = False
        self.message = "type a goal, or /help"
        self.started_at: float | None = None
        self._sc: dict = {}
        self._sc_at = 0.0
        self._tok_s = 0.0
        self._tick = 0
        self.phase = ""
        self._task_total = 0
        self.scroll = 0           # conversation scroll: lines held back from the bottom
        self._max_scroll = 0      # last render's clamp ceiling (updated every draw)
        self.pending: list = []   # messages queued while a turn is running
        self._log = None
        self._orig_io = None

    # ── output redirect (keep the engine's stdout off the curses screen) ──────
    def _redirect_output(self) -> None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(LOG_PATH, "a", buffering=1)
        self._orig_io = (sys.stdout, sys.stderr, sys.stdin)
        sys.stdout = sys.stderr = self._log
        sys.stdin = _EOFStdin()

    def _restore_output(self) -> None:
        if self._orig_io:
            sys.stdout, sys.stderr, sys.stdin = self._orig_io
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
                self.scroll = 0          # a new turn snaps the view back to the live tail
            elif kind == "boot":
                self.phase = "warming runtime"
            elif kind == "ready":
                # T2: runner finished warm-up; surface it (was silently dropped).
                self.chat.add("status", f"runtime ready · gate {ev.get('gate')}")
                self.phase = ""
            elif kind == "status":
                # R4: front-door status lines (e.g. 'router unsure — routing to the swarm').
                self.chat.add("status", ev.get("text", ""))
            elif kind == "queued":
                q = ev.get("queue") or {}
                self.chat.add("status", f"queued → task added ({q.get('pending', 0)} pending)")
            elif kind == "manager":
                self.chat.add("status", f"manager · {ev.get('text', '')}")
            elif kind == "route":
                if ev.get("mode") == "swarm":
                    self.phase = "planning"
            elif kind == "planning":
                self.phase = "planning"
            elif kind == "planned":
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
            elif kind == "steer":
                # A typed message delivered INTO the running turn — show it inline so the
                # user sees what they injected and that it landed.
                self.chat.add("steer", ev.get("text", ""))
                self.scroll = 0
            elif kind == "interrupt":
                self.chat.add("status", f"⛔ interrupt sent to {ev.get('reached', 0)} agent(s)")
            elif kind == "btw":
                self.chat.add("btw", ev.get("text", ""))
            elif kind == "final":
                gid = ev.get("goal_id")
                if gid and len(self.swarm.active_views()) > 1:
                    label = self.swarm.goal_label(gid) or gid
                    self.chat.add("status", f"▸ {label[:40]}")
                self.chat.add("assistant", ev.get("text", ""))
                self.chat.add("status", self._summary_line(ev.get("stats")))
                self.phase = ""
            elif kind == "error":
                # Tag which concurrent goal failed (same header as 'final'), so a failure
                # among several parallel goals is attributable without opening /tasks.
                gid = ev.get("goal_id")
                if gid and len(self.swarm.active_views()) > 1:
                    label = self.swarm.goal_label(gid) or gid
                    self.chat.add("status", f"▸ {label[:40]}")
                self.chat.add("error", ev.get("text", "error"))
                self.phase = ""
            elif kind == "idle":
                # One turn ended — but with parallel goals OTHERS may still be in flight, and
                # idle is emitted per-turn (carries goal_id). Only reset the GLOBAL status
                # chrome (ready / elapsed / phase) when nothing is left running, so a finished
                # goal doesn't make the UI report "ready" while peers keep working.
                if not self.runner.busy:
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
        elif command == "task":
            if not argument:
                self.message = "usage: /task GOAL"
            else:
                self.runner.enqueue_task(argument)
                self.message = (
                    f"queued · {self.runner.tasks.counts()['pending']} pending in task queue")
        elif command == "tasks":
            snapshot = self.runner.tasks.snapshot()
            if not snapshot:
                self.chat.add("status", "task queue is empty")
            else:
                q = self.runner.tasks.counts()
                self.chat.add("status", (
                    f"task queue · {q['pending']} pending · {q['running']} running · "
                    f"{q['done']} done · {q['failed']} failed"))
                for rec in snapshot:
                    self.chat.add(
                        "status", f"[{rec['state']}] {rec['id']} · {rec['goal'][:60]}")
        elif command in {"stop", "interrupt"}:
            if self.runner.busy:
                n = self.runner.interrupt()
                self.message = f"⛔ interrupting {n} running agent(s)…"
            else:
                self.message = "nothing is running"
        elif command == "btw":
            if not argument:
                self.message = "usage: /btw <question about the current situation>"
            else:
                self.runner.ask_status(argument, self._situation_snapshot())
                self.message = "asking an independent worker… (answer appears below)"
        elif command == "clear":
            self.chat.clear()
            self.swarm = MultiSwarmView()
            self.scroll = 0
            self.message = "cleared"
        elif command == "gate":
            self._set_gate(argument)
        elif command == "copy":
            # Copy the CHAT pane only (the live status panel is never copied). No arg /
            # "all" → whole conversation; "last"/"answer"/"reply" → just the last reply.
            self._copy(last_only=argument.strip().lower() in {"last", "answer", "reply"})
        else:
            self.message = f"unknown command: /{command}"
        return True

    def _dispatch(self, text: str, force_mode) -> None:
        if not text.strip():
            self.message = "nothing to send"
            return
        if self.runner.busy:
            if force_mode is None:
                # Plain message while a turn is running → INTERJECT it into the live work
                # (HermesAgent /steer): the model sees it on its next tool iteration without
                # losing the work in progress. Explicit /swarm or /chat still queue below.
                n = self.runner.steer(text)
                self.message = (f"↪ interjected into {n} running agent(s)" if n else
                                "↪ interjected — will reach the next agent that starts")
                return
            # Explicit /swarm or /chat — the user wants a NEW turn; queue it to run in order
            # as each turn finishes (use /stop to interrupt the current one instead).
            self.pending.append((text, force_mode))
            self.message = f"queued · {len(self.pending)} waiting (runs when ready)"
            return
        self.runner.submit(text, force_mode=force_mode)

    def _drain_pending(self) -> None:
        """Submit the next queued message once the runner is free (called on idle)."""
        if self.pending and not self.runner.busy:
            text, force_mode = self.pending.pop(0)
            self.runner.submit(text, force_mode=force_mode)

    def _situation_snapshot(self) -> str:
        """A concise plain-text summary of what the swarm is doing RIGHT NOW, for /btw.
        Pulls only from what the App already holds — no extra model/IO calls. Summarises
        EVERY active goal (parallel consumption), not just one."""
        L: list[str] = []
        L.append(f"Runner: {'BUSY' if self.runner.busy else 'idle'}"
                 + (f' · phase: {self.phase}' if self.phase else ''))
        active = self.swarm.active_views()
        if active:
            L.append(f"Active swarms: {len(active)}")
            for key, sv in active:
                if not sv.order and not sv.goal:
                    continue
                c = sv.counts or {}
                head = sv.goal or ("(typed turn)" if key == "_interactive" else key)
                L.append(f"• {head} — {len(sv.order)} tasks: "
                         f"done {c.get('done',0)}, running {c.get('running',0)}, "
                         f"failed {c.get('failed',0)}")
                for tid in sv.order[:8]:
                    t = sv.tasks.get(tid, {})
                    L.append(f"    [{t.get('state','?')}] {tid} ({t.get('lane','?')}): "
                             f"{(t.get('prompt') or '')[:60]}")
        q = self.runner.tasks.counts()
        if q.get("total"):
            L.append(f"Persistent task queue: pending {q['pending']}, running {q['running']}, "
                     f"done {q['done']}, failed {q['failed']}")
            for r in self.runner.tasks.snapshot():
                if r["state"] in ("pending", "running"):
                    L.append(f"  [{r['state']}] {r['goal'][:60]}")
        if self._sc:
            gate = self.runner.gate.get_limit() if self.runner.gate is not None else "?"
            L.append(f"Metrics: gate {gate} · running {self._sc.get('running','?')} · "
                     f"kv {self._sc.get('kv','?')} · {self._tok_s:.0f} tok/s")
        hist = self.runner.history[-4:]
        if hist:
            L.append("Recent conversation:")
            for role, text in hist:
                L.append(f"  {role}: {text[:80]}")
        return "\n".join(L)

    # ── clipboard: copy the CHAT pane (never the status panel) ───────────────────
    def _copy(self, last_only: bool) -> None:
        text = _chat_plaintext(self.chat.blocks, last_only=last_only)
        if not text.strip():
            self.message = "nothing to copy yet"
            return
        method = _copy_to_clipboard(text)
        what = "last reply" if last_only else "conversation"
        self.message = (f"copied {what} to clipboard ({method})" if method
                        else "no clipboard tool — install wl-clipboard or xclip")

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
        comp_w = max(1, w - 4)
        input_lines = _hard_break(self.input, comp_w)
        if self.input and _disp_width(input_lines[-1]) >= comp_w:
            input_lines.append("")
        max_comp_rows = max(1, min(8, h - 6))
        comp_lines = input_lines[-max_comp_rows:]
        n_comp = len(comp_lines)
        pal = _palette()
        self._tick += 1
        busy = self.runner.busy
        gate = self.runner.gate.get_limit() if self.runner.gate is not None else config.DECODE_GATE_START
        running = int(self._sc.get("running", 0)) if self._sc else 0
        kv = f"{self._sc.get('kv', 0) * 100:.0f}%" if self._sc else "?"
        elapsed = int(time.monotonic() - self.started_at) if self.started_at else 0

        # header (1 line)
        self._safe(0, 1, "swarm-agent", pal["gold"])
        self._safe(0, 13, f"· {config.MODEL} · high-concurrency Hermes runtime", pal["dim"])

        body_top, body_bottom = 2, h - 3 - n_comp
        # right pane = the status panel, pinned to the right 1/3 of the screen.
        divider = max(28, (w * 2) // 3)
        # vertical divider (full panel height, incl. the bottom metrics row)
        for y in range(body_top, body_bottom + 1):
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
            self._max_scroll = self.chat.render(self._safe, chat_geom, pal, self.scroll)
            if self.scroll > self._max_scroll:
                self.scroll = self._max_scroll
        if busy and not self.show_help:
            spin = _SPIN[self._tick % len(_SPIN)]
            el = int(time.monotonic() - self.started_at) if self.started_at else 0
            self._safe(body_bottom, 2, f"{spin} {self.phase or 'working'}… {el}s"[:divider - 3],
                       pal["run"])

        # right: live swarm dashboard (now the single status panel). Pass the runner's
        # busy/phase so the WORKERS list shows the front-door "main" worker animating
        # even for a chat reply that spawns no fleet.
        self.swarm.render(self._safe, (body_top, divider + 2, body_bottom, w - 2),
                          palette=pal, gate_limit=gate, running=running,
                          kv_pct=kv, tok_s=self._tok_s, elapsed=elapsed,
                          busy=busy, phase=self.phase, queue=self.runner.tasks.counts())

        # composer — a plain rule frames it; ALL status now lives in the right panel.
        rule_y = h - 2 - n_comp
        comp_top = h - 1 - n_comp
        self._safe(rule_y, 1, "─" * max(0, w - 2), pal["dim"])
        self._safe(comp_top, 1, "❯ ", pal["gold"])
        if self.input:
            for i, seg in enumerate(comp_lines):
                self._safe(comp_top + i, 3, _fit(seg, comp_w), pal["fg"])
        elif self.scroll > 0:
            # Scrolled up to read history — say so, and how to get back to the live tail.
            self._safe(comp_top, 3, _fit(
                f"⇡ scrolled {self.scroll} line(s) up · ↓/End to follow the latest",
                comp_w), pal["warn"])
        else:
            # Keep transient command feedback (queued / cleared / gate …) visible here,
            # since the status rule that used to show self.message is gone.
            hint = ("working… type to interject · /stop to interrupt · Ctrl+D quits"
                    if busy else (self.message or "Type a goal or /help"))
            self._safe(comp_top, 3, _fit(hint, comp_w), pal["dim"])
        self._safe(h - 1, 1,
                   "Enter send · type mid-run to interject · /stop interrupt · ↑↓ scroll · /help",
                   pal["dim"])
        try:
            curses.curs_set(1)
            # T3: position the caret by DISPLAY width, not len(); CJK glyphs are 2 cells.
            # Map the caret's char-index to a wrapped (row, col), then into the visible
            # window (input_lines is tail-clipped to comp_lines), so ←/→ track on screen.
            cur_row, cur_col = _cursor_rowcol(self.input, self.cursor, comp_w)
            vis_row = cur_row - (len(input_lines) - n_comp)
            cur_y = comp_top + max(0, min(n_comp - 1, vis_row))
            cur_x = min(w - 2, 3 + cur_col)
            self.stdscr.move(cur_y, cur_x)
        except curses.error:
            pass
        self.stdscr.refresh()

    # ── scrolling ──────────────────────────────────────────────────────────────
    def _page(self) -> int:
        """Lines to move per PageUp/PageDown (≈ one chat-pane height)."""
        h, _ = self.stdscr.getmaxyx()
        return max(1, (h - 4) - 2 - 1)

    def _scroll_by(self, delta: int) -> None:
        """delta>0 = older (up), delta<0 = newer (down). Clamped to the live tail."""
        self.scroll = max(0, min(self._max_scroll, self.scroll + delta))

    def _welcome(self, geom, pal) -> None:
        top, left, bottom, right = geom
        width = right - left
        y = top + 1
        if width >= max(len(s) for s in LOGO) and bottom - top > 9:
            for line in LOGO:
                self._safe(y, left, line, pal["gold"]); y += 1
            y += 1
        self._safe(y, left, "Swarm on startup - type a goal and the planner fans it", pal["dim"])
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
            # Leave the mouse to the TERMINAL — do NOT enable curses mouse reporting.
            # With reporting OFF, click-drag works natively, so the user can select and
            # COPY chat text with the mouse the normal way. (Capturing the mouse for
            # wheel-scroll would suppress that native selection.) Scroll with the keyboard
            # instead: ↑↓ / PgUp / PgDn / Home / End; the view also follows the live tail.
            try:
                curses.mousemask(0)
            except (curses.error, AttributeError):
                pass
            self.stdscr.nodelay(True)
            self.stdscr.timeout(100)
            # Only NOW (inside a real TUI session) start the completion manager, so it
            # can resume any goals persisted from a previous session and drive the queue.
            self.runner.start_manager()
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
                    self.input, self.cursor = "", 0
                    self.show_help = False
                elif key in (curses.KEY_ENTER, "\n", "\r"):
                    text, self.input, self.cursor = self.input, "", 0
                    if not self.submit(text):
                        return 0
                elif key in (curses.KEY_BACKSPACE, "\b", "\x7f"):
                    if self.cursor > 0:                  # delete the glyph before the caret
                        self.input = self.input[:self.cursor - 1] + self.input[self.cursor:]
                        self.cursor -= 1
                elif key == curses.KEY_DC:               # Delete → glyph at the caret
                    self.input = self.input[:self.cursor] + self.input[self.cursor + 1:]
                elif key == "\x15":                      # Ctrl+U
                    self.input, self.cursor = "", 0
                elif key == curses.KEY_LEFT:             # caret: one glyph left
                    self.cursor = max(0, self.cursor - 1)
                elif key == curses.KEY_RIGHT:            # caret: one glyph right
                    self.cursor = min(len(self.input), self.cursor + 1)
                elif key == "\x01":                      # Ctrl+A → start of input
                    self.cursor = 0
                elif key == "\x05":                      # Ctrl+E → end of input
                    self.cursor = len(self.input)
                elif key == "\x19":                      # Ctrl+Y → copy last reply
                    self._copy(last_only=True)
                elif key == curses.KEY_UP:               # scroll conversation: older
                    self._scroll_by(1)
                elif key == curses.KEY_DOWN:             # scroll: newer / follow
                    self._scroll_by(-1)
                elif key == curses.KEY_PPAGE:            # PageUp
                    self._scroll_by(self._page())
                elif key == curses.KEY_NPAGE:            # PageDown
                    self._scroll_by(-self._page())
                elif key == curses.KEY_HOME:             # jump to the oldest line
                    self.scroll = self._max_scroll
                elif key == curses.KEY_END:              # jump back to the live tail
                    self.scroll = 0
                elif isinstance(key, str) and key.isprintable():
                    self.input = self.input[:self.cursor] + key + self.input[self.cursor:]
                    self.cursor += 1
        finally:
            self.runner.shutdown()
            self._restore_output()


def main() -> int:
    return curses.wrapper(lambda stdscr: App(stdscr).run())


if __name__ == "__main__":
    raise SystemExit(main())
